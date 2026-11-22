"""Turn Context is Key (CiK) rows into OpenTSLM prompt/answer pairs, and score forecasts.

This file has no torch dependency, so it can be tested anywhere.

A CiK row (Hugging Face ServiceNow/context-is-key) holds:
  - past_time / future_time: JSON objects {column: {timestamp: value}}
  - background / scenario / constraints: the text context
  - fields for scoring: metric_scaling, region_of_interest, constraint_*, weight, seasonal_period

CiK ships only a test split (355 rows = 71 task types x 5 seeds). We therefore split by
task type, so all seeds of one task land in the same split. Many task types are variants of
each other on the very same series (e.g. the 7 heat-wave electricity tasks share their data), so
tasks that share a series are kept together too (see `series_groups`); otherwise the model
could see a test answer during training.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

HF_PARQUET = "hf://datasets/ServiceNow/context-is-key/data/test-00000-of-00001.parquet"

SYSTEM_LINE = "You are a forecasting analyst at a consulting firm."


# --------------------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------------------


def load_rows(path: str = HF_PARQUET) -> list[dict[str, Any]]:
    """Read the CiK parquet (local path or hf:// URL) into a list of plain dicts."""
    import pandas as pd

    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
    return rows


TIMENET_REF = "servicenow/context-is-key@1.0.0"


def load_rows_from_timenet(ref: str = TIMENET_REF, registry: str | None = None) -> list[dict[str, Any]]:
    """Read CiK through TimeNet (the connector from PR 788) and return rows in the Hugging Face shape.

    Each TimeNet ForecastingTask holds a history Record (input), a future Record (target), the
    context as its prompt and the scoring fields as metadata. We turn it back into the same
    dict as a Hugging Face row, so everything downstream is shared.

    Args:
        ref: Dataset and version, as `timenet-build build servicenow/context-is-key` makes it.
        registry: Local registry folder. Default: TimeNet's own (~/.timenet/registry).
    """
    import timenet

    client = timenet.TimeNet(sources=[registry]) if registry else timenet.TimeNet()
    dataset = client.load(ref)
    rows = []
    for task in dataset.get_all():
        md = dict(task.metadata or {})
        history, future = task.inputs[0], task.targets[0]
        rows.append({
            "name": md["task_name"],
            "seed": int(md["seed"]),
            "weight": md.get("weight", "1"),
            "context_sources": list(md.get("context_sources") or []),
            "skills": list(md.get("skills") or []),
            "background": md.get("background") or "",
            "scenario": md.get("scenario") or "",
            "constraints": md.get("constraints") or "",
            "seasonal_period": md.get("seasonal_period", -1),
            "past_time": _record_to_frame_json(history),
            "future_time": _record_to_frame_json(future),
            "metric_scaling": md.get("metric_scaling", 1.0),
            "region_of_interest": list(md.get("region_of_interest") or []),
            "constraint_min": md.get("constraint_min"),
            "constraint_max": md.get("constraint_max"),
            "constraint_variable_max_index": list(md.get("constraint_variable_max_index") or []),
            "constraint_variable_max_values": list(md.get("constraint_variable_max_values") or []),
        })
    return rows


def _record_to_frame_json(record) -> str:
    """Encode a TimeNet Record's signals as {signal name: {timestamp: value}} JSON."""
    from datetime import UTC, datetime, timedelta

    origin_us = record.start_time.timestamp or 0
    frame = {}
    for signal in record.signals:
        values = signal.to_numpy().astype(float)
        axis = signal.time_axis
        if hasattr(axis, "period_us"):  # regular axis: offsets follow from the step size
            offsets = [int(axis.offset_us + i * axis.period_us) for i in range(len(values))]
        else:
            offsets = [int(x) for x in signal.time_offsets_us()]
        moments = [datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=origin_us + o) for o in offsets]
        if all(m.hour == m.minute == m.second == m.microsecond == 0 for m in moments):
            fmt_ = "%Y-%m-%d"
        elif all(m.second == m.microsecond == 0 for m in moments):
            fmt_ = "%Y-%m-%d %H:%M"
        else:  # second-level data (the wind tunnel tasks); minutes alone would merge points
            fmt_ = "%Y-%m-%d %H:%M:%S"
        stamps = [m.strftime(fmt_) for m in moments]
        frame[signal.name] = dict(zip(stamps, values.tolist()))
    return json.dumps(frame)


def _series_keys(row: dict[str, Any]) -> list[str]:
    keys = []
    for col in ("past_time", "future_time"):
        raw = row.get(col)
        if not raw:
            continue
        frame = json.loads(raw) if isinstance(raw, str) else raw
        for values in frame.values():
            vals = tuple(round(float(v), 4) for v in values.values() if v is not None and v == v)
            if len(vals) > 3:
                keys.append(repr(vals))
    return keys


def series_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map each task name to a group id. Two tasks are in the same group when any of their
    instances share an identical series (same values in past or future), i.e. they are
    variants of one underlying problem."""
    parent = {r["name"]: r["name"] for r in rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner: dict[str, str] = {}
    for r in rows:
        for key in _series_keys(r):
            if key in owner:
                a, b = find(owner[key]), find(r["name"])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[key] = r["name"]
    return {n: find(n) for n in parent}


def split_by_task(
    rows: list[dict[str, Any]],
    fractions: tuple[float, float, float] = (0.6, 0.1, 0.3),
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows into train / validation / test so that no series appears in two splits.

    Tasks that share a series form one group (`series_groups`); whole groups are dealt out,
    largest first (ties shuffled with a fixed seed), each to the split that is furthest below
    its target share of rows. With real CiK this gives 21 groups and roughly 60/10/30 of the
    355 rows. Test numbers therefore measure how well the model handles new kinds of problems.
    """
    group_of = series_groups(rows)
    size: dict[str, int] = {}
    for r in rows:
        size[group_of[r["name"]]] = size.get(group_of[r["name"]], 0) + 1
    groups = sorted(size)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -size[g])  # stable: shuffled order kept among equal sizes

    total = len(rows)
    targets = {"train": fractions[0] * total, "val": fractions[1] * total, "test": fractions[2] * total}
    filled = {k: 0 for k in targets}
    split_of_group: dict[str, str] = {}
    for g in groups:
        # the split that would be least over-full after taking this group, as a share of its target
        k = min(targets, key=lambda k: (filled[k] + size[g]) / targets[k] if targets[k] else math.inf)
        split_of_group[g] = k
        filled[k] += size[g]

    part = {k: [] for k in targets}
    for r in rows:
        part[split_of_group[group_of[r["name"]]]].append(r)
    return part["train"], part["val"], part["test"]


def task_split_table(rows: list[dict[str, Any]], **kwargs) -> dict[str, list[str]]:
    """Task names per split, for logging and for the write-up."""
    train, val, test = split_by_task(rows, **kwargs)
    return {k: sorted({r["name"] for r in part}) for k, part in (("train", train), ("val", val), ("test", test))}


# --------------------------------------------------------------------------------------
# Building one example
# --------------------------------------------------------------------------------------


@dataclass
class Column:
    name: str
    stamps: list[str]
    values: np.ndarray


def parse_frame(encoded: str) -> list[Column]:
    """Decode {column: {timestamp: value}} JSON, keeping source order."""
    frame = json.loads(encoded)
    return [
        Column(str(name), list(cells.keys()), np.asarray([float(v) for v in cells.values()], dtype=np.float64))
        for name, cells in frame.items()
    ]


def _stats(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    mean = float(finite.mean())
    std = float(finite.std())
    # A nearly flat history (e.g. a fan idling at 288 rpm) would make the std tiny and any
    # change look like +-200 in the normalized scale; keep the scale at least 5% of the level.
    std = max(std, 0.05 * abs(mean))
    return mean, (std if std > 1e-8 else 1.0)


def context_text(row: dict[str, Any]) -> str:
    parts = (row.get("background") or "", row.get("scenario") or "", row.get("constraints") or "")
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


@dataclass
class Example:
    pre_prompt: str
    series_texts: list[str]
    series: list[list[float]]
    post_prompt: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


def fmt(v: float) -> str:
    return f"{v:.2f}"


def build_example(
    row: dict[str, Any],
    *,
    include_context: bool = True,
    rationale: str | None = None,
    max_history: int = 512,
    show_baseline: bool = True,
) -> Example:
    """Turn one CiK row into an OpenTSLM example.

    Every history column becomes one time series input, z-normalized with its own mean/std
    (the mean and std are written in the text, as OpenTSLM's M4 captioning stage does).
    The model answers in the normalized scale of the target's history; `meta` keeps what
    is needed to map the answer back to real units and score it.
    """
    past = parse_frame(row["past_time"])
    future = parse_frame(row["future_time"])
    stats: dict[str, tuple[float, float]] = {}

    series_texts, series = [], []
    for col in past:
        values = col.values[-max_history:]
        stamps = col.stamps[-max_history:]
        values = np.where(np.isfinite(values), values, np.nan)
        mean, std = _stats(values)
        stats[col.name] = (mean, std)
        norm = np.nan_to_num((values - mean) / std, nan=0.0)
        trunc = " (most recent part)" if len(col.values) > max_history else ""
        series_texts.append(
            f"History of '{col.name}'{trunc}: {len(values)} points from {stamps[0]} to {stamps[-1]}, "
            f"mean {mean:.4f}, std {std:.4f}:"
        )
        series.append(norm.astype(np.float32).tolist())

    pre = SYSTEM_LINE
    ctx = context_text(row)
    if include_context and ctx:
        pre += "\n\nContext:\n" + ctx

    targets = []
    lines = []
    for col in future:
        mean, std = stats.get(col.name, _stats(col.values))
        targets.append(
            {"name": col.name, "stamps": col.stamps, "values": col.values.tolist(), "mean": mean, "std": std}
        )
        lines.append(f"'{col.name}': {len(col.values)} values at {col.stamps[0]} to {col.stamps[-1]}")

    ask = "Forecast the next values of " + "; ".join(lines) + "."
    ask += " Give them in the normalized scale of the history after 'Forecast:', one numbered line per step ('1: value')"
    if len(future) > 1:
        ask += " (one 'Forecast <name>:' line per series)"
    ask += "."
    if show_baseline:
        # Give the model the "repeat the last season" forecast to start from. Copying it is easy for an
        # LLM, so the model only has to learn where the context says to change it.
        naive = seasonal_naive({"targets": targets, "seasonal_period": int(row.get("seasonal_period") or -1),
                                "history_last": {c.name: c.values[np.isfinite(c.values)].tolist() for c in past}})
        for t in targets:
            z = (naive[t["name"]] - t["mean"]) / t["std"]
            name = "" if len(targets) == 1 else f" for '{t['name']}'"
            ask += (f"\nBaseline{name} (the last season repeated, normalized), to adjust using the context:\n"
                    + "\n".join(f"{i}: {fmt(v)}" for i, v in enumerate(z, 1)))
        ask += "\n"
    if rationale is not None:
        ask += " Then, after 'Explanation:', explain in 2 to 3 sentences for a business reader what in the context and history drives the forecast."

    answer_lines = []
    for t in targets:
        norm = (np.asarray(t["values"]) - t["mean"]) / t["std"]
        label = "Forecast:" if len(targets) == 1 else f"Forecast {t['name']}:"
        # One numbered line per step, so a small model does not have to count a long comma list.
        answer_lines.append(label + "\n" + "\n".join(f"{i}: {fmt(v)}" for i, v in enumerate(norm, 1)))
    if rationale is not None:
        answer_lines.append("Explanation: " + rationale.strip())

    meta = {
        "task_name": row["name"],
        "seed": int(row["seed"]),
        "weight": str(row.get("weight", "1")),
        "targets": targets,
        "history_last": {c.name: c.values[np.isfinite(c.values)].tolist() for c in past},
        "seasonal_period": int(row.get("seasonal_period") or -1),
        "metric_scaling": float(row.get("metric_scaling") or 1.0),
        "region_of_interest": list(row.get("region_of_interest") or []),
        "constraint_min": _finite_or_none(row.get("constraint_min")),
        "constraint_max": _finite_or_none(row.get("constraint_max")),
        "include_context": include_context,
        "show_baseline": show_baseline,
    }
    return Example(pre, series_texts, series, ask, "\n".join(answer_lines), meta)


def _finite_or_none(v: Any) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------------------
# Parsing model output
# --------------------------------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _longest_number_run(text: str) -> str:
    """The longest stretch of comma/space separated numbers, for outputs that skip the 'Forecast:' label."""
    runs = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*[,;]\s*-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)+", text)
    return max(runs, key=lambda r: len(_NUM.findall(r)), default="")


def parse_answer(text: str, meta: dict[str, Any]) -> tuple[dict[str, np.ndarray], bool, str]:
    """Read forecasts (in real units) and the explanation from generated text.

    Expects the numbered format ("Forecast:\\n1: 0.12\\n2: -0.40 ...") and also accepts a plain
    comma separated list. Returns (forecast per target column, whether every step 1..horizon was
    found, explanation). Missing steps are filled from their neighbours (or the history mean when
    nothing was found), so a score can always be computed; `valid` reports the problem.
    """
    explanation = ""
    m = re.search(r"Explanation:\s*(.*)", text, flags=re.S)
    if m:
        explanation = m.group(1).strip()
        text = text[: m.start()]

    targets = meta["targets"]
    if len(targets) == 1:
        m = re.search(r"Forecast\s*:(.*)", text, flags=re.S)
        sections = [m.group(1) if m else text]
    else:
        heads = list(re.finditer(r"Forecast\s*([^:\n]*):", text))
        by_name = {h.group(1).strip().strip("'\""): text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
                   for i, h in enumerate(heads)}
        ordered = list(by_name.values())
        sections = [by_name.get(t["name"], ordered[i] if i < len(ordered) else "") for i, t in enumerate(targets)]

    out: dict[str, np.ndarray] = {}
    valid = True
    for t, sec in zip(targets, sections):
        h = len(t["values"])
        vals, ok = _read_steps(sec, h)
        valid &= ok
        # Guard against a stray huge number (e.g. the model writing rpm instead of the normalized value).
        vals = np.clip(vals, -25.0, 25.0)
        out[t["name"]] = vals * t["std"] + t["mean"]
    return out, valid, explanation


_STEP = re.compile(r"^\s*(\d+)\s*[:.)]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", flags=re.M)


def _read_steps(section: str, h: int) -> tuple[np.ndarray, bool]:
    """Values for steps 1..h (normalized scale) and whether all of them were present exactly once."""
    steps = {}
    for idx, val in _STEP.findall(section):
        i = int(idx)
        if 1 <= i <= h and i not in steps:
            steps[i] = float(val)
    if steps:
        ok = len(steps) == h
        known = np.array(sorted(steps))
        vals = np.interp(np.arange(1, h + 1), known, [steps[i] for i in known])
        return vals, ok
    # fallback: a plain comma separated list
    nums = [float(x) for x in _NUM.findall(_longest_number_run(section) or section)]
    ok = len(nums) == h
    nums = nums[:h] or [0.0]
    nums = nums + [nums[-1]] * (h - len(nums))
    return np.asarray(nums, float), ok


# --------------------------------------------------------------------------------------
# Baselines and metrics
# --------------------------------------------------------------------------------------


def seasonal_naive(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """Repeat the last season of the history (or the last value when there is no season)."""
    out = {}
    for t in meta["targets"]:
        hist = np.asarray(meta["history_last"].get(t["name"]) or [t["mean"]])
        h = len(t["values"])
        p = meta["seasonal_period"]
        if p and 1 < p <= len(hist):
            season = hist[-p:]
            out[t["name"]] = np.resize(season, h)
        else:
            out[t["name"]] = np.full(h, hist[-1])
    return out


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-step CRPS from forecast samples [n, h] and truth [h] (energy form)."""
    samples = np.atleast_2d(samples)
    term1 = np.abs(samples - y[None, :]).mean(axis=0)
    term2 = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    return term1 - 0.5 * term2


def score_instance(meta: dict[str, Any], samples: dict[str, np.ndarray]) -> dict[str, float]:
    """Score one instance. `samples[col]` is [n_samples, horizon] in real units.

    - mae: MAE of the median forecast, times CiK's metric_scaling (so tasks are comparable)
    - crps: sample CRPS times metric_scaling, with the region of interest weighted to half
      the total like CiK's RCRPS (an approximation; use CiK's own script for official numbers)
    - violations: share of forecast steps outside the stated constraint bounds
    """
    maes, crpss, viol, steps = [], [], 0, 0
    roi = set(meta["region_of_interest"])
    for t in meta["targets"]:
        y = np.asarray(t["values"], dtype=float)
        s = np.atleast_2d(samples[t["name"]])
        med = np.median(s, axis=0)
        maes.append(np.abs(med - y).mean())
        c = crps_from_samples(s, y)
        if roi and len(roi) < len(y):
            in_roi = np.array([i in roi for i in range(len(y))])
            crpss.append(0.5 * c[in_roi].mean() + 0.5 * c[~in_roi].mean())
        else:
            crpss.append(c.mean())
        lo, hi = meta["constraint_min"], meta["constraint_max"]
        if lo is not None:
            viol += int((med < lo - 1e-9).sum())
        if hi is not None:
            viol += int((med > hi + 1e-9).sum())
        steps += len(y)
    k = meta["metric_scaling"]
    return {"mae": float(np.mean(maes) * k), "crps": float(np.mean(crpss) * k), "violations": viol / max(steps, 1)}


def aggregate(results: Iterable[tuple[dict[str, Any], dict[str, float]]]) -> dict[str, float]:
    """Weighted mean over instances, using CiK's per-instance `weight` (e.g. '1/8')."""
    totals: dict[str, float] = {}
    wsum = 0.0
    n = 0
    for meta, scores in results:
        w = float(Fraction(meta["weight"])) if meta.get("weight") else 1.0
        wsum += w
        n += 1
        for k, v in scores.items():
            totals[k] = totals.get(k, 0.0) + w * v
    out = {k: v / wsum for k, v in totals.items()} if wsum else {}
    out["n_instances"] = n
    return out


def load_rationales(path: str | None) -> dict[tuple[str, int], str]:
    """Read rationales JSONL with fields task_name, seed, rationale."""
    if not path:
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[(obj["task_name"], int(obj["seed"]))] = obj["rationale"]
    return out
