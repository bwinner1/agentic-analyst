"""Score systems on the triage examples, all through one path: answer text -> parse_answer -> Label.

Headline measures (higher is better):
- verdict macro-F1 over the four verdicts (balanced classes, so 0.25 accuracy is the floor);
- full match: verdict, shape and place all right (place = start within `tolerance(window)` points);
- attention F1: does it flag a window that needs attention at all.

Intervals resample whole background series (examples cut from one series are not independent).
Paired differences against a reference system use the same resampling, on the same examples.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .labels import VERDICTS, Label, parse_answer

N_BOOT = 1000


def tolerance(window: int) -> int:
    return max(1, window // 10)


def read_predictions(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d.get("text") or "" for d in map(json.loads, filter(str.strip, f))}


def write_predictions(path: Path, answers: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, t in answers.items():
            f.write(json.dumps({"id": i, "text": t}) + "\n")


def judge(row: dict, pred: Label | None) -> dict:
    true = Label.from_dict(row["label"])
    out = {"id": row["id"], "series_key": row["series_key"], "true": true.verdict, "pred": pred.verdict if pred else "unreadable"}
    out["verdict_ok"] = pred is not None and pred.verdict == true.verdict
    out["attention_true"] = true.needs_attention
    out["attention_pred"] = pred is not None and pred.needs_attention
    if true.shape == "none":
        out["shape_ok"] = pred is not None and pred.shape == "none"
        out["place_ok"] = out["shape_ok"]
    else:
        out["shape_ok"] = pred is not None and pred.shape == true.shape
        out["place_ok"] = pred is not None and pred.start is not None and abs(pred.start - true.start) <= tolerance(row["window"])
    out["full_ok"] = out["verdict_ok"] and out["shape_ok"] and out["place_ok"]
    return out


def macro_f1(true: list[str], pred: list[str]) -> float:
    f1s = []
    for v in VERDICTS:
        tp = sum(t == v and p == v for t, p in zip(true, pred))
        fp = sum(t != v and p == v for t, p in zip(true, pred))
        fn = sum(t == v and p != v for t, p in zip(true, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def attention_f1(j: list[dict]) -> float:
    tp = sum(x["attention_true"] and x["attention_pred"] for x in j)
    fp = sum(not x["attention_true"] and x["attention_pred"] for x in j)
    fn = sum(x["attention_true"] and not x["attention_pred"] for x in j)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


MEASURES = {
    "verdict_macro_f1": lambda j: macro_f1([x["true"] for x in j], [x["pred"] for x in j]),
    "verdict_accuracy": lambda j: float(np.mean([x["verdict_ok"] for x in j])),
    "full_match": lambda j: float(np.mean([x["full_ok"] for x in j])),
    "attention_f1": attention_f1,
    "unreadable": lambda j: float(np.mean([x["pred"] == "unreadable" for x in j])),
}


def _clusters(j: list[dict]) -> list[list[dict]]:
    by = defaultdict(list)
    for x in j:
        by[x["series_key"]].append(x)
    return list(by.values())


def bootstrap(j: list[dict], measure, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    clusters = _clusters(j)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(len(clusters), size=len(clusters))
        vals.append(measure([x for i in pick for x in clusters[i]]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired(ja: list[dict], jb: list[dict], measure, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """measure(a) - measure(b) on the same examples, with a series-level bootstrap interval."""
    b_by_id = {x["id"]: x for x in jb}
    pairs = defaultdict(list)
    for x in ja:
        pairs[x["series_key"]].append((x, b_by_id[x["id"]]))
    groups = list(pairs.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        sa = [a for i in pick for a, _ in groups[i]]
        sb = [b for i in pick for _, b in groups[i]]
        diffs.append(measure(sa) - measure(sb))
    return measure(ja) - measure(jb), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def judge_system(rows: list[dict], answers: dict[str, str]) -> list[dict]:
    missing = [r["id"] for r in rows if r["id"] not in answers]
    if missing:
        raise ValueError(f"{len(missing)} examples have no answer, e.g. {missing[:3]}")
    return [judge(r, parse_answer(answers[r["id"]], r["window"])[0]) for r in rows]


def _note_group(kind: str, verdict: str) -> str:
    if kind == "none":
        return "no note"
    if kind in ("irrelevant", "no_effect"):
        return "note unrelated or without effect"
    if kind.startswith("data_"):
        return "data-problem note"
    return "event note, matching" if verdict == "explained_event" else "event note, opposite direction"


def breakdown(rows: list[dict], judged: list[dict], key) -> dict[str, tuple[float, int]]:
    by = defaultdict(list)
    for r, x in zip(rows, judged):
        by[key(r)].append(x["full_ok"])
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(by.items())}


def report(rows: list[dict], systems: dict[str, dict[str, str]], reference: str = "rules") -> str:
    judged = {name: judge_system(rows, ans) for name, ans in systems.items()}
    order = sorted(judged, key=lambda n: -MEASURES["verdict_macro_f1"](judged[n]))
    lines = [
        f"# KPI triage results ({len(rows)} test examples, {len({r['series_key'] for r in rows})} series)",
        "",
        "Verdicts are balanced, so always answering one verdict scores 0.25 accuracy. Higher is better everywhere.",
        "Intervals: 95%, resampling whole series. Full match = verdict, shape and start (within a tenth of the window) all right.",
        "",
        "| System | Verdict macro-F1 | 95% interval | Verdict accuracy | Full match | 95% interval | Attention F1 | Unreadable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in order:
        j = judged[n]
        f_lo, f_hi = bootstrap(j, MEASURES["verdict_macro_f1"])
        m_lo, m_hi = bootstrap(j, MEASURES["full_match"])
        lines.append(
            f"| {n} | {MEASURES['verdict_macro_f1'](j):.3f} | {f_lo:.3f} to {f_hi:.3f} | {MEASURES['verdict_accuracy'](j):.3f} "
            f"| {MEASURES['full_match'](j):.3f} | {m_lo:.3f} to {m_hi:.3f} | {attention_f1(j):.3f} | {MEASURES['unreadable'](j):.0%} |"
        )
    if reference in judged and len(judged) > 1:
        lines += ["", f"## Each system against `{reference}`, example by example", "",
                  "Positive means better than the reference. If the interval includes 0, this test cannot tell them apart.", "",
                  "| System | Macro-F1 difference | 95% interval | Full-match difference | 95% interval |", "|---|---|---|---|---|"]
        for n in order:
            if n == reference:
                continue
            d, lo, hi = paired(judged[n], judged[reference], MEASURES["verdict_macro_f1"])
            d2, lo2, hi2 = paired(judged[n], judged[reference], MEASURES["full_match"])
            lines.append(f"| {n} | {d:+.3f} | {lo:+.3f} to {hi:+.3f} | {d2:+.3f} | {lo2:+.3f} to {hi2:+.3f} |")
    cuts = {
        "note": lambda r: _note_group(r["note_kind"], r["label"]["verdict"]),
        "event injected": lambda r: r["injector"] or "none (normal)",
        "source": lambda r: r["source"],
    }
    for title, key in cuts.items():
        groups = sorted({key(r) for r in rows})
        lines += ["", f"## Full match by {title}", "", "| " + title.capitalize() + " | n | " + " | ".join(order) + " |",
                  "|---|---|" + "---|" * len(order)]
        per = {n: breakdown(rows, judged[n], key) for n in order}
        for g in groups:
            lines.append(f"| {g} | {per[order[0]][g][1]} | " + " | ".join(f"{per[n][g][0]:.2f}" for n in order) + " |")
    best = order[0]
    conf = Counter((x["true"], x["pred"]) for x in judged[best])
    preds = list(VERDICTS) + (["unreadable"] if any(x["pred"] == "unreadable" for x in judged[best]) else [])
    lines += ["", f"## Confusion of `{best}` (rows: true verdict, columns: answered)", "",
              "| True \\ answered | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
    for t in VERDICTS:
        lines.append(f"| {t} | " + " | ".join(str(conf[(t, p)]) for p in preds) + " |")
    return "\n".join(lines) + "\n"
