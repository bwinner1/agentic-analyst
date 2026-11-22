"""Make extra Context is Key training examples with CiK's own task generator.

Three steps (run_nuvolos.sh `extra` does all of them):

  1. plan (training venv):   python cik_extra.py plan --out extra/plan.json
     Lists the training task types (from our split, so no test or validation task type is used) and
     fingerprints every series in the test and validation instances.
  2. generate (CiK venv):    python cik_extra.py generate --plan extra/plan.json --cik_repo <repo> --out extra/rows.parquet
     Runs each training task type with new seeds (1000+). An instance is dropped if any stretch of
     its data matches a test/validation series (leak check). Prefers instances whose future moves
     away from the "repeat the last season" baseline, so copying the baseline isn't rewarded.
  3. rationales (training venv): python build_rationales.py extra/rows.parquet extra/rationales.jsonl

train_cik.py --extra_train extra/rows.parquet --extra_rationales extra/rationales.jsonl then adds them
to the training split only. Validation and test stay exactly the CiK benchmark instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

WINDOW = 8  # length of the value stretches compared in the leak check


# --------------------------------------------------------------------------------------
# Fingerprints (shared by both steps; numpy only)
# --------------------------------------------------------------------------------------


def _frame_series(raw) -> list[np.ndarray]:
    frame = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for values in frame.values():
        vals = np.array([float(v) for v in values.values() if v is not None], float)
        out.append(vals[np.isfinite(vals)])
    return out


def fingerprints(values: np.ndarray) -> set[str]:
    """Hashes of every WINDOW-long stretch that is not flat (flat stretches like night-time zeros match everything)."""
    out = set()
    r = np.round(values, 3)
    for i in range(len(r) - WINDOW + 1):
        w = r[i: i + WINDOW]
        if len(np.unique(w)) >= 3:
            out.add(hashlib.md5(w.tobytes()).hexdigest()[:16])
    return out


def row_fingerprints(row) -> set[str]:
    fp = set()
    for col in ("past_time", "future_time"):
        for vals in _frame_series(row[col]):
            fp |= fingerprints(vals)
    return fp


# --------------------------------------------------------------------------------------
# Step 1: plan
# --------------------------------------------------------------------------------------


def plan(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from opentslm.time_series_datasets.cik import cik_core
    except ImportError:
        import cik_core
    if args.data:
        rows = cik_core.load_rows(args.data)
    else:
        rows = cik_core.load_rows_from_timenet(args.timenet_ref, args.timenet_registry)
    train, val, test = cik_core.split_by_task(rows)
    group_of = cik_core.series_groups(rows)
    groups: dict[str, list[str]] = {}
    for name in sorted({r["name"] for r in train}):
        groups.setdefault(group_of[name], []).append(name)
    fp = set()
    for r in val + test:
        fp |= row_fingerprints(r)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"groups": groups, "heldout_fingerprints": sorted(fp),
                   "heldout_tasks": sorted({r["name"] for r in val + test})}, f)
    print(f"Plan: {sum(map(len, groups.values()))} training task types in {len(groups)} groups; "
          f"{len(fp)} fingerprints from {len(val) + len(test)} validation/test instances -> {args.out}")


# --------------------------------------------------------------------------------------
# Step 2: generate (runs in a separate environment that has CiK's dependencies)
# --------------------------------------------------------------------------------------


def _install_stubs():
    """CiK's package imports its baseline models (lag-llama, chronos, ...) at import time. We only need the
    task generators, so missing baseline packages are replaced by empty stand-ins."""
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import types

    names = ("lag_llama", "tactis", "chronos", "uni2ts", "timellm", "unitime", "llm_processes",
             "lmformatenforcer", "openai", "nixtla", "gradio", "transformers", "torch", "sentencepiece")
    active = {n for n in names if importlib.util.find_spec(n) is None}

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return _Dummy()

        def __getattr__(self, n):
            return _Dummy()

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in active:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda n: _Dummy
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())


def _task_to_row(task, name, seed):
    import pandas as pd

    def frame_json(df, all_columns=False):
        # the benchmark's published rows keep only the target column (the last one)
        df = df.copy() if all_columns else df.iloc[:, -1:].copy()
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp()
        df.columns = [str(c) for c in df.columns]
        return df.to_json(date_format="iso", date_unit="ms")

    background = task.background or ""
    causal = getattr(task, "causal_context", None)
    if causal:
        background = (background + "\n" + causal).strip()
    roi = getattr(task, "region_of_interest", None)
    if roi is None or isinstance(roi, slice):
        roi = []
    try:
        sp = int(task.seasonal_period)
    except Exception:
        sp = -1
    return {
        "name": name, "seed": int(seed), "weight": "1", "context_sources": [], "skills": [],
        "background": background, "scenario": task.scenario or "", "constraints": task.constraints or "",
        "seasonal_period": sp,
        "past_time": frame_json(task.past_time), "future_time": frame_json(task.future_time),
        "metric_scaling": 1.0, "region_of_interest": [int(i) for i in np.atleast_1d(roi)],
        "constraint_min": None, "constraint_max": None,
        "constraint_variable_max_index": [], "constraint_variable_max_values": [],
        # every column (covariates too), only for the leak check; dropped before saving
        "_all_past": frame_json(task.past_time, True), "_all_future": frame_json(task.future_time, True),
    }


def _departure(row) -> float:
    """How far the true future is from 'repeat the last season', in units of the history's spread."""
    past = json.loads(row["past_time"])
    fut = json.loads(row["future_time"])
    col = list(fut)[-1]
    h = np.array(list(past.get(col, list(past.values())[-1]).values()), float)
    y = np.array(list(fut[col].values()), float)
    h, y = h[np.isfinite(h)], y[np.isfinite(y)]
    if len(h) == 0 or len(y) == 0:
        return 0.0
    p = row["seasonal_period"]
    base = np.resize(h[-p:], len(y)) if p and 1 < p <= len(h) else np.full(len(y), h[-1])
    scale = max(h.std(), 0.05 * abs(h.mean()), 1e-6)
    return float(np.mean(np.abs(y - base)) / scale)


def generate(args):
    import pandas as pd

    _install_stubs()
    sys.path.insert(0, os.path.abspath(args.cik_repo))
    import warnings

    warnings.filterwarnings("ignore")
    import cik_benchmark

    classes = {t.__name__: t for t in cik_benchmark.ALL_TASKS}
    with open(args.plan) as f:
        p = json.load(f)
    heldout_fp = set(p["heldout_fingerprints"])
    heldout_tasks = set(p["heldout_tasks"])
    groups = p["groups"]
    n_tasks = sum(len(v) for v in groups.values())
    budget = args.minutes * 60 / max(1, n_tasks)
    print(f"Generating about {args.per_group} examples per group for {len(groups)} groups "
          f"({n_tasks} task types, up to {budget:.0f}s each)")

    rows, report = [], []
    for gid, names in groups.items():
        want = math.ceil(args.per_group / len(names))
        for name in names:
            assert name not in heldout_tasks, name  # never generate a validation/test task type
            cls = classes.get(name)
            if cls is None:
                report.append((name, 0, "not in CiK repo"))
                continue
            cands, leaks, errors, dups, seed, t0 = [], 0, 0, 0, args.first_seed, time.time()
            seen = set()
            seen_first = False
            while len(cands) < want * args.oversample and time.time() - t0 < budget and seed < args.first_seed + 50 * want:
                seed += 1
                try:
                    task = cls(seed=seed)
                    row = _task_to_row(task, name, seed)
                except Exception as exc:  # a download failure or an unlucky random window
                    errors += 1
                    if errors in (1, 10):
                        print(f"  {name} seed {seed}: {type(exc).__name__}: {str(exc)[:150]}")
                    if errors >= 25 and not cands:
                        break
                    continue
                full = {"past_time": row.pop("_all_past"), "future_time": row.pop("_all_future")}
                if row_fingerprints(full) & heldout_fp:
                    leaks += 1
                    continue
                key = hashlib.md5((row["past_time"] + row["future_time"]).encode()).hexdigest()
                if key in seen:  # small datasets (e.g. Montreal) can give the same window twice
                    dups += 1
                    continue
                seen.add(key)
                cands.append((_departure(row), row))
                if len(cands) == 1 and seen_first is False:
                    t0, seen_first = time.time(), True  # the first instance includes the data download; don't count it
            # keep the instances that move most from the baseline (70%), plus a random 30% of the rest
            cands.sort(key=lambda c: -c[0])
            n_top = int(round(0.7 * want))
            keep = cands[:n_top]
            rest = cands[n_top:]
            rng = np.random.RandomState(0)
            if rest:
                idx = rng.permutation(len(rest))[: max(0, want - n_top)]
                keep += [rest[i] for i in idx]
            rows += [r for _, r in keep]
            report.append((name, len(keep), f"{leaks} dropped as overlapping test data, {dups} duplicates, {errors} errors, "
                                            f"{time.time() - t0:.0f}s"))
            print(f"  {name}: kept {len(keep)} ({report[-1][2]})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"Wrote {len(rows)} extra training examples to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--out", required=True)
    a.add_argument("--data", default=None, help="CiK parquet instead of TimeNet")
    a.add_argument("--timenet_ref", default="servicenow/context-is-key@1.0.0")
    a.add_argument("--timenet_registry", default=None)
    g = sub.add_parser("generate")
    g.add_argument("--plan", required=True)
    g.add_argument("--cik_repo", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--per_group", type=int, default=120, help="Extra examples per group of related task types.")
    g.add_argument("--oversample", type=float, default=1.5, help="Make this many times more, keep the most informative.")
    g.add_argument("--first_seed", type=int, default=1000, help="Seeds start above this (the benchmark uses 1-5).")
    g.add_argument("--minutes", type=float, default=30, help="Total time budget for generating.")
    args = ap.parse_args()
    plan(args) if args.cmd == "plan" else generate(args)


if __name__ == "__main__":
    main()
