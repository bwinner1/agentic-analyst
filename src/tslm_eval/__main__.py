"""Command line: run from the repository root.

    python -m tslm_eval fetch                 download the pinned CiK file (sha256-checked)
    python -m tslm_eval split                 write the test set (the instances OpenTSLM was held out from)
    python -m tslm_eval import-opentslm opentslm-cik/results/full --name opentslm-finetuned-cik
                                              convert an OpenTSLM eval_cik.py run into a prediction file
    python -m tslm_eval baselines             naive and seasonal-naive prediction files
    python -m tslm_eval baselines --samples   baselines with 5 forecasts each, for the uncertainty comparison
    python -m tslm_eval export --split test   model inputs (no answers) for another runner, e.g. OpenTSLM
    python -m tslm_eval claude --limit 3      Claude run; prints a cost estimate, needs --yes to spend
    python -m tslm_eval report                score every file in evaluation/runs, write evaluation/results
    python -m tslm_eval report-uncertainty    score the sample sets in evaluation/runs-samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import baselines, cik, splits
from .forecasts import write_predictions

RUNS = Path("evaluation/runs")
SAMPLE_RUNS = Path("evaluation/runs-samples")  # several forecasts per instance, for comparing ranges
RESULTS = Path("evaluation/results")
INPUTS = Path("evaluation/inputs")
OPENTSLM_MAIN_RUN = "opentslm-cik/results/full"
# Test-set instances: one where the note helped the fine-tuned model, one where it went badly wrong.
DEFAULT_EXAMPLES = ["ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow/4", "DirectNormalIrradianceFromClearsky/5"]


def _instances(args):
    return cik.load(cik.fetch(Path(args.data)))


def _select(instances, split: str, split_file: Path):
    if split == "all":
        return instances
    ids = splits.read_ids(split_file)
    chosen = [i for i in instances if i.id in ids]
    return chosen if split == "test" else [i for i in instances if i.id not in ids]


def cmd_fetch(args):
    path = cik.fetch(Path(args.data))
    print(f"{path}: {len(cik.load(path))} instances, sha256 ok")


def cmd_split(args):
    from . import opentslm

    instances = _instances(args)
    known = {i.id for i in instances}
    if args.groups:
        groups = tuple(g.strip() for g in args.groups.split(","))
        ids = splits.group_holdout(instances, groups)
        what = f"task types in scenario groups {', '.join(groups)}"
        source = f"python -m tslm_eval split --groups {args.groups}"
    else:
        ids = set(opentslm.evaluated_ids(Path(args.from_opentslm)))
        unknown = ids - known
        if unknown:
            raise SystemExit(f"{len(unknown)} ids in {args.from_opentslm} are not CiK instances, e.g. {sorted(unknown)[0]}")
        what = "task types OpenTSLM never saw in training"
        source = f"python -m tslm_eval split --from-opentslm {args.from_opentslm}"
    tasks = {i.task for i in instances if i.id in ids}
    weight = sum(i.weight for i in instances if i.id in ids) / sum(i.weight for i in instances)
    header = (f"{len(tasks)} {what} ({len(ids)} instances, {weight:.0%} of CiK task weight)\n"
              f"Regenerate with: {source}")
    splits.write_ids(Path(args.split_file), ids, header)
    print(f"wrote {args.split_file}: {len(ids)} instances from {len(tasks)} tasks ({weight:.0%} of weight)")


def cmd_import_opentslm(args):
    from . import opentslm

    runs = Path(args.runs) if args.runs else (SAMPLE_RUNS if args.samples else RUNS)
    out = runs / f"{args.name}.jsonl"
    ids = opentslm.import_run(Path(args.results_dir), out, _instances(args), samples=args.samples)
    print(f"wrote {out} and {out.with_suffix('.meta.json')}: {len(ids)} instances from {args.results_dir}")


def cmd_baselines(args):
    instances = _instances(args)
    if args.samples:
        # The single-forecast seasonal naive goes alongside, to show what a range alone is worth.
        runs, chosen = SAMPLE_RUNS, {**baselines.SAMPLED_BASELINES, "seasonal-naive": baselines.seasonal_naive}
    else:
        runs, chosen = RUNS, baselines.BASELINES
    for name, fn in chosen.items():
        rows = []
        for i in instances:
            f = fn(i)
            rows.append({"id": i.id, "samples" if f.ndim == 2 else "forecast": f.tolist()})
        write_predictions(runs / f"{name}.jsonl", rows)
        print(f"wrote {runs / f'{name}.jsonl'}")


def cmd_export(args):
    instances = _select(_instances(args), args.split, Path(args.split_file))
    suffix = "-no-context" if args.no_context else ""
    out = INPUTS / f"cik-{args.split}-inputs{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with_targets = args.split == "train"  # training needs the answers; test and all never get them
    with out.open("w", encoding="utf-8") as f:
        for i in instances:
            rec = {
                "id": i.id, "task": i.task, "seed": i.seed, "context": "" if args.no_context else i.context_text(),
                "past_times": list(i.past_times), "past": [float(v) for v in i.past],
                "future_times": list(i.future_times), "horizon": i.horizon,
            }
            if with_targets:
                rec["future"] = [float(v) for v in i.future]
            f.write(json.dumps(rec) + "\n")
    what = "with future values (training targets)" if with_targets else "inputs only, no future values"
    print(f"wrote {out}: {len(instances)} instances ({what})")


def cmd_claude(args):
    from . import claude_runner

    instances = _select(_instances(args), args.split, Path(args.split_file))
    if args.limit:
        instances = instances[: args.limit]
    name = args.name or (args.model + ("" if not args.no_context else "-no-context"))
    out = RUNS / f"{name}.jsonl"
    estimate = claude_runner.estimate_cost(len(instances), args.model)
    print(f"{len(instances)} instances -> {out} with {args.model}, effort {args.effort}, "
          f"context {'withheld' if args.no_context else 'given'}. Rough upper cost estimate: ${estimate:.2f} "
          "(instances already in the file are skipped).")
    if not args.yes:
        print("Not run. Add --yes to call the API.")
        return
    totals = claude_runner.run(instances, out, with_context=not args.no_context, model=args.model,
                               effort=args.effort, workers=args.workers)
    print(f"done {totals['done']}, failed {totals['failed']} (rerun to retry), "
          f"tokens in {totals['input_tokens']:,} / out {totals['output_tokens']:,}, cost ${totals['cost_usd']:.2f}")


def cmd_report(args):
    from .report import write_report

    instances = _instances(args)
    split_file = Path(args.split_file)
    test_ids = splits.read_ids(split_file) if split_file.exists() else None
    label = splits.read_label(split_file) if test_ids else ""
    path = write_report(Path(args.runs), Path(args.out), instances, test_ids, label,
                        examples=args.example, highlight=args.highlight)
    print(f"wrote {path}")


def cmd_report_uncertainty(args):
    from .report import write_uncertainty_report

    split_file = Path(args.split_file)
    test_ids = splits.read_ids(split_file)
    path = write_uncertainty_report(Path(args.runs), Path(args.out), _instances(args), test_ids,
                                    splits.read_label(split_file), ours=args.ours)
    print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tslm_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(cik.DEFAULT_PATH))
    p.add_argument("--split-file", default=str(splits.TEST_SPLIT_FILE))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    s = sub.add_parser("split", help="write the test-set id file")
    s.add_argument("--from-opentslm", default=OPENTSLM_MAIN_RUN,
                   help="an eval_cik.py results folder: its test instances become the test set")
    s.add_argument("--groups", default="", help="instead, hold out whole scenario groups, e.g. atm,unemployment")
    s.set_defaults(fn=cmd_split)
    i = sub.add_parser("import-opentslm", help="convert an eval_cik.py results folder into a prediction file")
    i.add_argument("results_dir", help="folder with test_predictions.jsonl, e.g. opentslm-cik/results/full")
    i.add_argument("--name", required=True, help="system name, e.g. opentslm-finetuned-cik")
    i.add_argument("--samples", action="store_true",
                   help=f"import every readable sample, not the median (default folder: {SAMPLE_RUNS})")
    i.add_argument("--runs", default="", help=f"output folder (default: {RUNS})")
    i.set_defaults(fn=cmd_import_opentslm)
    b = sub.add_parser("baselines")
    b.add_argument("--samples", action="store_true", help=f"baselines with 5 forecasts each, into {SAMPLE_RUNS}")
    b.set_defaults(fn=cmd_baselines)
    e = sub.add_parser("export")
    e.add_argument("--split", choices=["all", "test", "train"], default="test")
    e.add_argument("--no-context", action="store_true", help="blank the text (does the model use it?)")
    e.set_defaults(fn=cmd_export)
    c = sub.add_parser("claude")
    c.add_argument("--split", choices=["all", "test", "train"], default="all")
    c.add_argument("--no-context", action="store_true", help="withhold the text (ablation)")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--model", default="claude-opus-5-5")
    c.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--name", default="", help="prediction file name (default: model id)")
    c.add_argument("--yes", action="store_true", help="actually call the API")
    c.set_defaults(fn=cmd_claude)
    r = sub.add_parser("report")
    r.add_argument("--runs", default=str(RUNS))
    r.add_argument("--out", default=str(RESULTS))
    r.add_argument("--example", nargs="*", default=DEFAULT_EXAMPLES,
                   help="instance ids (Task/seed) to chart; the first becomes chart_example.png")
    r.add_argument("--highlight", default=None, help="system to colour as ours (default: name with 'finetuned')")
    r.set_defaults(fn=cmd_report)
    u = sub.add_parser("report-uncertainty", help="compare forecast ranges on the held-out test set")
    u.add_argument("--runs", default=str(SAMPLE_RUNS))
    u.add_argument("--out", default=str(RESULTS))
    u.add_argument("--ours", default=None, help="our model's system name (default: name with 'finetuned')")
    u.set_defaults(fn=cmd_report_uncertainty)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
