"""Command line: run from the repository root.

    python -m tslm_eval fetch                 download the pinned CiK file (sha256-checked)
    python -m tslm_eval split                 write the proposed held-out test set
    python -m tslm_eval baselines             naive and seasonal-naive prediction files
    python -m tslm_eval export --split test   model inputs (no answers) for another runner, e.g. OpenTSLM
    python -m tslm_eval claude --limit 3      Claude run; prints a cost estimate, needs --yes to spend
    python -m tslm_eval report                score every file in evaluation/runs, write evaluation/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import baselines, cik, splits
from .forecasts import write_predictions

RUNS = Path("evaluation/runs")
RESULTS = Path("evaluation/results")
INPUTS = Path("evaluation/inputs")
DEFAULT_EXAMPLE = "ATMBuildingClosedTask/1"


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
    instances = _instances(args)
    ids = splits.group_holdout(instances, splits.PROPOSED_TEST_GROUPS)
    tasks = {i.task for i in instances if i.id in ids}
    weight = sum(i.weight for i in instances if i.id in ids) / sum(i.weight for i in instances)
    header = (f"Held-out test set: whole scenario groups {', '.join(splits.PROPOSED_TEST_GROUPS)}.\n"
              f"{len(ids)} instances, {len(tasks)} tasks, {weight:.0%} of the CiK task weight.\n"
              "Do not train on these. Regenerate with: python -m tslm_eval split")
    splits.write_ids(Path(args.split_file), ids, header)
    print(f"wrote {args.split_file}: {len(ids)} instances from {len(tasks)} tasks ({weight:.0%} of weight)")


def cmd_baselines(args):
    instances = _instances(args)
    for name, fn in baselines.BASELINES.items():
        rows = [{"id": i.id, "forecast": [float(v) for v in fn(i)]} for i in instances]
        write_predictions(RUNS / f"{name}.jsonl", rows)
        print(f"wrote {RUNS / f'{name}.jsonl'}")


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
    label = (f"scenario groups {', '.join(splits.PROPOSED_TEST_GROUPS)} "
             f"({len(test_ids)} instances)") if test_ids else ""
    path = write_report(Path(args.runs), Path(args.out), instances, test_ids, label,
                        example_id=args.example, highlight=args.highlight)
    print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tslm_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(cik.DEFAULT_PATH))
    p.add_argument("--split-file", default=str(splits.PROPOSED_SPLIT_FILE))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("split").set_defaults(fn=cmd_split)
    sub.add_parser("baselines").set_defaults(fn=cmd_baselines)
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
    r.add_argument("--example", default=DEFAULT_EXAMPLE)
    r.add_argument("--highlight", default=None, help="system to colour as ours (default: name with 'finetuned')")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
