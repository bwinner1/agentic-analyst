"""Command line for KPI triage.

    python -m kpi_triage generate                 # examples -> data/kpi_triage/generated/{train,val,test}.jsonl
    python -m kpi_triage baselines                # rules, series-only, note-only, majority -> evaluation/kpi_triage/runs/
    python -m kpi_triage score                    # every runs/*.jsonl -> evaluation/kpi_triage/results.md
    python -m kpi_triage show test-00003          # one example as the text prompt a model sees, plus its answer
    python -m kpi_triage tune-rules               # grid-search the rule thresholds on training examples
    python -m kpi_triage combine opentslm-finetuned   # rules find the event, that model's verdict reads the note

Model runs (OpenTSLM, text-only Llama) happen on a GPU with opentslm-triage/; copy their
predictions.jsonl into the runs folder under a system name and score again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import backgrounds, baselines, generate, prompting, score

DATA = Path("data/kpi_triage/generated")
RUNS = Path("evaluation/kpi_triage/runs")
OUT = Path("evaluation/kpi_triage")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kpi_triage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, default=DATA)
    g.add_argument("--seed", type=int, default=0)
    for split, n in generate.DEFAULT_SIZES.items():
        g.add_argument(f"--{split}", type=int, default=n)
    b = sub.add_parser("baselines")
    b.add_argument("--data", type=Path, default=DATA)
    b.add_argument("--split", default="test")
    b.add_argument("--runs", type=Path, default=RUNS)
    s = sub.add_parser("score")
    s.add_argument("--data", type=Path, default=DATA)
    s.add_argument("--split", default="test")
    s.add_argument("--runs", type=Path, default=RUNS)
    s.add_argument("--out", type=Path, default=OUT)
    s.add_argument("--reference", default="rules")
    t = sub.add_parser("tune-rules")
    t.add_argument("--data", type=Path, default=DATA)
    t.add_argument("--n", type=int, default=1000, help="training examples used (a fixed random sample)")
    c = sub.add_parser("combine")
    c.add_argument("model", help="a system in the runs folder whose verdicts read the note")
    c.add_argument("--data", type=Path, default=DATA)
    c.add_argument("--runs", type=Path, default=RUNS)
    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args(argv)

    if args.cmd == "generate":
        rows = generate.generate(backgrounds.load_all(), {"train": args.train, "val": args.val, "test": args.test}, args.seed)
        generate.write(rows, args.out)
        for split, r in rows.items():
            print(f"[{split}] " + generate.summary(r))
        print(f"wrote {args.out}")
    elif args.cmd == "baselines":
        rows = generate.read(args.data / f"{args.split}.jsonl")
        train = generate.read(args.data / "train.jsonl")
        for name, system in baselines.SYSTEMS.items():
            path = args.runs / f"{name}.jsonl"
            score.write_predictions(path, system(rows, train))
            print(f"wrote {path}")
    elif args.cmd == "score":
        rows = generate.read(args.data / f"{args.split}.jsonl")
        systems = {f.stem: score.read_predictions(f) for f in sorted(args.runs.glob("*.jsonl"))}
        text = score.report(rows, systems, args.reference)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.md").write_text(text, encoding="utf-8")
        print(text)
    elif args.cmd == "tune-rules":
        import random

        train = generate.read(args.data / "train.jsonl")
        sample = random.Random(0).sample(train, min(args.n, len(train)))
        results = baselines.tune(sample)
        for acc, th in results[:10]:
            print(f"{acc:.3f}  spike={th.spike} span={th.span} level={th.level}")
        print(f"current TUNED: {baselines.TUNED}")
    elif args.cmd == "combine":
        rows = generate.read(args.data / "test.jsonl")
        answers = score.read_predictions(args.runs / f"{args.model}.jsonl")
        path = args.runs / f"rules+{args.model}.jsonl"
        score.write_predictions(path, baselines.code_finds_model_reads(rows, answers))
        print(f"wrote {path}")
    elif args.cmd == "show":
        split = args.id.split("-")[0]
        row = next(r for r in generate.read(args.data / f"{split}.jsonl") if r["id"] == args.id)
        prompt, answer = prompting.text_prompt(row)
        print(prompt, answer, sep="\n")


if __name__ == "__main__":
    main()
