"""Score model predictions on the business question benchmark.

    python business-qa/score.py business-qa/results/llama-3.2-1b.jsonl [more prediction files...]

A prediction file has one JSON object per line: {"id": "...", "reply": "<the model's raw text>"}.
Reports accuracy and macro-F1 per question type and overall, the share of replies with no readable letter,
and two reference rows: random guessing and always answering the most common letter.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from prompts import parse_letter

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> dict[str, dict]:
    return {q["id"]: q for q in map(json.loads, open(path))}


def macro_f1(pairs: list[tuple[str, str | None]]) -> float:
    """Macro-F1 over the classes A, B, C. An unreadable reply counts as wrong for every class."""
    scores = []
    for c in "ABC":
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / 3


def score(questions: dict[str, dict], preds: dict[str, str | None]) -> dict:
    rows = {}
    for t in [*sorted({q["type"] for q in questions.values()}), "all"]:
        qs = [q for q in questions.values() if t in ("all", q["type"])]
        pairs = [(q["answer"], preds.get(q["id"])) for q in qs]
        rows[t] = {"n": len(pairs), "accuracy": sum(g == p for g, p in pairs) / len(pairs),
                   "macro_f1": macro_f1(pairs), "unreadable": sum(p is None for _, p in pairs) / len(pairs)}
    return rows


def reference_rows(questions: dict[str, dict]) -> dict[str, dict]:
    common = Counter(q["answer"] for q in questions.values()).most_common(1)[0][0]
    return {"random guess (expected)": {"accuracy": 1 / 3},
            f"always '{common}'": score(questions, {k: common for k in questions})["all"]}


def main(paths: list[str]) -> None:
    questions = load_questions()
    print(f"{len(questions)} questions\n")
    for name, row in reference_rows(questions).items():
        print(f"{name:28s} accuracy {row['accuracy']:.3f}")
    for path in paths:
        preds = {}
        for rec in map(json.loads, open(path)):
            preds[rec["id"]] = parse_letter(rec["reply"], questions.get(rec["id"]))
        missing = len(set(questions) - set(preds))
        print(f"\n{path}" + (f"  ({missing} questions have no prediction; counted as wrong)" if missing else ""))
        for t, row in score(questions, preds).items():
            print(f"  {t:9s} n={row['n']:4d}  accuracy {row['accuracy']:.3f}  macro-F1 {row['macro_f1']:.3f}  "
                  f"unreadable {row['unreadable']:.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
