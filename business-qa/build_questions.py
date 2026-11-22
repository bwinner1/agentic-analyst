"""Build the business question benchmark: multiple-choice questions about US Census retail series.

Same format as TSQA in the OpenTSLM paper (a series, a question, three options, one right answer), but the
questions are the ones a consulting partner asks. Every answer is computed by code from the data, through the
same measures the analyst engine checks, so the answer key is correct by construction.

    PYTHONPATH=src python business-qa/build_questions.py

Writes business-qa/questions.jsonl. Question types (three options each):
  seasonal  Is this month's change normal for the season, weaker than every reference year, or stronger?
  market    Over the year, did this store type grow faster than, in line with, or slower than total retail?
  trend     Against the same month last year, are sales up more than 2%, down more than 2%, or within 2%?
  (A 'peak month' type was dropped: the answer is December in 47 of 48 store-years, a giveaway.)
Classes are balanced within each type by sampling, so always picking one answer scores about a third.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from agentic_analyst import load_census_retail
from agentic_analyst.answer_types.in_context import (
    REFERENCE_YEARS, _shown, market_verdict, month_change, seasonal_verdict, year_on_year)
from agentic_analyst.series import MONTH_NAMES
from agentic_analyst.sources.census_retail import CATEGORIES, MARKET_NAME

OUT = Path(__file__).parent / "questions.jsonl"
SEED = 20261122
TREND_BAND = 2.0  # percent


def _series(s, start: int, end: int) -> dict:
    return {"name": s.name, "unit": s.unit, "dates": [f"{d:%Y-%m}" for d in s.dates[start:end + 1]],
            "values": [round(v) for v in s.values[start:end + 1]]}


def _shuffle(item_id: str, correct: str, wrong: list[str]) -> tuple[dict, str]:
    """Deterministic option order per question, so the right answer is not always in one position."""
    opts = [correct, *wrong]
    random.Random(int(hashlib.sha256(item_id.encode()).hexdigest(), 16)).shuffle(opts)
    letters = "ABC"
    return {letters[k]: o for k, o in enumerate(opts)}, letters[opts.index(correct)]


def seasonal_items(s, sid: str) -> list[dict]:
    choices = ["normal for the season", "weaker than in any reference year", "stronger than in any reference year"]
    out = []
    for i, d in enumerate(s.dates):
        verdict = seasonal_verdict(s, i)
        if verdict is None:
            continue
        start = s.index_of(d.year - REFERENCE_YEARS, 1) or 0
        prev = MONTH_NAMES[(d.month - 2) % 12]
        item_id = f"seasonal-{sid}-{d:%Y-%m}"
        options, answer = _shuffle(item_id, verdict, [c for c in choices if c != verdict])
        out.append({"id": item_id, "type": "seasonal", "series": [_series(s, start, i)],
                    "question": (f"Look at the change from {prev} to {MONTH_NAMES[d.month - 1]} {d.year}. Compare it with "
                                 f"the same {prev}-to-{MONTH_NAMES[d.month - 1]} change in each of the previous "
                                 f"{REFERENCE_YEARS} years, leaving out 2020 and 2021. Is this year's change:"),
                    "options": options, "answer": answer, "label": verdict,
                    "explanation": f"change {_shown(month_change(s, i)):+.1f}%; engine verdict: {verdict}"})
    return out


def market_items(s, sid: str) -> list[dict]:
    choices = {"ahead of the market": "faster than total retail",
               "in line with the market": "in line with total retail (within 1.0 percentage point)",
               "behind the market": "slower than total retail"}
    out = []
    b = s.benchmark
    for i, d in enumerate(s.dates):
        verdict = market_verdict(s, i)
        if verdict is None:
            continue
        j = s.index_of(d.year - 1, d.month)
        bi, bj = b.index_of(d.year, d.month), b.index_of(d.year - 1, d.month)
        item_id = f"market-{sid}-{d:%Y-%m}"
        options, answer = _shuffle(item_id, choices[verdict], [v for k, v in choices.items() if k != verdict])
        out.append({"id": item_id, "type": "market", "series": [_series(s, j, i), _series(b, bj, bi)],
                    "question": (f"From {MONTH_NAMES[d.month - 1]} {d.year - 1} to {MONTH_NAMES[d.month - 1]} {d.year}, "
                                 f"did US {s.name.replace(' sales (US)', '').lower()} sales grow:"),
                    "options": options, "answer": answer, "label": verdict,
                    "explanation": f"own {_shown(year_on_year(s, i)):+.1f}% vs {MARKET_NAME}; engine verdict: {verdict}"})
    return out


def trend_items(s, sid: str) -> list[dict]:
    out = []
    for i, d in enumerate(s.dates):
        yoy = year_on_year(s, i)
        if yoy is None:
            continue
        p = _shown(yoy)
        label = "up more than 2%" if p > TREND_BAND else "down more than 2%" if p < -TREND_BAND else "within 2%"
        j = s.index_of(d.year - 1, d.month)
        item_id = f"trend-{sid}-{d:%Y-%m}"
        choices = ["up by more than 2%", "down by more than 2%", "within 2% either way"]
        correct = {"up more than 2%": choices[0], "down more than 2%": choices[1], "within 2%": choices[2]}[label]
        options, answer = _shuffle(item_id, correct, [c for c in choices if c != correct])
        out.append({"id": item_id, "type": "trend", "series": [_series(s, j, i)],
                    "question": (f"Compared with {MONTH_NAMES[d.month - 1]} {d.year - 1}, were sales in "
                                 f"{MONTH_NAMES[d.month - 1]} {d.year}:"),
                    "options": options, "answer": answer, "label": label,
                    "explanation": f"{s.values[i]:,.0f} / {s.values[j]:,.0f} - 1 = {p:+.1f}%"})
    return out



def balance(items: list[dict], rng: random.Random) -> list[dict]:
    """Keep every item of the rarest class and the same number from each other class."""
    by = {}
    for it in items:
        by.setdefault(it["label"], []).append(it)
    if len(by) < 3:
        return items
    n = min(len(v) for v in by.values())
    return sorted([it for v in by.values() for it in rng.sample(v, n)], key=lambda it: it["id"])


def main() -> None:
    rng = random.Random(SEED)
    all_items = []
    for kind in (seasonal_items, market_items, trend_items):
        pooled = [it for sid in CATEGORIES for it in kind(load_census_retail(sid), sid)]
        kept = balance(pooled, rng)
        all_items += kept
        counts = {}
        for it in kept:
            counts[it["label"]] = counts.get(it["label"], 0) + 1
        print(f"{kind.__name__[:-6]:9s} {len(pooled):4d} built, {len(kept):4d} kept {counts if len(counts) <= 3 else ''}")
    with open(OUT, "w") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {OUT} ({len(all_items)} questions)")


if __name__ == "__main__":
    main()
