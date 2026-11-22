"""The business question benchmark: answer key, balance, prompts and scoring."""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

BQA = Path(__file__).resolve().parents[1] / "business-qa"
sys.path.insert(0, str(BQA))

from prompts import opentslm_parts, parse_letter, text_prompt  # noqa: E402
from score import macro_f1, score  # noqa: E402

QUESTIONS = {q["id"]: q for q in map(json.loads, open(BQA / "questions.jsonl"))}


def test_every_question_is_well_formed():
    for q in QUESTIONS.values():
        assert set(q["options"]) == {"A", "B", "C"}
        assert len(set(q["options"].values())) == 3
        assert q["answer"] in "ABC"
        assert all(len(s["dates"]) == len(s["values"]) for s in q["series"])


def test_classes_are_balanced_within_each_type():
    for t in ("seasonal", "market", "trend"):
        labels = Counter(q["label"] for q in QUESTIONS.values() if q["type"] == t)
        assert len(labels) == 3 and len(set(labels.values())) == 1, (t, labels)


def test_answer_letters_are_spread_so_always_one_letter_scores_near_a_third():
    letters = Counter(q["answer"] for q in QUESTIONS.values())
    assert max(letters.values()) / len(QUESTIONS) < 0.40


def test_answer_key_checked_by_hand_for_two_questions():
    """Shoe stores, September 2026 (data/census_retail, FRED snapshot of 22 Nov 2026), done by hand:
    2,500 / 3,784 - 1 = -33.9%, inside the reference range -37.6% to -25.1% -> normal for the season.
    Trend: 2,500 / 2,407 - 1 = +3.9% -> up by more than 2%."""
    for qid, label in [("seasonal-MRTSSM4482USN-2026-09", "normal for the season"),
                       ("trend-MRTSSM4482USN-2026-09", "up more than 2%")]:
        q = QUESTIONS[qid]  # both are in the balanced sample (fixed seed)
        assert q["label"] == label
        assert q["options"][q["answer"]] in {"normal for the season", "up by more than 2%"}


def test_trend_answers_recomputed_from_the_series_shown():
    for q in (q for q in QUESTIONS.values() if q["type"] == "trend"):
        v = q["series"][0]["values"]
        p = round((v[-1] / v[0] - 1) * 100, 1)
        expected = "up more than 2%" if p > 2 else "down more than 2%" if p < -2 else "within 2%"
        assert q["label"] == expected, q["id"]


def test_prompts_carry_the_series_question_and_options():
    q = next(iter(QUESTIONS.values()))
    text = text_prompt(q)
    assert q["question"] in text and all(f"{k}) {v}" in text for k, v in q["options"].items())
    assert f"{q['series'][0]['dates'][-1]}: {q['series'][0]['values'][-1]:,}" in text
    pre, series, post = opentslm_parts(q)
    assert len(series) == len(q["series"]) and q["question"] in post


@pytest.mark.parametrize("reply, letter", [("B", "B"), ("B) slower", "B"), ("(C).", "C"), ("The answer is A.", "A"),
                                           ("  a", None), ("I am not sure", None)])
def test_parse_letter(reply, letter):
    assert parse_letter(reply) == letter


def test_scoring_perfect_and_constant_answers():
    perfect = score(QUESTIONS, {k: q["answer"] for k, q in QUESTIONS.items()})
    assert perfect["all"]["accuracy"] == 1.0 and perfect["all"]["macro_f1"] == 1.0
    always_a = score(QUESTIONS, {k: "A" for k in QUESTIONS})
    assert 0.25 < always_a["all"]["accuracy"] < 0.40
    assert macro_f1([("A", None), ("B", None)]) == 0.0
