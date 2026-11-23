"""KPI triage: labels, injected events, generated examples, prompts, baselines and scoring.

All tests use small synthetic backgrounds, so they need no download.
"""

from collections import Counter

import pytest

np = pytest.importorskip("numpy")

from kpi_triage import baselines, events, generate, notes, prompting, score  # noqa: E402
from kpi_triage.backgrounds import Background  # noqa: E402
from kpi_triage.labels import Label, format_answer, parse_answer  # noqa: E402


def _series(n=200, period=5, seed=0, level=100.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return level + 20 * np.sin(2 * np.pi * t / period) + rng.normal(0, 2, n)


def _background(group, seed, freq="trading-day", period=5, n=200):
    days = [f"2010-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
    return Background(f"synthetic/{group}/{seed}", "online_retail", group, "Online revenue, test", "GBP", freq, period,
                      tuple(days), _series(n, period or 5, seed))


@pytest.fixture(scope="module")
def rows():
    bgs = [_background(g, s) for g in ("retail:United Kingdom", "retail:EIRE", "retail:Germany") for s in range(3)]
    return generate.generate(bgs, {"train": 120, "val": 40, "test": 80}, seed=1)


# --- answer format ---------------------------------------------------------------------------


def test_answer_text_round_trips():
    for label in (Label("normal", "none"), Label("data_issue", "flatline", 3, 8), Label("explained_event", "level_up", 0, 19)):
        parsed, reason = parse_answer(format_answer(label, "because"), window=20)
        assert parsed == label and reason == "because"


def test_unreadable_or_contradictory_answers_parse_to_none():
    assert parse_answer("I think it is fine", 20)[0] is None
    assert parse_answer("Verdict: data_issue\nShape: none\nWindow points: none", 20)[0] is None
    assert parse_answer("Verdict: business_change\nShape: dip\nWindow points: 19-25", 20)[0] is None  # past the window
    assert parse_answer("verdict: Normal", 20)[0] == Label("normal", "none")


def test_labels_reject_inconsistent_parts():
    with pytest.raises(ValueError):
        Label("normal", "spike", 1, 2)
    with pytest.raises(ValueError):
        Label("business_change", "dip")


# --- injected events ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(events.ALL))
def test_injectors_change_only_the_labelled_span_inside_the_window(name):
    w, placed = 20, 0
    for seed in range(30):
        y = _series(seed=seed)
        inj = events.ALL[name](y, w, events.robust_scale(y[:-w], 5), np.random.default_rng(seed), "trading-day")
        if inj is None:
            continue
        placed += 1
        changed = np.flatnonzero(~np.isclose(inj.values, y))
        assert np.all(changed >= len(y) - w), "the history must stay untouched"
        span = set(range(len(y) - w + inj.start, len(y) - w + inj.end + 1))
        assert set(changed) <= span
        if inj.shape in ("level_up", "level_down"):
            assert inj.end == w - 1
    assert placed > 0, f"{name} never found a place on a clean seasonal series"


# --- generated examples -------------------------------------------------------------------------


def test_verdicts_are_exactly_balanced_and_splits_keep_their_groups(rows):
    for split, rs in rows.items():
        counts = Counter(r["label"]["verdict"] for r in rs)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert {r["split"] for r in rs} == {split}
    assert {r["group"] for r in rows["test"]} == {"retail:Germany"}
    assert {r["group"] for r in rows["val"]} == {"retail:EIRE"}


def test_test_notes_use_only_held_out_paraphrases(rows):
    for r in rows["test"]:
        if r["note_template"] != "none":
            kind, i = r["note_template"].rsplit(".", 1)
            assert int(i) >= len(notes.NOTES[kind]) - notes.HELD_OUT
    for r in rows["train"]:
        if r["note_template"] != "none":
            kind, i = r["note_template"].rsplit(".", 1)
            assert int(i) < len(notes.NOTES[kind]) - notes.HELD_OUT


def test_note_matches_the_verdict(rows):
    for r in rows["train"] + rows["test"]:
        v, kind = r["label"]["verdict"], r["note_kind"]
        if v == "explained_event":
            assert kind == notes.EVENT_NOTE_FOR_SHAPE[r["label"]["shape"]]
        if v == "business_change":
            assert kind in ("none", "irrelevant") or kind == notes.OPPOSITE_EVENT_NOTE[r["label"]["shape"]]
        if v == "normal":
            assert kind in ("none", "irrelevant", "no_effect")
        if r["injector"] in events.DATA_NOTE_ONLY:
            assert kind.startswith("data_"), "a data issue that looks like business needs the note"


def test_generation_is_deterministic():
    bgs = [_background(g, 0) for g in ("retail:United Kingdom", "retail:EIRE", "retail:Germany")]
    a = generate.generate(bgs, {"train": 12, "val": 4, "test": 4}, seed=3)
    b = generate.generate(bgs, {"train": 12, "val": 4, "test": 4}, seed=3)
    assert a == b


# --- prompts ----------------------------------------------------------------------------------


def test_both_prompts_carry_the_same_window_and_answer(rows):
    r = next(r for r in rows["test"] if r["label"]["verdict"] == "data_issue")
    ex = prompting.opentslm_example(r)
    assert [len(s) for s in ex.series] == [len(r["values"]) - r["window"], r["window"]]
    hist = np.asarray(r["values"][: -r["window"]])
    assert np.isclose(np.mean(ex.series[0]), 0, atol=1e-9) and np.isclose(np.std(ex.series[0]), 1)
    assert np.isclose(ex.series[1][0], (r["values"][-r["window"]] - hist.mean()) / hist.std())
    text, answer = prompting.text_prompt(r)
    assert answer == ex.answer and r["note"] in text and r["note"] in ex.pre_prompt
    assert parse_answer(answer, r["window"])[0] == Label.from_dict(r["label"])


# --- baselines ----------------------------------------------------------------------------------


def test_rules_see_series_signatures_and_leave_clean_windows_alone():
    y = _series()
    assert baselines.find_event(y, 20, 5) is None
    frozen = y.copy()
    frozen[-12:-6] = frozen[-13]
    f = baselines.find_event(frozen, 20, 5)
    assert f.shape == "flatline" and f.signature and f.start == 8
    unit = y.copy()
    unit[-10:] *= 1000
    f = baselines.find_event(unit, 20, 5)
    assert f.shape == "level_up" and f.signature and f.start == 10


def test_note_classifier_learns_only_from_training_notes(rows):
    clf = baselines.NoteClassifier(rows["train"])
    assert clf("") == "nothing"
    assert clf("The load job ran twice on Monday, so some records are duplicated.") == "data"


# --- scoring -----------------------------------------------------------------------------------


def test_perfect_answers_score_one_and_majority_scores_a_quarter(rows):
    rs = rows["test"]
    perfect = {r["id"]: format_answer(Label.from_dict(r["label"]), "x") for r in rs}
    j = score.judge_system(rs, perfect)
    assert score.MEASURES["verdict_macro_f1"](j) == 1.0 and score.MEASURES["full_match"](j) == 1.0
    j = score.judge_system(rs, baselines.majority(rs, rows["train"]))
    assert score.MEASURES["verdict_accuracy"](j) == pytest.approx(0.25, abs=0.02)


def test_unreadable_answers_count_as_wrong_and_missing_answers_raise(rows):
    rs = rows["test"]
    j = score.judge_system(rs, {r["id"]: "no idea" for r in rs})
    assert score.MEASURES["unreadable"](j) == 1.0 and score.MEASURES["verdict_accuracy"](j) == 0.0
    with pytest.raises(ValueError, match="no answer"):
        score.judge_system(rs, {})


def test_a_system_against_itself_differs_by_zero(rows):
    rs = rows["test"]
    j = score.judge_system(rs, baselines.rules(rs, rows["train"]))
    d, lo, hi = score.paired(j, j, score.MEASURES["full_match"], n_boot=50)
    assert d == lo == hi == 0.0


def test_combination_keeps_rule_places_and_takes_model_verdicts(rows):
    rs = rows["test"]
    perfect = {r["id"]: format_answer(Label.from_dict(r["label"]), "x") for r in rs}
    out = baselines.code_finds_model_reads(rs, perfect)
    for r in rs:
        f = baselines.find_event(r["values"], r["window"], r["period"])
        got = parse_answer(out[r["id"]], r["window"])[0]
        if f is None:
            assert got == Label("normal", "none")
        else:
            assert (got.shape, got.start, got.end) == (f.shape, f.start, f.end)
            if not f.signature and r["label"]["verdict"] != "normal":
                assert got.verdict == r["label"]["verdict"]
