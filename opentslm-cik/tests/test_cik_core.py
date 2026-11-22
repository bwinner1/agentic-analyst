"""Tests for cik_core on hand-written rows shaped like CiK rows (no real benchmark data)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cik_core  # noqa: E402


def row(name="HourlyTask", seed=1, **kw):
    r = {
        "name": name,
        "seed": seed,
        "weight": "1/2",
        "context_sources": ["c_cov"],
        "skills": ["forecasting", "instruction following"],
        "background": "Hourly electricity load of a factory in kW.",
        "scenario": "The factory shuts down for maintenance from 2020-01-02 02:00 for 2 hours.",
        "constraints": "",
        "seasonal_period": 3,
        "past_time": json.dumps({"load": {f"2020-01-01T{h:02d}:00:00.000": float(10 + h % 3) for h in range(12)}}),
        "future_time": json.dumps({"load": {"2020-01-02T00:00:00.000": 10.0, "2020-01-02T01:00:00.000": 11.0,
                                            "2020-01-02T02:00:00.000": 0.0, "2020-01-02T03:00:00.000": 0.0}}),
        "metric_scaling": 0.5,
        "region_of_interest": [2, 3],
        "constraint_min": 0.0,
        "constraint_max": float("inf"),
        "constraint_variable_max_index": [],
        "constraint_variable_max_values": [],
    }
    r.update(kw)
    return r


def series_row(name, seed, base):
    past = {f"2020-01-01T{h:02d}:00:00.000": float(base + h) for h in range(12)}
    fut = {f"2020-01-02T{h:02d}:00:00.000": float(base + 100 + h) for h in range(4)}
    return row(name=name, seed=seed, past_time=json.dumps({"load": past}), future_time=json.dumps({"load": fut}))


def test_split_keeps_all_seeds_of_a_task_together():
    rows = [series_row(f"Task{t}", s, base=1000 * t + 10 * s) for t in range(71) for s in range(1, 6)]
    train, val, test = cik_core.split_by_task(rows)
    names = [{r["name"] for r in part} for part in (train, val, test)]
    assert not (names[0] & names[1]) and not (names[0] & names[2]) and not (names[1] & names[2])
    assert len(train) + len(val) + len(test) == len(rows)
    assert 200 <= len(train) <= 230 and 25 <= len(val) <= 45 and 95 <= len(test) <= 115
    assert cik_core.split_by_task(rows)[2] == test  # deterministic


def test_split_keeps_variants_on_the_same_series_together():
    # TaskA and TaskB are two wordings of the same problem (identical series for seed 3 only);
    # they must land in the same split, or the test answer leaks into training.
    rows = [series_row(f"Task{t}", s, base=1000 * t + 10 * s) for t in range(20) for s in range(1, 6)]
    rows += [series_row("TaskA", s, base=500_000 + s) for s in range(1, 6)]
    rows += [series_row("TaskB", s, base=500_000 + s if s == 3 else 600_000 + s) for s in range(1, 6)]
    groups = cik_core.series_groups(rows)
    assert groups["TaskA"] == groups["TaskB"] and groups["Task1"] != groups["Task2"]
    split = {r["name"]: k for k, part in zip(("train", "val", "test"), cik_core.split_by_task(rows)) for r in part}
    assert split["TaskA"] == split["TaskB"]


def test_build_example_prompt_and_answer_round_trip():
    ex = cik_core.build_example(row(), rationale="Load drops to zero during the shutdown.")
    assert "Context:" in ex.pre_prompt and "maintenance" in ex.pre_prompt
    assert len(ex.series) == 1 and len(ex.series[0]) == 12
    assert abs(np.mean(ex.series[0])) < 1e-6
    assert ex.answer.startswith("Forecast:\n1: ") and "\n4: " in ex.answer and "Explanation: Load drops" in ex.answer
    fc, valid, expl = cik_core.parse_answer(ex.answer, ex.meta)
    assert valid and expl.startswith("Load drops")
    np.testing.assert_allclose(fc["load"], [10, 11, 0, 0], atol=0.02)


def test_no_context_removes_text_but_keeps_series():
    ex = cik_core.build_example(row(), include_context=False)
    assert "maintenance" not in ex.pre_prompt and "Context:" not in ex.pre_prompt
    assert "Explanation" not in ex.answer and "Explanation" not in ex.post_prompt
    assert len(ex.series[0]) == 12


def test_history_truncation():
    long_past = json.dumps({"x": {f"2020-01-{1 + i // 24:02d}T{i % 24:02d}:00:00.000": float(i) for i in range(600)}})
    ex = cik_core.build_example(row(past_time=long_past, future_time=json.dumps({"x": {"2020-02-01T00:00:00.000": 1.0}})),
                                max_history=100)
    assert len(ex.series[0]) == 100 and "most recent part" in ex.series_texts[0]


def test_multivariate_history_and_multiple_targets():
    past = json.dumps({"a": {"2021-01": 1.0, "2021-02": 2.0, "2021-03": 3.0},
                       "b": {"2021-01": 5.0, "2021-02": 5.0, "2021-03": 5.0}})
    fut = json.dumps({"a": {"2021-04": 4.0}, "b": {"2021-04": 5.0}})
    ex = cik_core.build_example(row(past_time=past, future_time=fut))
    assert len(ex.series) == 2
    assert "Forecast a:" in ex.answer and "Forecast b:" in ex.answer
    fc, valid, _ = cik_core.parse_answer(ex.answer, ex.meta)
    assert valid
    np.testing.assert_allclose(fc["a"], [4.0], atol=0.02)
    np.testing.assert_allclose(fc["b"], [5.0], atol=0.02)


def test_parse_handles_malformed_output():
    ex = cik_core.build_example(row())
    fc, valid, _ = cik_core.parse_answer("Forecast: 0.1, 0.2", ex.meta)
    assert not valid and len(fc["load"]) == 4
    fc, valid, _ = cik_core.parse_answer("I am not sure.", ex.meta)
    assert not valid and len(fc["load"]) == 4


def test_scores_perfect_forecast_is_zero_and_naive_is_worse():
    ex = cik_core.build_example(row())
    truth = {"load": np.array([[10.0, 11.0, 0.0, 0.0]])}
    perfect = cik_core.score_instance(ex.meta, truth)
    assert perfect["mae"] == pytest.approx(0) and perfect["crps"] == pytest.approx(0) and perfect["violations"] == 0
    naive = cik_core.score_instance(ex.meta, {k: v[None] for k, v in cik_core.seasonal_naive(ex.meta).items()})
    assert naive["mae"] > 0
    neg = cik_core.score_instance(ex.meta, {"load": np.array([[-1.0, -1.0, -1.0, -1.0]])})
    assert neg["violations"] == 1.0


def test_aggregate_uses_weights():
    metas = [{"weight": "1/4"}, {"weight": "3/4"}]
    out = cik_core.aggregate([(metas[0], {"mae": 1.0}), (metas[1], {"mae": 3.0})])
    assert out["mae"] == pytest.approx(2.5) and out["n_instances"] == 2


def test_load_rows_from_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "cik.parquet"
    pd.DataFrame([row(), row(seed=2)]).to_parquet(path)
    rows = cik_core.load_rows(str(path))
    assert len(rows) == 2 and isinstance(rows[0]["skills"], list)
    cik_core.build_example(rows[0])


def test_parse_answer_without_label_and_with_bad_output():
    meta = {"targets": [{"name": "load", "values": [1, 2, 3, 4], "mean": 10.0, "std": 2.0}]}
    fc, ok, _ = cik_core.parse_answer("Values on 2013-05-28 will be 0.1, 0.2, 0.3, 0.4 roughly", meta)
    assert ok and np.allclose(fc["load"], [10.2, 10.4, 10.6, 10.8])
    fc, ok, _ = cik_core.parse_answer("This series shows a daily pattern.", meta)
    assert not ok and np.allclose(fc["load"], 10.0)


def test_parse_numbered_lines_with_missing_and_extra_steps():
    meta = {"targets": [{"name": "load", "values": [1, 2, 3, 4], "mean": 10.0, "std": 2.0}]}
    fc, ok, e = cik_core.parse_answer("Forecast:\n1: 0.0\n2: 1.0\n4: 3.0\n5: 9.0\nExplanation: why", meta)
    assert not ok and e == "why"
    np.testing.assert_allclose(fc["load"], [10, 12, 14, 16])  # step 3 filled between 2 and 4; step 5 ignored
    fc, ok, _ = cik_core.parse_answer("Forecast:\n1: 0.0\n2: 1.0\n3: 2.0\n4: 3.0", meta)
    assert ok


def test_parse_two_targets():
    meta = {"targets": [{"name": "a", "values": [1, 2], "mean": 0.0, "std": 1.0},
                        {"name": "b", "values": [1, 2], "mean": 0.0, "std": 1.0}]}
    fc, ok, _ = cik_core.parse_answer("Forecast b:\n1: 5\n2: 6\nForecast a:\n1: 1\n2: 2", meta)
    assert ok and list(fc["a"]) == [1, 2] and list(fc["b"]) == [5, 6]
