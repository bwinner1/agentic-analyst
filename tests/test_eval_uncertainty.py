"""Baselines with a range, range coverage, sample imports and the uncertainty report."""

import json
import math

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import baselines, metric, opentslm, report  # noqa: E402
from tslm_eval.forecasts import read_predictions  # noqa: E402


def test_past_seasons_repeats_each_of_the_last_seasons():
    inst = make_instance(past=(1, 2, 3, 10, 20, 30, 100, 200, 300), future=(0,) * 4, period=3)
    paths = baselines.past_seasons(inst, n=5)
    assert paths.tolist() == [[100, 200, 300, 100], [10, 20, 30, 10], [1, 2, 3, 1]]  # only 3 seasons exist


def test_past_seasons_without_a_season_repeats_the_last_values():
    inst = make_instance(past=(1, 2, 3, 4, 5, 6), future=(0, 0), period=-1)
    assert baselines.past_seasons(inst, n=3).tolist() == [[6, 6], [5, 5], [4, 4]]


@pytest.mark.parametrize("period", [3, -1])
def test_bootstrap_paths_come_from_the_history_only(period):
    a = make_instance(past=(1, 5, 2, 8, 3, 9, 4, 7, 6), future=(0, 0, 0, 0), period=period)
    b = make_instance(past=a.past, future=(99, -99, 99, -99), period=period)
    pa, pb = baselines.bootstrap_errors(a), baselines.bootstrap_errors(b)
    assert pa.shape == (baselines.N_SAMPLES, 4)
    np.testing.assert_array_equal(pa, pb)  # the future never enters
    np.testing.assert_array_equal(pa, baselines.bootstrap_errors(a))  # deterministic


def test_seasonal_bootstrap_adds_past_seasonal_errors_to_the_seasonal_naive_forecast():
    inst = make_instance(past=(1, 2, 3, 2, 3, 4), future=(0, 0, 0), period=3)
    # every past seasonal difference is +1, so every path is seasonal naive + 1
    assert baselines.bootstrap_errors(inst).tolist() == [[3, 4, 5]] * baselines.N_SAMPLES


def test_coverage_counts_steps_inside_the_range_and_expects_four_in_six_for_five_samples():
    inst = make_instance(future=(5.0, 6.0, 7.0))
    samples = np.array([[4, 4, 4], [6, 7, 6], [5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=float)
    covered, expected = metric.range_coverage(inst, samples)
    assert covered == pytest.approx(2 / 3) and expected == pytest.approx(4 / 6)
    assert all(math.isnan(v) for v in metric.range_coverage(inst, np.array([5.0, 6.0, 7.0])))
    assert all(math.isnan(v) for v in metric.range_coverage(inst, None))


def _eval_row(seed, samples, valid, truth=(5, 6, 7)):
    return {"task_name": "ATMBuildingClosedTask", "seed": seed, "forecast_median": {"0": [0, 0, 0]},
            "forecast_samples": {"0": samples}, "samples_valid": valid, "truth": {"0": list(truth)},
            "parsed_ok": all(valid), "explanation": "", "raw_output": ""}


def _write_run(folder, rows):
    folder.mkdir()
    (folder / "test_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps({"settings": {"num_samples": 3, "ckpt": "none"}}))


def test_sample_import_keeps_readable_paths_and_fails_cases_with_none(tmp_path):
    insts = [make_instance(seed=s) for s in (1, 2)]
    _write_run(tmp_path / "run", [_eval_row(1, [[5, 6, 7], [6, 7, 8], [0, 0, 0]], [True, True, False]),
                                  _eval_row(2, [[1, 1, 1], [2, 2, 2], [3, 3, 3]], [False, False, False])])
    out = tmp_path / "runs" / "m.jsonl"
    opentslm.import_run(tmp_path / "run", out, insts, samples=True)
    preds = read_predictions(out, insts)
    assert preds["ATMBuildingClosedTask/1"].forecast.tolist() == [[5, 6, 7], [6, 7, 8]]
    assert preds["ATMBuildingClosedTask/2"].forecast is None
    assert json.loads(out.with_suffix(".meta.json").read_text())["forecast_kind"] == "samples"


def test_sample_import_refuses_a_run_that_saved_only_the_median(tmp_path):
    row = _eval_row(1, [[5, 6, 7]], [True])
    del row["forecast_samples"], row["samples_valid"]
    _write_run(tmp_path / "run", [row])
    with pytest.raises(ValueError, match="only the median"):
        opentslm.import_run(tmp_path / "run", tmp_path / "x.jsonl", [make_instance()], samples=True)


def _paired(other, low, high):
    return {"other": other, "diff": (low + high) / 2, "ci_low": low, "ci_high": high}


@pytest.mark.parametrize("rows, expected", [
    ([_paired("seasonal-naive-past-seasons", -0.2, -0.1), _paired("seasonal-naive-bootstrap", -0.3, -0.05)], "better"),
    ([_paired("seasonal-naive-past-seasons", -0.2, -0.1), _paired("seasonal-naive-bootstrap", -0.1, 0.05)], "level"),
    ([_paired("seasonal-naive-past-seasons", -0.2, -0.1), _paired("seasonal-naive-bootstrap", 0.01, 0.2)], "worse"),
    ([_paired("seasonal-naive", 0.1, 0.2)], None),  # a single-forecast baseline does not decide it
])
def test_verdict_follows_the_rule_fixed_before_the_runs(rows, expected):
    assert report.uncertainty_verdict(rows) == expected


def test_uncertainty_report_scores_sample_sets_and_states_the_verdict(tmp_path):
    insts = [make_instance(task=t, seed=s, past=(1, 2, 3, 4, 5, 6), future=(20, 21, 22), scaling=0.1, period=3)
             for t in ("ATMBuildingClosedTask", "ElectricityIncreaseInPredictionTask") for s in (1, 2)]
    runs = tmp_path / "runs"
    runs.mkdir()
    systems = {
        "opentslm-finetuned-cik": lambda i: {"samples": [[19, 20, 21], [20, 21, 22], [21, 22, 23]]},
        "seasonal-naive-past-seasons": lambda i: {"samples": baselines.past_seasons(i).tolist()},
        "seasonal-naive-bootstrap": lambda i: {"samples": baselines.bootstrap_errors(i).tolist()},
        "seasonal-naive": lambda i: {"forecast": baselines.seasonal_naive(i).tolist()},
    }
    for name, fn in systems.items():
        (runs / f"{name}.jsonl").write_text("\n".join(json.dumps({"id": i.id, **fn(i)}) for i in insts) + "\n",
                                            encoding="utf-8")
    path = report.write_uncertainty_report(runs, tmp_path / "out", insts, {i.id for i in insts}, "all four")
    text = path.read_text(encoding="utf-8")
    summary = pd.read_csv(tmp_path / "out" / "uncertainty_summary.csv").set_index("system")
    assert summary.loc["opentslm-finetuned-cik", "samples"] == 3
    assert summary.loc["opentslm-finetuned-cik", "coverage"] == pytest.approx(1.0)
    assert "n/a (single forecast)" in text and "### Verdict" in text
    assert "Better on uncertainty" in text  # ours brackets the truth; the baselines sit below it
