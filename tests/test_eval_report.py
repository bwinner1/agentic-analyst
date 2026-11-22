import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import report  # noqa: E402


def _instances():
    return [make_instance(task=t, seed=s, past=(1, 2, 3, 4), future=(8, 9, 10), scaling=0.1)
            for t in ("ATMBuildingClosedTask", "ElectricityIncreaseInPredictionTask") for s in (1, 2)]


def _write_runs(runs, insts):
    runs.mkdir()
    systems = {
        "naive": lambda i: [4, 4, 4],
        "oracle": lambda i: [8, 9, 10],
        "broken": lambda i: [1, 2],
    }
    for name, fn in systems.items():
        lines = [json.dumps({"id": i.id, "forecast": fn(i)}) for i in insts]
        (runs / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_summary_ranks_systems_and_compares_with_naive(tmp_path):
    insts = _instances()
    _write_runs(tmp_path / "runs", insts)
    summary = report.summarize(report.score_runs(tmp_path / "runs", insts)).set_index("system")
    assert summary.loc["oracle", "rcrps"] == 0.0
    assert summary.loc["oracle", "better_than_baseline"] == 1.0  # baseline: naive (no seasonal-naive here)
    assert summary.loc["broken", "rcrps"] == 5.0 and summary.loc["broken", "valid_rate"] == 0.0
    assert list(summary.index) == ["oracle", "naive", "broken"]


def test_test_set_summary_uses_only_held_out_ids(tmp_path):
    insts = _instances()
    _write_runs(tmp_path / "runs", insts)
    scored = report.score_runs(tmp_path / "runs", insts)
    test_ids = {i.id for i in insts if i.task.startswith("ATM")}
    summary = report.summarize(scored, test_ids)
    assert set(summary.n_instances) == {2} and set(summary.n_tasks) == {1}


def test_write_report_produces_table_and_charts(tmp_path):
    insts = _instances()
    _write_runs(tmp_path / "runs", insts)
    out = tmp_path / "results"
    test_ids = {i.id for i in insts if i.task.startswith("ATM")}
    path = report.write_report(tmp_path / "runs", out, insts, test_ids, "atm group",
                               examples=["ATMBuildingClosedTask/1", "ElectricityIncreaseInPredictionTask/2"],
                               highlight="oracle")
    text = path.read_text(encoding="utf-8")
    assert "Held-out test set: atm group" in text and "| oracle | 0.000 |" in text
    for name in ("chart_test.png", "chart_all.png", "chart_example.png", "chart_example_2.png",
                 "per_instance.csv", "paired_test.csv"):
        assert (out / name).stat().st_size > 0


def test_a_system_run_on_the_test_set_only_is_kept_out_of_the_all_instances_table(tmp_path):
    insts = _instances()
    runs = tmp_path / "runs"
    _write_runs(runs, insts)
    test_ids = {i.id for i in insts if i.task.startswith("ATM")}
    lines = [json.dumps({"id": i, "forecast": [8, 9, 10]}) for i in sorted(test_ids)]
    (runs / "model-finetuned.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "results"
    text = report.write_report(runs, out, insts, test_ids, "atm group").read_text(encoding="utf-8")

    all_table = text.split("## All 355 instances")[1].split("## By scenario group")[0]
    assert "| model-finetuned |" not in all_table
    assert "run on the test set only: model-finetuned" in all_table
    assert "| model-finetuned | 0.000 |" in text  # in the test-set table
    per_instance = pd.read_csv(out / "per_instance.csv")
    assert set(per_instance[per_instance.system == "model-finetuned"].id) == test_ids


def test_paired_difference_is_ours_minus_other():
    insts = _instances()
    scored = {"ours": pd.DataFrame({"id": [i.id for i in insts], "task": [i.task for i in insts],
                                    "weight": 1.0, "rcrps": [0.1, 0.1, 0.5, 0.5]}),
              "other": pd.DataFrame({"id": [i.id for i in insts], "task": [i.task for i in insts],
                                     "weight": 1.0, "rcrps": [0.3, 0.3, 0.4, 0.4]})}
    row = report.paired(scored, "ours", "other")
    assert row["diff"] == pytest.approx(-0.05)
    assert row["lower"] == 0.5 and row["higher"] == 0.5 and row["n"] == 4


def test_seasonal_naive_is_the_reference_when_present():
    assert report.reference({"naive": None, "seasonal-naive": None}) == "seasonal-naive"
    assert report.reference({"naive": None}) == "naive"
    assert report.default_highlight({"opentslm-finetuned-cik-extra": 0, "opentslm-finetuned-cik": 0})         == "opentslm-finetuned-cik"
