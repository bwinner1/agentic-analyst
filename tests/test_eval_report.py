import json

import pytest

pytest.importorskip("pandas")
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
    assert summary.loc["oracle", "better_than_naive"] == 1.0
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
                               example_id="ATMBuildingClosedTask/1", highlight="oracle")
    text = path.read_text(encoding="utf-8")
    assert "Held-out test set: atm group" in text and "| oracle | 0.000 |" in text
    for name in ("chart_test.png", "chart_all.png", "chart_example.png", "per_instance.csv"):
        assert (out / name).stat().st_size > 0
