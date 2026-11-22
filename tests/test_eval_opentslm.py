import json

import pytest

pytest.importorskip("numpy")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import opentslm  # noqa: E402
from tslm_eval.forecasts import read_predictions  # noqa: E402


def _write_eval_output(folder, rows, settings=None, history=None):
    folder.mkdir()
    with (folder / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if settings is not None:
        (folder / "metrics.json").write_text(json.dumps({"settings": settings, "parsed_ok_rate": 1.0}))
    if history is not None:
        (folder / "train_log.json").write_text(json.dumps(
            {"args": {}, "history": history, "best_val_loss": min(h["val_loss"] for h in history)}))


def _row(task, seed, forecast, truth):
    return {"task_name": task, "seed": seed, "forecast_median": {"0": forecast}, "truth": {"0": truth},
            "explanation": "", "raw_output": "", "parsed_ok": True}


def test_import_writes_median_forecasts_and_run_settings(tmp_path):
    insts = [make_instance(seed=s) for s in (1, 2)]
    _write_eval_output(
        tmp_path / "full",
        [_row("ATMBuildingClosedTask", 1, [5.5, 6, 7], [5, 6, 7])],
        settings={"base": "OpenTSLM/llama-3.2-1b-m4-sp", "ckpt": "best.pt", "no_context": False,
                  "num_samples": 5, "temperature": 0.7},
        history=[{"epoch": 1, "val_loss": 1.09}, {"epoch": 2, "val_loss": 1.17}],
    )
    out = tmp_path / "runs" / "opentslm-finetuned-cik.jsonl"
    out.parent.mkdir()
    ids = opentslm.import_run(tmp_path / "full", out, insts)

    assert ids == ["ATMBuildingClosedTask/1"]
    preds = read_predictions(out, insts)
    assert list(preds["ATMBuildingClosedTask/1"].forecast) == [5.5, 6, 7]
    assert preds["ATMBuildingClosedTask/2"].problem == "missing"
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["system_type"] == "opentslm" and meta["with_context"] is True
    assert meta["best_epoch"] == 1 and meta["epochs_run"] == 2 and meta["num_samples"] == 5


def test_import_refuses_a_run_whose_truth_differs_from_cik(tmp_path):
    _write_eval_output(tmp_path / "run", [_row("ATMBuildingClosedTask", 1, [5, 6, 7], [5, 6, 8])])
    with pytest.raises(ValueError, match="differs from CiK"):
        opentslm.import_run(tmp_path / "run", tmp_path / "x.jsonl", [make_instance()])


def test_import_refuses_unknown_instances(tmp_path):
    _write_eval_output(tmp_path / "run", [_row("NoSuchTask", 1, [5, 6, 7], [5, 6, 7])])
    with pytest.raises(ValueError, match="unknown instance"):
        opentslm.import_run(tmp_path / "run", tmp_path / "x.jsonl", [make_instance()])


def test_non_finite_forecast_becomes_a_failed_output(tmp_path):
    _write_eval_output(tmp_path / "run", [_row("ATMBuildingClosedTask", 1, [5, float("nan"), 7], [5, 6, 7])])
    out = tmp_path / "x.jsonl"
    opentslm.import_run(tmp_path / "run", out, [make_instance()])
    assert read_predictions(out, [make_instance()])["ATMBuildingClosedTask/1"].forecast is None


def test_evaluated_ids_are_the_test_set(tmp_path):
    _write_eval_output(tmp_path / "run", [_row("A", 1, [1], [1]), _row("B", 3, [1], [1])])
    assert opentslm.evaluated_ids(tmp_path / "run") == ["A/1", "B/3"]
