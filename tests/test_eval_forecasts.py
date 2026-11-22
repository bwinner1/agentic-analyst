import json

import pytest

np = pytest.importorskip("numpy")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import baselines  # noqa: E402
from tslm_eval.forecasts import parse_forecast_text, read_predictions  # noqa: E402


def test_parses_timestamped_lines_inside_forecast_tags():
    text = ("I expect a drop.\n<forecast>\n2024-02-01 00:00:00,5.5\n2024-02-02 00:00:00,-1e2\n"
            "2024-02-03 12:30:00,.25\n</forecast>")
    assert parse_forecast_text(text, 3).tolist() == [5.5, -100.0, 0.25]


def test_uses_the_last_forecast_block():
    text = "<forecast>1, 2, 3</forecast> on reflection: <forecast>4, 5, 6</forecast>"
    assert parse_forecast_text(text, 3).tolist() == [4.0, 5.0, 6.0]


def test_plain_text_without_tags_and_extra_numbers():
    assert parse_forecast_text("10 11 12 13", 3).tolist() == [10.0, 11.0, 12.0]


@pytest.mark.parametrize("text", ["", "<forecast>1, 2</forecast>", "no numbers here"])
def test_too_few_numbers_is_a_failed_forecast(text):
    assert parse_forecast_text(text, 3) is None


def _write(tmp_path, rows):
    path = tmp_path / "system.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_reads_every_record_shape_and_marks_problems(tmp_path):
    insts = [make_instance(seed=s) for s in (1, 2, 3, 4, 5)]
    path = _write(tmp_path, [
        {"id": "ATMBuildingClosedTask/1", "forecast": [5, 6, 7]},
        {"task": "ATMBuildingClosedTask", "seed": 2, "text": "<forecast>5\n6\n7</forecast>"},
        {"id": "ATMBuildingClosedTask/3", "samples": [[5, 6, 7], [5, 6, 8]]},
        {"id": "ATMBuildingClosedTask/4", "forecast": [5, 6]},
    ])
    preds = read_predictions(path, insts)
    assert preds["ATMBuildingClosedTask/1"].forecast.tolist() == [5, 6, 7]
    assert preds["ATMBuildingClosedTask/2"].forecast.tolist() == [5, 6, 7]
    assert preds["ATMBuildingClosedTask/3"].forecast.shape == (2, 3)
    assert preds["ATMBuildingClosedTask/4"].forecast is None
    assert "expected 3" in preds["ATMBuildingClosedTask/4"].problem
    assert preds["ATMBuildingClosedTask/5"].problem == "missing"


def test_unknown_instance_id_is_an_error(tmp_path):
    path = _write(tmp_path, [{"id": "NoSuchTask/1", "forecast": [1, 2, 3]}])
    with pytest.raises(ValueError, match="unknown instance"):
        read_predictions(path, [make_instance()])


def test_naive_repeats_the_last_value():
    assert baselines.naive(make_instance(past=(1, 2, 9), future=(0, 0))).tolist() == [9, 9]


def test_seasonal_naive_repeats_the_last_season():
    inst = make_instance(past=(1, 2, 3, 10, 20, 30), future=(0,) * 5, period=3)
    assert baselines.seasonal_naive(inst).tolist() == [10, 20, 30, 10, 20]


def test_seasonal_naive_without_a_season_falls_back_to_naive():
    inst = make_instance(past=(1, 2, 3), future=(0, 0), period=-1)
    assert baselines.seasonal_naive(inst).tolist() == [3, 3]
