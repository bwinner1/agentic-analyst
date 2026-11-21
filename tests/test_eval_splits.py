from pathlib import Path

import pytest

pytest.importorskip("numpy")

from eval_helpers import make_instance  # noqa: E402
from tslm_eval import cik, splits  # noqa: E402

REAL_DATA = Path("evaluation/data/cik-test.parquet")


def test_group_holdout_takes_whole_groups_only():
    insts = [make_instance(task=t, seed=s) for t in
             ("ATMBuildingClosedTask", "CashDepletedinATMScenarioTask", "SensorSpikeTask",
              "ElectricityIncreaseInPredictionTask") for s in (1, 2)]
    test = splits.group_holdout(insts, ("atm",))
    assert test == {"ATMBuildingClosedTask/1", "ATMBuildingClosedTask/2",
                    "CashDepletedinATMScenarioTask/1", "CashDepletedinATMScenarioTask/2"}


def test_unknown_group_is_an_error():
    with pytest.raises(ValueError, match="unknown groups"):
        splits.group_holdout([make_instance()], ("retail",))


def test_split_file_round_trip(tmp_path):
    ids = {"A/1", "B/2"}
    path = tmp_path / "split.txt"
    splits.write_ids(path, ids, "two lines\nof header")
    assert splits.read_ids(path) == ids


@pytest.mark.skipif(not REAL_DATA.exists(), reason="run `python -m tslm_eval fetch` first")
def test_every_real_task_has_a_group_and_the_proposed_split_is_whole_tasks():
    insts = cik.load(REAL_DATA)
    assert len(insts) == 355
    groups = {i.task: splits.scenario_group(i.task) for i in insts}
    assert len(groups) == 71
    test = splits.group_holdout(insts)
    test_tasks = {i.task for i in insts if i.id in test}
    train_tasks = {i.task for i in insts if i.id not in test}
    assert not test_tasks & train_tasks
    assert len(test) == 65
