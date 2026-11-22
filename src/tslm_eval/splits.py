"""Which instances are held out for testing.

CiK tasks come in families of near-copies (16 Montreal-fire causal variants, 7 electricity
heat-wave variants, ...). Holding out single tasks or seeds would test mostly memorisation.

The test set we report on is the one the fine-tuned OpenTSLM model was actually held out from
(`TEST_SPLIT_FILE`): 21 task types whose series never appear in its training or validation data,
written by `python -m tslm_eval split` from its evaluation output. `group_holdout` (whole scenario
groups) is the earlier proposal, kept for a future retrain.
"""

from __future__ import annotations

import re
from pathlib import Path

# First match wins. Covers all 71 tasks (a test checks this).
GROUP_RULES: list[tuple[str, str]] = [
    ("montreal-fire", r"^MontrealFire"),
    ("atm", r"ATM|CashDepleted|IncreasedWithdrawal"),
    ("electricity", r"Electricity"),
    ("traffic", r"Traffic"),
    ("sensor", r"^Sensor"),
    ("solar", r"Irradiance|Solar"),
    ("wind-tunnel", r"PressureFromSpeed|SpeedFromLoad"),
    ("causal", r"SVAR"),
    ("unemployment", r"^Unemployment"),
    ("constraints", r"BoundedPred|OraclePred"),
]

# Business-like scenarios the model must not see in training: ATM cash withdrawals,
# county unemployment, highway sensor faults. 13 tasks, 65 instances, 24% of the task weight.
PROPOSED_TEST_GROUPS = ("atm", "unemployment", "sensor")

# The default test set: the instances the OpenTSLM fine-tune never trained or validated on.
TEST_SPLIT_FILE = Path("evaluation/splits/test-opentslm-cik.txt")


def scenario_group(task: str) -> str:
    for group, pattern in GROUP_RULES:
        if re.search(pattern, task):
            return group
    raise ValueError(f"task {task!r} matches no scenario group; add a rule")


def group_holdout(instances, groups=PROPOSED_TEST_GROUPS) -> set[str]:
    unknown = set(groups) - {g for g, _ in GROUP_RULES}
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    return {i.id for i in instances if scenario_group(i.task) in groups}


def seed_holdout(instances, seeds=(5,)) -> set[str]:
    """Weaker split: same tasks, unseen seeds. Use only if training already used every task."""
    return {i.id for i in instances if i.seed in seeds}


def read_ids(path: Path) -> set[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def read_label(path: Path) -> str:
    """The first comment line of a split file, which names the test set for the report."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return Path(path).stem


def write_ids(path: Path, ids: set[str], header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(sorted(ids))
    comment = "\n".join(f"# {line}" for line in header.splitlines())
    path.write_text(f"{comment}\n{body}\n", encoding="utf-8")
