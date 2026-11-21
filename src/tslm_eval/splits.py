"""Which instances are held out for testing.

CiK tasks come in families of near-copies (16 Montreal-fire causal variants, 7 electricity
heat-wave variants, ...). Holding out single tasks or seeds would test mostly memorisation, so the
proposed split holds out whole scenario groups: the model never saw the scenario during training.
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
PROPOSED_SPLIT_FILE = Path("evaluation/splits/test-holdout-groups.txt")


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


def write_ids(path: Path, ids: set[str], header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(sorted(ids))
    comment = "\n".join(f"# {line}" for line in header.splitlines())
    path.write_text(f"{comment}\n{body}\n", encoding="utf-8")
