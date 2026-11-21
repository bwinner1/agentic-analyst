"""Load the Context is Key (CiK) benchmark from its pinned Hugging Face parquet file.

One row is one instance: a task (scenario template) and a seed. It holds the history, the
future to forecast, the context text, and the fields the official metric needs
(metric_scaling, region_of_interest, constraints, task weight).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

HF_COMMIT = "32da467ade142a16d099675e78b1129745c5ef7d"
URL = (
    "https://huggingface.co/datasets/ServiceNow/context-is-key/resolve/"
    f"{HF_COMMIT}/data/test-00000-of-00001.parquet"
)
SHA256 = "f04bebe14b792bb47acfd3dccacb8469ac49f98eb601880e640407e177caad60"
DEFAULT_PATH = Path("evaluation/data/cik-test.parquet")


@dataclass(frozen=True, eq=False)
class Instance:
    task: str
    seed: int
    weight: float  # the task's share in the aggregate (CiK down-weights families of near-identical tasks)
    background: str
    scenario: str
    constraints: str
    past_times: tuple[str, ...]
    past: np.ndarray
    future_times: tuple[str, ...]
    future: np.ndarray
    seasonal_period: int  # -1 when the benchmark gives none
    metric_scaling: float
    roi: np.ndarray  # indices into the future where the context matters most
    constraint_min: float  # -inf when absent
    constraint_max: float  # +inf when absent
    var_max_index: np.ndarray
    var_max_values: np.ndarray
    context_sources: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{self.task}/{self.seed}"

    @property
    def horizon(self) -> int:
        return len(self.future)

    def context_text(self) -> str:
        """Background, scenario and constraints, the way CiK's own prompts combine them."""
        parts = [
            ("Background", self.background),
            ("Scenario", self.scenario),
            ("Constraints", self.constraints),
        ]
        return "\n".join(f"{label}: {text.strip()}" for label, text in parts if text and text.strip())


def fetch(path: Path = DEFAULT_PATH) -> Path:
    """Download the pinned parquet file once and check its sha256."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: sha256 {digest} does not match the pinned {SHA256}")
    return path


def _series(cell: str) -> tuple[tuple[str, ...], np.ndarray]:
    columns = json.loads(cell)
    if len(columns) != 1:  # usually named "0", sometimes e.g. "Occupancy (%)"
        raise ValueError(f"expected one target column, got {list(columns)}")
    (values_by_time,) = columns.values()
    points = sorted(values_by_time.items())  # ISO timestamps sort chronologically
    times = tuple(_clean_time(t) for t, _ in points)
    values = np.array([v for _, v in points], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains missing or non-finite values")
    return times, values


def _clean_time(t: str) -> str:
    return t.replace("T", " ").removesuffix(".000")


def load(path: Path = DEFAULT_PATH) -> list[Instance]:
    import pandas as pd

    df = pd.read_parquet(path)
    instances = []
    for row in df.itertuples(index=False):
        past_times, past = _series(row.past_time)
        future_times, future = _series(row.future_time)
        instances.append(
            Instance(
                task=row.name,
                seed=int(row.seed),
                weight=float(Fraction(row.weight)),
                background=row.background or "",
                scenario=row.scenario or "",
                constraints=row.constraints or "",
                past_times=past_times,
                past=past,
                future_times=future_times,
                future=future,
                seasonal_period=int(row.seasonal_period),
                metric_scaling=float(row.metric_scaling),
                roi=np.asarray(row.region_of_interest, dtype=int),
                constraint_min=float(row.constraint_min),
                constraint_max=float(row.constraint_max),
                var_max_index=np.asarray(row.constraint_variable_max_index, dtype=int),
                var_max_values=np.asarray(row.constraint_variable_max_values, dtype=float),
                context_sources=tuple(row.context_sources),
            )
        )
    ids = [i.id for i in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate (task, seed) pairs in the dataset")
    return instances
