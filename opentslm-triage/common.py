"""Shared paths, JSONL input and reproducibility helpers."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

BASE = "OpenTSLM/llama-3.2-1b-m4-sp"


def add_repo_src() -> None:
    """Find src next to this script's folder, including a copied VM bundle."""
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def read_rows(data_dir: str | Path, split: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Read one split without changing its order."""
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    filename = {"train": "train", "val": "val", "validation": "val", "test": "test"}[split]
    path = Path(data_dir) / f"{filename}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def write_json(path: str | Path, value: Any) -> None:
    """Write readable, standards-compliant JSON."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    """Seed the libraries already supplied by OpenTSLM."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> str:
    """Default to CUDA when available."""
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")
