"""Prediction files: one JSONL file per system, one line per instance.

Each line names the instance and gives the forecast in one of three ways:

    {"id": "ElectricityIncreaseInPredictionTask/1", "forecast": [351.2, 296.4, ...]}
    {"id": "...", "samples": [[...], [...]]}          # several sample paths, if the model has them
    {"id": "...", "text": "<raw model output>"}       # parsed here, e.g. OpenTSLM's generated text

`"task"` + `"seed"` may replace `"id"`. A forecast with the wrong length, unparseable text, or a
missing line all count as a failed forecast, scored at the metric's cap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TAGGED = re.compile(r"<forecast>(.*?)</forecast>", re.DOTALL | re.IGNORECASE)
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_forecast_text(text: str, horizon: int) -> np.ndarray | None:
    """Pull `horizon` numbers out of model text; None if there are fewer.

    Uses the last <forecast>...</forecast> block when present. Timestamps are removed first so
    `2013-05-28 12:00:00, 351.2` lines yield 351.2. Commas separate values; thousands separators
    are not supported.
    """
    if not text:
        return None
    blocks = _TAGGED.findall(text)
    body = blocks[-1] if blocks else text
    body = _TIMESTAMP.sub(" ", body)
    numbers = [float(x) for x in _NUMBER.findall(body)]
    if len(numbers) < horizon:
        return None
    values = np.array(numbers[:horizon])
    return values if np.all(np.isfinite(values)) else None


@dataclass
class Prediction:
    forecast: np.ndarray | None  # (horizon,) or (n_samples, horizon); None = failed
    problem: str = ""


def _record_id(rec: dict) -> str:
    if "id" in rec:
        return str(rec["id"])
    if "task" in rec and "seed" in rec:
        return f"{rec['task']}/{int(rec['seed'])}"
    raise ValueError(f"record has neither 'id' nor 'task'+'seed': {sorted(rec)}")


def to_prediction(rec: dict, horizon: int) -> Prediction:
    if rec.get("forecast") is not None:
        arr = np.asarray(rec["forecast"], dtype=float)
    elif rec.get("samples") is not None:
        arr = np.asarray(rec["samples"], dtype=float)
    elif "text" in rec:
        arr = parse_forecast_text(rec.get("text") or "", horizon)
        if arr is None:
            return Prediction(None, "no forecast found in text")
    else:
        return Prediction(None, "record has no forecast, samples or text")
    if arr.shape[-1] != horizon:
        return Prediction(None, f"length {arr.shape[-1]}, expected {horizon}")
    if not np.all(np.isfinite(arr)):
        return Prediction(None, "non-finite values")
    return Prediction(arr)


def read_predictions(path: Path, instances) -> dict[str, Prediction]:
    """Predictions keyed by instance id. Instances without a line get a failed Prediction."""
    by_id = {i.id: i for i in instances}
    out: dict[str, Prediction] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = _record_id(rec)
        if rid not in by_id:
            raise ValueError(f"{path}:{n}: unknown instance id {rid!r}")
        out[rid] = to_prediction(rec, by_id[rid].horizon)
    for i in instances:
        out.setdefault(i.id, Prediction(None, "missing"))
    return out


def write_predictions(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
