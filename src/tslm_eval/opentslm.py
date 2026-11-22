"""Import the OpenTSLM evaluation output (`opentslm-cik/eval_cik.py`) into the harness.

`eval_cik.py` writes, per run directory:
  - test_predictions.jsonl: task_name, seed, forecast_median {column: [...]}, truth {column: [...]}, ...
  - metrics.json: its own approximate scores plus the settings (base model, checkpoint, samples, context)
  - train_log.json (next to it when training ran there): epochs and validation loss

The median of the model's sampled forecasts becomes a point forecast, scored with the official
RCRPS like every other system. When the run saved which samples were readable, only those are
used, and an instance with none counts as a failed output rather than as the placeholder values
eval_cik.py fills in (see `_forecast`). Each instance's `truth` must equal CiK's own future values, which
catches mismatched ids or a different CiK version before anything is scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .forecasts import write_predictions

PREDICTIONS = "test_predictions.jsonl"


def _rows(results_dir: Path) -> list[dict]:
    path = Path(results_dir) / PREDICTIONS
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _only(column_map: dict, what: str, iid: str) -> list:
    if len(column_map) != 1:
        raise ValueError(f"{iid}: expected one {what} column, got {sorted(column_map)}")
    return next(iter(column_map.values()))


def _readable_samples(r: dict, iid: str) -> list | None:
    """The readable sample paths, or None when the run saved no samples or none were readable."""
    samples, valid = r.get("forecast_samples"), r.get("samples_valid")
    if samples is None or valid is None:
        raise ValueError(f"{iid}: this run saved only the median; rerun eval_cik.py with sample saving")
    s = np.atleast_2d(np.asarray(_only(samples, "forecast", iid), dtype=float))
    if len(valid) != len(s):
        raise ValueError(f"{iid}: {len(s)} samples but {len(valid)} validity flags")
    readable = s[np.asarray(valid, dtype=bool)]
    if len(readable) == 0 or not np.all(np.isfinite(readable)):
        return None
    return readable.tolist()


def _forecast(r: dict, iid: str) -> list | None:
    """The model's point forecast for one instance, or None when it gave no readable forecast.

    eval_cik.py fills steps it could not read (from neighbouring steps, or the history mean), so a
    filled-in forecast is not the model's own. When the run saved every sample and which ones were
    readable, the median is taken over the readable samples only, and the instance fails if none
    were. Older runs saved only the median; there the median is used as it is (a sample with a gap
    cannot be told apart from the others), and `run_meta` counts how many instances had one.
    """
    if "forecast_samples" in r and "samples_valid" in r:
        readable = _readable_samples(r, iid)
        return None if readable is None else np.median(np.asarray(readable), axis=0).tolist()
    forecast = np.asarray(_only(r["forecast_median"], "forecast", iid), dtype=float)
    return forecast.tolist() if np.all(np.isfinite(forecast)) else None


def evaluated_ids(results_dir: Path) -> list[str]:
    """Instance ids (Task/seed) the run was evaluated on, in file order."""
    return [f"{r['task_name']}/{r['seed']}" for r in _rows(results_dir)]


def import_run(results_dir: Path, out_path: Path, instances, samples: bool = False) -> list[str]:
    """Write `out_path` (harness prediction format) and `<name>.meta.json` from one eval_cik.py run.

    By default each instance gets the median as a point forecast. With `samples=True` it gets every
    readable sample path instead, for comparing forecast ranges; the run must have saved them.
    """
    results_dir = Path(results_dir)
    by_id = {i.id: i for i in instances}
    records, ids = [], []
    for r in _rows(results_dir):
        iid = f"{r['task_name']}/{r['seed']}"
        inst = by_id.get(iid)
        if inst is None:
            raise ValueError(f"{results_dir}: unknown instance {iid}")
        truth = np.asarray(_only(r["truth"], "truth", iid), dtype=float)
        if truth.shape != inst.future.shape or not np.allclose(truth, inst.future, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{iid}: the run's true future differs from CiK's; ids or data version do not match")
        if samples:
            paths = _readable_samples(r, iid)
            records.append({"id": iid, "samples": paths} if paths else {"id": iid, "forecast": None})
        else:
            records.append({"id": iid, "forecast": _forecast(r, iid)})  # None is scored as a failed output
        ids.append(iid)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{results_dir}: duplicate instance ids")
    write_predictions(Path(out_path), records)
    meta = run_meta(results_dir) | {"forecast_kind": "samples" if samples else "median"}
    Path(out_path).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return ids


def run_meta(results_dir: Path) -> dict:
    """What the report prints about the run: model, checkpoint, context, sampling, training."""
    results_dir = Path(results_dir)
    metrics_path, log_path = results_dir / "metrics.json", results_dir / "train_log.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    settings = metrics.get("settings", {})
    meta = {
        "system_type": "opentslm",
        "source": results_dir.as_posix(),
        "model": settings.get("base"),
        "checkpoint": settings.get("ckpt"),
        "with_context": not settings.get("no_context", False),
        "num_samples": settings.get("num_samples"),
        "temperature": settings.get("temperature"),
        "parsed_ok_rate": metrics.get("parsed_ok_rate"),
    }
    if (results_dir / PREDICTIONS).exists():
        rows = _rows(results_dir)
        meta["instances_with_an_unreadable_sample"] = sum(not r.get("parsed_ok", True) for r in rows)
        meta["per_sample_validity_saved"] = all("samples_valid" in r for r in rows)
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        history = log.get("history", [])
        best = min(history, key=lambda h: h["val_loss"]) if history else None
        meta.update({
            "epochs_run": len(history),
            "best_epoch": best and best["epoch"],
            "best_val_loss": log.get("best_val_loss"),
            "extra_train": log.get("args", {}).get("extra_train"),
        })
    return meta
