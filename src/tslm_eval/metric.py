"""The CiK benchmark's metric, RCRPS, reimplemented from the official code.

Sources (read 2026-11-21): ServiceNow/context-is-key-forecasting `cik_benchmark/metrics/crps.py`,
`roi_metric.py` (threshold_weighted_crps), and the dataset's `compute_rcrps_with_hf_dataset.py`.

Per instance:  RCRPS = metric_scaling * CRPS_part + CRPS(10 * constraint violation, 0)
where CRPS_part gives half the weight to the region of interest and half to the rest (or a
plain mean when there is no region of interest). Scores of 5 or more, and NaN, are capped at 5,
which is also the score of a forecast that could not be produced (the official convention).
Lower is better; 0 is a perfect forecast.

A point forecast is one sample, for which CRPS is the absolute error. The official estimator
needs at least two samples, so scores of point forecasts are comparable with each other but
not with the published leaderboard, which uses 25-50 samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAP = 5.0
ROI_WEIGHT = 0.5
VIOLATION_FACTOR = 10.0


def crps(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS per position. samples: (n_samples, *target.shape)."""
    target = np.asarray(target, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.shape[1:] != target.shape:
        raise ValueError(f"shape mismatch: target {target.shape}, samples {samples.shape}")
    n = samples.shape[0]
    if n == 1:
        return np.abs(samples[0] - target)
    s = np.sort(samples, axis=0)
    abs_diff = np.abs(target[None] - s).sum(axis=0) / n
    beta0 = s.sum(axis=0) / n
    i = np.arange(n).reshape((n,) + (1,) * target.ndim)
    beta1 = (i * s).sum(axis=0) / (n * (n - 1))
    return abs_diff + beta0 - 2 * beta1


def constraint_violation(inst, samples: np.ndarray) -> np.ndarray:
    """Scaled amount by which each sample breaks the instance's constraints, shape (n_samples,)."""
    scale = inst.metric_scaling
    scaled = scale * samples
    v = np.zeros(samples.shape[0])
    if np.isfinite(inst.constraint_min):
        v += (scale * inst.constraint_min - scaled).clip(min=0).mean(axis=1)
    if np.isfinite(inst.constraint_max):
        v += (scaled - scale * inst.constraint_max).clip(min=0).mean(axis=1)
    if len(inst.var_max_index):
        limits = scale * np.asarray(inst.var_max_values, dtype=float)
        v += (scaled[:, inst.var_max_index] - limits[None, :]).clip(min=0).mean(axis=1)
    return v


def rcrps(inst, forecast: np.ndarray | None) -> float:
    """Score one instance. forecast: (horizon,) point forecast or (n_samples, horizon); None = failed."""
    if forecast is None:
        return CAP
    samples = np.asarray(forecast, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[1] != inst.horizon or not np.all(np.isfinite(samples)):
        return CAP
    per_step = crps(inst.future, samples)
    if len(inst.roi):
        in_roi = np.zeros(inst.horizon, dtype=bool)
        in_roi[inst.roi] = True
        if in_roi.all():
            part = per_step.mean()
        else:
            part = ROI_WEIGHT * per_step[in_roi].mean() + (1 - ROI_WEIGHT) * per_step[~in_roi].mean()
    else:
        part = per_step.mean()
    penalty = VIOLATION_FACTOR * constraint_violation(inst, samples)
    violation_crps = crps(np.zeros(1), penalty[:, None])[0]
    score = inst.metric_scaling * part + violation_crps
    if not np.isfinite(score) or score >= CAP:
        return CAP
    return float(score)


def range_coverage(inst, forecast: np.ndarray | None) -> tuple[float, float]:
    """How often the truth falls inside the samples' range, and how often it should.

    Returns (share of forecast steps where the true value lies between the lowest and highest
    sample, the share expected for well-calibrated samples). With n samples drawn from the same
    distribution as the truth, the truth is the most extreme of the n + 1 values with probability
    2 / (n + 1), so it lies inside the range with probability (n - 1) / (n + 1): 4 in 6 for five
    samples. Much lower means the ranges are too narrow (overconfident); much higher, too wide.
    NaN for a point forecast or a failed one.
    """
    if forecast is None:
        return float("nan"), float("nan")
    samples = np.asarray(forecast, dtype=float)
    if samples.ndim == 1 or samples.shape[0] < 2 or samples.shape[1] != inst.horizon:
        return float("nan"), float("nan")
    inside = (samples.min(axis=0) <= inst.future) & (inst.future <= samples.max(axis=0))
    n = samples.shape[0]
    return float(inside.mean()), (n - 1) / (n + 1)


@dataclass(frozen=True)
class Aggregate:
    mean: float
    ci_low: float
    ci_high: float
    n_instances: int
    n_tasks: int


def aggregate(tasks: list[str], scores: list[float], weights: list[float],
              n_boot: int = 2000, seed: int = 0) -> Aggregate:
    """Weighted mean (CiK task weights) with a 95% interval from resampling whole tasks.

    Resampling tasks, not instances, because the five seeds of a task are not independent.
    """
    tasks_arr = np.asarray(tasks)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float((w * s).sum() / w.sum())
    unique = np.unique(tasks_arr)
    by_task = [(w[tasks_arr == t] * s[tasks_arr == t]).sum() for t in unique]
    w_task = [w[tasks_arr == t].sum() for t in unique]
    by_task, w_task = np.asarray(by_task), np.asarray(w_task)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    boot = by_task[draws].sum(axis=1) / w_task[draws].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return Aggregate(mean, float(low), float(high), len(s), len(unique))
