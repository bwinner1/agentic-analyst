"""Trivial forecasts that ignore the text. A model that reads the context should beat them.

`BASELINES` give one forecast each. `SAMPLED_BASELINES` give several sample paths, so that a
model's sampled forecasts can be compared with a baseline that also has a spread: CRPS rewards
spread, so a single forecast is at a disadvantage against any set of samples. Every baseline
uses the history only, never the future values.
"""

from __future__ import annotations

import zlib

import numpy as np

N_SAMPLES = 5  # what the OpenTSLM runs use


def naive(inst) -> np.ndarray:
    """Repeat the last observed value."""
    return np.full(inst.horizon, inst.past[-1])


def _has_season(inst, seasons: int = 1) -> bool:
    return inst.seasonal_period > 1 and len(inst.past) >= seasons * inst.seasonal_period


def _repeat(season: np.ndarray, horizon: int) -> np.ndarray:
    reps = -(-horizon // len(season))
    return np.tile(season, reps)[:horizon]


def seasonal_naive(inst) -> np.ndarray:
    """Repeat the last full season (24 h, 7 days, ...); falls back to naive without a known season."""
    if not _has_season(inst):
        return naive(inst)
    return _repeat(inst.past[-inst.seasonal_period:], inst.horizon)


def past_seasons(inst, n: int = N_SAMPLES) -> np.ndarray:
    """`n` sample paths, each repeating one of the last `n` seasons.

    Without a season (or with fewer than two in the history), each path repeats one of the last
    `n` values instead. Short histories give fewer paths. Shape (paths, horizon).
    """
    past, p = inst.past, inst.seasonal_period
    if _has_season(inst, seasons=2):
        k = min(n, len(past) // p)
        return np.stack([_repeat(past[len(past) - (j + 1) * p: len(past) - j * p], inst.horizon)
                         for j in range(k)])
    k = min(n, len(past))
    return np.stack([np.full(inst.horizon, past[-(j + 1)]) for j in range(k)])


def bootstrap_errors(inst, n: int = N_SAMPLES) -> np.ndarray:
    """The seasonal-naive forecast plus `n` runs of the errors that rule made in the history.

    Seasonal: errors are the past differences y[t] - y[t - season], taken in runs of `horizon`
    consecutive steps from random starting points (wrapping round a short history). Without a
    season: the naive forecast plus the running sum of a run of past one-step changes, a random
    walk. Deterministic for each instance. Shape (n, horizon).
    """
    past, h = inst.past, inst.horizon
    rng = np.random.default_rng(zlib.crc32(inst.id.encode()))
    if _has_season(inst, seasons=2):
        p = inst.seasonal_period
        errors = past[p:] - past[:-p]
        point, cumulative = seasonal_naive(inst), False
    else:
        errors = np.diff(past)
        point, cumulative = naive(inst), True
    if len(errors) == 0:
        return np.tile(point, (n, 1))
    starts = rng.integers(0, len(errors), size=n)
    runs = np.stack([errors[(s + np.arange(h)) % len(errors)] for s in starts])
    return point[None, :] + (np.cumsum(runs, axis=1) if cumulative else runs)


BASELINES = {"naive": naive, "seasonal-naive": seasonal_naive}
SAMPLED_BASELINES = {"seasonal-naive-past-seasons": past_seasons, "seasonal-naive-bootstrap": bootstrap_errors}
