"""Trivial forecasts that ignore the text. A model that reads the context should beat them."""

from __future__ import annotations

import numpy as np


def naive(inst) -> np.ndarray:
    """Repeat the last observed value."""
    return np.full(inst.horizon, inst.past[-1])


def seasonal_naive(inst) -> np.ndarray:
    """Repeat the last full season (24 h, 7 days, ...); falls back to naive without a known season."""
    period = inst.seasonal_period
    if period <= 0 or len(inst.past) < period:
        return naive(inst)
    season = inst.past[-period:]
    reps = -(-inst.horizon // period)
    return np.tile(season, reps)[: inst.horizon]


BASELINES = {"naive": naive, "seasonal-naive": seasonal_naive}
