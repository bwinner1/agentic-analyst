"""Events added by code to a copy of a real series, inside the review window, with an exact label.

Each injector returns the edited values and the span it touched, or None when the series gives no
place where the event would be clearly visible (then the generator draws again). "Clearly visible"
means at least `MIN_EFFECT` robust standard deviations, so every non-normal label is something an
attentive analyst could see.

Business injectors (spike, dip, level_up, level_down) make real-looking changes; whether they are a
`business_change` or an `explained_event` depends only on the note. Data injectors are of two kinds:
some leave a signature in the series alone (exact zeros, a frozen value, a unit change by x100 or
x1000), others look exactly like a business change and only the note tells them apart (a double
load, a partial load, a change of definition). That second kind is what a rule cannot do and what
reading the note together with the series should.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_EFFECT = 3.0  # robust standard deviations


@dataclass(frozen=True)
class Injected:
    values: np.ndarray
    shape: str
    start: int  # 0-based, inside the review window
    end: int  # inclusive
    params: dict


def robust_scale(hist: np.ndarray, period: int | None) -> float:
    """Typical size of one step's surprise in the history: seasonal differences if there is a
    period and enough history, else first differences; robust to the odd outlier."""
    lag = period if period and len(hist) > 3 * period else 1
    d = hist[lag:] - hist[:-lag]
    s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    if not np.isfinite(s) or s <= 1e-9:
        s = max(float(np.std(hist)), 1e-3 * (abs(float(np.median(hist))) + 1.0))
    return float(s)


def _nonneg(y: np.ndarray) -> bool:
    return bool(np.min(y) >= 0)


def _span(rng, w: int, lo: int, hi: int, start_max: int | None = None) -> tuple[int, int]:
    n = int(rng.integers(lo, hi + 1))
    n = min(n, w)
    last_start = (w - n) if start_max is None else min(w - n, start_max)
    s = int(rng.integers(0, last_start + 1))
    return s, s + n - 1


def _busy_start(rng, win: np.ndarray, n: int) -> int | None:
    """A start where the span sits at or above the window median, so a drop is visible."""
    ok = [s for s in range(0, len(win) - n + 1) if np.all(win[s:s + n] >= np.median(win))]
    return int(rng.choice(ok)) if ok else None


def _short_span(freq: str) -> tuple[int, int]:
    return (2, 5) if freq == "hourly" else (1, 3)


# --- business-looking changes ------------------------------------------------------------------


def spike(y, w, s, rng, freq) -> Injected | None:
    a, b = _span(rng, w, *_short_span(freq))
    out = y.copy()
    k = len(y) - w
    size = float(rng.uniform(5, 9)) * s
    out[k + a:k + b + 1] += size
    return Injected(out, "spike", a, b, {"size_sd": size / s})


def dip(y, w, s, rng, freq) -> Injected | None:
    lo, hi = _short_span(freq)
    n = int(rng.integers(lo, hi + 1))
    k = len(y) - w
    a = _busy_start(rng, y[k:], n)
    if a is None:
        return None
    out = y.copy()
    seg = out[k + a:k + a + n]
    new = seg - float(rng.uniform(5, 9)) * s
    if _nonneg(y):
        new = np.maximum(new, seg * float(rng.uniform(0.1, 0.3)))  # stays above zero: not an outage
    if np.min(seg - new) < MIN_EFFECT * s:
        return None
    out[k + a:k + a + n] = new
    return Injected(out, "dip", a, a + n - 1, {"drop_sd": float(np.mean(seg - new)) / s})


def _level(y, w, s, rng, sign: int) -> Injected | None:
    k = len(y) - w
    a = int(rng.integers(0, max(1, int(w * 0.6)) + 1))
    if w - a < 3:
        return None
    out = y.copy()
    size = float(rng.uniform(4, 7)) * s
    tail = out[k + a:]
    if sign < 0 and _nonneg(y):
        new = np.maximum(tail - size, tail * float(rng.uniform(0.2, 0.4)))
    else:
        new = tail + sign * size
    if np.mean(np.abs(new - tail)) < MIN_EFFECT * s:
        return None
    out[k + a:] = new
    return Injected(out, "level_up" if sign > 0 else "level_down", a, w - 1, {"shift_sd": sign * float(np.mean(new - tail)) / s})


def level_up(y, w, s, rng, freq):
    return _level(y, w, s, rng, +1)


def level_down(y, w, s, rng, freq):
    return _level(y, w, s, rng, -1)


# --- data issues visible in the series alone ---------------------------------------------------


def zero_outage(y, w, s, rng, freq) -> Injected | None:
    lo, hi = _short_span(freq)
    n = int(rng.integers(lo, hi + 1))
    k = len(y) - w
    if not _nonneg(y):
        return None
    a = _busy_start(rng, y[k:], n)
    if a is None or np.min(y[k + a:k + a + n]) < MIN_EFFECT * s:
        return None
    out = y.copy()
    out[k + a:k + a + n] = 0.0
    return Injected(out, "dip", a, a + n - 1, {})


def flatline(y, w, s, rng, freq) -> Injected | None:
    n = int(rng.integers(5, 11)) if freq == "hourly" else int(rng.integers(4, 8))
    if w < n + 1:
        return None
    k = len(y) - w
    a = int(rng.integers(1, w - n + 1))
    out = y.copy()
    out[k + a:k + a + n] = out[k + a - 1]
    if np.max(np.abs(y[k + a:k + a + n] - out[k + a - 1])) < 2 * s:  # the real values barely moved
        return None
    return Injected(out, "flatline", a, a + n - 1, {"frozen_at": float(out[k + a - 1])})


def unit_change(y, w, s, rng, freq) -> Injected | None:
    k = len(y) - w
    a = int(rng.integers(0, max(1, int(w * 0.6)) + 1))
    if w - a < 3 or np.median(np.abs(y)) <= 0:
        return None
    up = bool(rng.random() < 0.5)
    factor = float(rng.choice([100.0, 1000.0]))
    out = y.copy()
    out[k + a:] = out[k + a:] * (factor if up else 1 / factor)
    return Injected(out, "level_up" if up else "level_down", a, w - 1, {"factor": factor if up else 1 / factor})


# --- data issues that look like business changes (only the note tells) --------------------------


def duplicate_load(y, w, s, rng, freq) -> Injected | None:
    lo, hi = _short_span(freq)
    n = int(rng.integers(lo, hi + 1))
    k = len(y) - w
    a = _busy_start(rng, y[k:], n)
    if a is None:
        return None
    out = y.copy()
    out[k + a:k + a + n] *= 2.0
    if np.min(y[k + a:k + a + n]) < MIN_EFFECT * s:
        return None
    return Injected(out, "spike", a, a + n - 1, {"factor": 2.0})


def partial_load(y, w, s, rng, freq) -> Injected | None:
    lo, hi = _short_span(freq)
    n = int(rng.integers(lo, hi + 1))
    k = len(y) - w
    a = _busy_start(rng, y[k:], n)
    if a is None or not _nonneg(y):
        return None
    share = float(rng.uniform(0.3, 0.6))  # share of records that arrived
    out = y.copy()
    seg = y[k + a:k + a + n]
    if np.min(seg * (1 - share)) < MIN_EFFECT * s:
        return None
    out[k + a:k + a + n] = seg * share
    return Injected(out, "dip", a, a + n - 1, {"share_loaded": share})


def definition_change(y, w, s, rng, freq) -> Injected | None:
    k = len(y) - w
    a = int(rng.integers(0, max(1, int(w * 0.6)) + 1))
    if w - a < 3 or not _nonneg(y):
        return None
    f = float(rng.uniform(0.25, 0.5))
    up = bool(rng.random() < 0.5)
    tail = y[k + a:]
    new = tail * (1 + f) if up else tail * (1 - f)
    if np.mean(np.abs(new - tail)) < MIN_EFFECT * s:
        return None
    out = y.copy()
    out[k + a:] = new
    return Injected(out, "level_up" if up else "level_down", a, w - 1, {"factor": float(new[0] / tail[0]) if tail[0] else None})


BUSINESS = {"spike": spike, "dip": dip, "level_up": level_up, "level_down": level_down}
DATA_SERIES_SIGNATURE = {"zero_outage": zero_outage, "flatline": flatline, "unit_change": unit_change}
DATA_NOTE_ONLY = {"duplicate_load": duplicate_load, "partial_load": partial_load, "definition_change": definition_change}
ALL = {**BUSINESS, **DATA_SERIES_SIGNATURE, **DATA_NOTE_ONLY}
