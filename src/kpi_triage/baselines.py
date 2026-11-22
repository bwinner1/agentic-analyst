"""Comparison systems that need no GPU. Each writes answers in the same text format as the models,
so one scorer reads everything.

- `rules`: what a careful analyst could script. Seasonal residuals against the same phase in past
  cycles, signature checks for exact zeros, frozen values and x100/x1000 unit changes, a level-shift
  search, and a note classifier fitted on the training notes only (nearest training note by shared
  words), so the held-out test wording is new to it as well. This is the bar a model must clear.
- `series-only`: the rules without the note. `note-only`: the note classifier without the series.
  Together they show how much each input alone is worth.
- `majority`: always "normal" (25% by construction, since verdicts are balanced).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .events import robust_scale
from .labels import Label, format_answer

Z_SPIKE = 4.0  # a point this many robust standard deviations from its seasonal expectation stands out
Z_SPAN = 2.5  # neighbours beyond this join the same event
Z_LEVEL = 3.0  # mean shift of the tail needed for a level change
UNIT_RATIO = 30.0  # a jump by this factor or more at once reads as a unit change

NOTE_CLASS = {
    "none": "nothing", "irrelevant": "nothing", "no_effect": "nothing",
    "event_spike": "event_up", "event_level_up": "event_up",
    "event_dip": "event_down", "event_level_down": "event_down",
    "data_missing": "data", "data_duplicate": "data", "data_definition": "data", "data_stuck": "data",
}
UP_SHAPES, DOWN_SHAPES = ("spike", "level_up"), ("dip", "level_down")


# --- reading the note -------------------------------------------------------------------------

_WORD = re.compile(r"[a-z]+")
_STOP = set("the a an and or of to in on at for was were is are be been has have had it its this that with from by as so some since after only".split())


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


class NoteClassifier:
    """Nearest training note by Jaccard overlap of content words; empty note means nothing."""

    def __init__(self, train_rows: list[dict]):
        self.memory = [(_tokens(r["note"]), NOTE_CLASS[r["note_kind"]]) for r in train_rows if r["note"]]

    def __call__(self, note: str) -> str:
        t = _tokens(note)
        if not t:
            return "nothing"
        best, cls = -1.0, "nothing"
        for m, c in self.memory:
            j = len(t & m) / len(t | m)
            if j > best:
                best, cls = j, c
        return cls if best > 0 else "nothing"


# --- reading the series -------------------------------------------------------------------------


@dataclass
class Finding:
    shape: str
    start: int
    end: int
    signature: bool  # a data-issue signature visible in the series alone


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _expected(y: np.ndarray, n_hist: int, period: int | None) -> np.ndarray:
    """Expected value at every position from the history only: the median of the same phase in up to
    four past cycles, or the median of the recent history when there is no usable period."""
    hist = y[:n_hist]
    if period and n_hist >= 2 * period:
        out = np.empty(len(y))
        for t in range(len(y)):
            same = [hist[i] for i in range(t - period, -1, -period) if i < n_hist][:4]
            out[t] = np.median(same) if same else np.median(hist[-period:])
        return out
    return np.full(len(y), np.median(hist[-max(10, len(hist) // 4):]))


def find_event(values, window: int, period: int | None) -> Finding | None:
    y = np.asarray(values, dtype=float)
    n_hist = len(y) - window
    hist, win = y[:n_hist], y[n_hist:]

    # Signatures that only data problems leave.
    frozen = np.r_[False, np.abs(np.diff(y)) <= 1e-9 * np.maximum(1.0, np.abs(y[1:]))][n_hist:]
    hist_frozen = max((b - a + 1 for a, b in _runs(np.abs(np.diff(hist)) <= 1e-9)), default=0)
    run = max(_runs(frozen), key=lambda r: r[1] - r[0], default=None)
    if run and run[1] - run[0] + 1 >= 3 and hist_frozen < 3:
        return Finding("flatline", run[0], run[1], True)
    if np.min(hist) > 0:
        zero = max(_runs(win == 0), key=lambda r: r[1] - r[0], default=None)
        if zero:
            return Finding("dip", zero[0], zero[1], True)
    base = np.median(np.abs(hist)) + 1e-12
    ratio = np.abs(win) / base
    for j in range(window - 2):
        tail = ratio[j:]
        if np.mean(tail >= UNIT_RATIO) >= 0.7 and ratio[j] >= UNIT_RATIO:
            return Finding("level_up", j, window - 1, True)
        if np.mean(tail <= 1 / UNIT_RATIO) >= 0.7 and ratio[j] <= 1 / UNIT_RATIO:
            return Finding("level_down", j, window - 1, True)

    # Business-looking changes, in seasonal residuals.
    exp = _expected(y, n_hist, period)
    s = robust_scale(hist - exp[:n_hist], None)
    z = (win - exp[n_hist:]) / s
    best = None
    for j in range(0, window - max(3, int(np.ceil(0.35 * window))) + 1):
        tail = z[j:]
        m = float(np.mean(tail))
        if abs(m) >= Z_LEVEL and np.mean(np.sign(tail) == np.sign(m)) >= 0.8:
            score = abs(m) * np.sqrt(len(tail))
            if best is None or score > best[0]:
                best = (score, j, m)
    if best is not None:
        _, j, m = best
        # Start where the shift begins, not at an earlier stray point.
        while j < window - 3 and abs(z[j]) < Z_SPAN:
            j += 1
        return Finding("level_up" if m > 0 else "level_down", j, window - 1, False)
    k = int(np.argmax(np.abs(z)))
    if abs(z[k]) >= Z_SPIKE:
        sign = np.sign(z[k])
        a = b = k
        while a > 0 and z[a - 1] * sign >= Z_SPAN:
            a -= 1
        while b < window - 1 and z[b + 1] * sign >= Z_SPAN:
            b += 1
        return Finding("spike" if sign > 0 else "dip", a, b, False)
    return None


# --- the systems --------------------------------------------------------------------------------


def _verdict(finding: Finding | None, note_class: str | None) -> Label:
    if finding is None:
        return Label("normal", "none")
    if finding.signature or note_class == "data":
        verdict = "data_issue"
    elif (note_class == "event_up" and finding.shape in UP_SHAPES) or (note_class == "event_down" and finding.shape in DOWN_SHAPES):
        verdict = "explained_event"
    else:
        verdict = "business_change"
    return Label(verdict, finding.shape, finding.start, finding.end)


def rules(rows: list[dict], train_rows: list[dict]) -> dict[str, str]:
    notes = NoteClassifier(train_rows)
    return {r["id"]: format_answer(_verdict(find_event(r["values"], r["window"], r["period"]), notes(r["note"])), "rules") for r in rows}


def series_only(rows: list[dict], train_rows: list[dict]) -> dict[str, str]:
    return {r["id"]: format_answer(_verdict(find_event(r["values"], r["window"], r["period"]), None), "rules, series only") for r in rows}


def note_only(rows: list[dict], train_rows: list[dict]) -> dict[str, str]:
    notes = NoteClassifier(train_rows)
    verdict = {"nothing": "normal", "event_up": "explained_event", "event_down": "explained_event", "data": "data_issue"}
    out = {}
    for r in rows:
        v = verdict[notes(r["note"])]
        # Without the series there is no place to point at; the whole window is the best guess.
        label = Label("normal", "none") if v == "normal" else Label(v, "dip", 0, r["window"] - 1)
        out[r["id"]] = format_answer(label, "note only")
    return out


def majority(rows: list[dict], train_rows: list[dict]) -> dict[str, str]:
    return {r["id"]: format_answer(Label("normal", "none"), "always normal") for r in rows}


SYSTEMS = {"rules": rules, "series-only": series_only, "note-only": note_only, "majority": majority}
