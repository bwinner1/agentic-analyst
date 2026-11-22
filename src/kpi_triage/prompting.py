"""Turn one example row into model input, in two forms that carry the same information:

- `opentslm_example(row)`: text before and after, plus labelled series that OpenTSLM-SP reads
  natively (the same shape as `opentslm-cik/cik_core.Example`).
- `text_prompt(row)`: everything as text, with the values written out, for text-only models
  (Llama-1B fine-tuned on text, Claude). This is ADR 0001's key baseline: if a text model given
  the same numbers spots equally well, a time-series encoder adds nothing.

Both normalise with the mean and spread of the history before the review window, so a level
shift inside the window stays visible, and both state those figures in the text.

Row fields used (see generate.py): name, unit, freq, timestamps, values, window, note, label, reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .labels import ANSWER_INSTRUCTIONS, Label, format_answer

CLIP = 25.0  # normalised values beyond this are clipped (e.g. a unit change times 1000)


@dataclass
class Example:
    pre_prompt: str
    series_texts: list[str]
    series: list[list[float]]
    post_prompt: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


def _stats(row) -> tuple[float, float]:
    hist = np.asarray(row["values"][: -row["window"]], dtype=float)
    m, s = float(np.mean(hist)), float(np.std(hist))
    return m, (s if s > 1e-12 else 1.0)


def normalised(row) -> tuple[np.ndarray, float, float]:
    m, s = _stats(row)
    z = np.clip((np.asarray(row["values"], dtype=float) - m) / s, -CLIP, CLIP)
    return z, m, s


def _header(row) -> str:
    ts, w = row["timestamps"], row["window"]
    note = row.get("note") or "none"
    return (
        "You are an analyst checking a client KPI before writing the period's commentary.\n"
        f"KPI: {row['name']} ({row['unit']}), {row['freq']} data, {len(ts)} points from {ts[0]} to {ts[-1]}.\n"
        f"The review window is the last {w} points, from {ts[-w]} to {ts[-1]}.\n"
        f"Analyst note: {note}"
    )


QUESTION = "Does anything in the review window need attention before the commentary is written?"


def answer_text(row) -> str:
    return format_answer(Label.from_dict(row["label"]), row["reason"])


MARK_CLIP = 3.0  # log10 ratios beyond a factor of 1,000 are clipped


def marks(row) -> tuple[list[float], list[float]]:
    """Two series that keep, inside patches of several points, what triage turns on:
    - the change from the previous point (history scale): exactly 0 where a value repeats;
    - log10 of the ratio to the history's median absolute value: 0 at the usual level,
      -3 at zero, +3 / -3 at a x1000 / /1000 unit change."""
    z, _, _ = normalised(row)
    w = row["window"]
    change = np.diff(z[-w - 1:])
    y = np.abs(np.asarray(row["values"], dtype=float))
    med = float(np.median(y[:-w])) or 1.0
    ratio = np.clip(np.log10(np.maximum(y[-w:], med * 10 ** -MARK_CLIP) / med), -MARK_CLIP, MARK_CLIP)
    return change.tolist(), ratio.tolist()


def opentslm_example(row, with_values: bool = False, with_marks: bool = False) -> Example:
    """With `with_values`, the same numbers the text-only model reads also follow the series as
    text, so OpenTSLM has everything the text model has plus its own series input: the fair test
    of whether the encoder adds anything on top of text. With `with_marks`, two more series (see
    `marks`) carry repeated values, zeros and unit changes in a form patches cannot blur."""
    z, m, s = normalised(row)
    w = row["window"]
    hist, win = z[:-w], z[-w:]
    scale = f"normalised with the history's mean {m:.4g} and standard deviation {s:.4g}"
    values = (
        f"The same values as text. History before the review window: {', '.join(_fmt(v) for v in hist)}\n"
        f"Review window: {', '.join(_fmt(v) for v in win)}\n"
    ) if with_values else ""
    texts = [f"History before the review window ({len(hist)} points, {scale}):", f"Review window ({w} points, same scale):"]
    series = [hist.tolist(), win.tolist()]
    if with_marks:
        change, ratio = marks(row)
        texts += [
            "Review window, change from the previous point (same scale; exactly 0 where the value repeats):",
            "Review window, log10 of the ratio to the history's usual level (0 usual, -3 zero, +/-3 a factor of 1,000):",
        ]
        series += [change, ratio]
    return Example(
        pre_prompt=_header(row),
        series_texts=texts,
        series=series,
        post_prompt=f"{values}{QUESTION}\n{ANSWER_INSTRUCTIONS}\nAnswer:",
        answer=answer_text(row),
        meta={"id": row["id"], "window": w},
    )


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def text_prompt(row) -> tuple[str, str]:
    """(prompt, answer) for a text-only model. Values are the same normalised values OpenTSLM sees."""
    z, m, s = normalised(row)
    w = row["window"]
    prompt = (
        f"{_header(row)}\n"
        f"Values are normalised with the history's mean {m:.4g} and standard deviation {s:.4g}.\n"
        f"History before the review window: {', '.join(_fmt(v) for v in z[:-w])}\n"
        f"Review window: {', '.join(_fmt(v) for v in z[-w:])}\n"
        f"{QUESTION}\n{ANSWER_INSTRUCTIONS}\nAnswer:"
    )
    return prompt, answer_text(row)
