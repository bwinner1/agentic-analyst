"""What a triage answer contains, and its exact text format (written by models, read back by code).

A Label describes the review window: a verdict, a shape and the span it covers. Positions are
0-based inside the review window in code and 1-based ("points 3-5") in text, because that is how a
reader counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERDICTS = ("normal", "business_change", "explained_event", "data_issue")
SHAPES = ("none", "spike", "dip", "level_up", "level_down", "flatline")

# What each verdict asks the analyst to do; used in reasons and in the plan.
VERDICT_MEANING = {
    "normal": "nothing in the review window needs a comment",
    "business_change": "a real change with no explanation in the note; investigate and report it",
    "explained_event": "a real change that the note explains; report it with that cause",
    "data_issue": "a problem in the data, not in the business; fix it before quoting any figure",
}


@dataclass(frozen=True)
class Label:
    verdict: str
    shape: str
    start: int | None = None  # 0-based index in the review window; None when shape is "none"
    end: int | None = None  # inclusive

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}")
        if (self.shape == "none") != (self.start is None) or (self.start is None) != (self.end is None):
            raise ValueError("start and end are set exactly when the shape is not 'none'")
        if self.verdict == "normal" and self.shape != "none":
            raise ValueError("a normal window has shape 'none'")
        if self.start is not None and not 0 <= self.start <= self.end:
            raise ValueError(f"bad span {self.start}-{self.end}")

    @property
    def needs_attention(self) -> bool:
        return self.verdict != "normal"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "shape": self.shape, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict) -> "Label":
        return cls(d["verdict"], d["shape"], d.get("start"), d.get("end"))


ANSWER_INSTRUCTIONS = (
    "Answer in exactly this format:\n"
    f"Verdict: one of {', '.join(VERDICTS)}\n"
    f"Shape: one of {', '.join(SHAPES)}\n"
    "Window points: first-last (1 is the first point of the review window), or none\n"
    "Reason: one sentence, without numbers"
)


def format_answer(label: Label, reason: str) -> str:
    where = "none" if label.start is None else f"{label.start + 1}-{label.end + 1}"
    return f"Verdict: {label.verdict}\nShape: {label.shape}\nWindow points: {where}\nReason: {reason}"


_VERDICT = re.compile(r"verdict\s*:\s*([a-z_ ]+)", re.I)
_SHAPE = re.compile(r"shape\s*:\s*([a-z_ ]+)", re.I)
_WHERE = re.compile(r"window points\s*:\s*(none|(\d+)\s*(?:-|to|–)\s*(\d+)|(\d+))", re.I)
_REASON = re.compile(r"reason\s*:\s*(.+)", re.I)


def _word(m: re.Match | None, allowed: tuple[str, ...]) -> str | None:
    if not m:
        return None
    w = m.group(1).strip().lower().replace(" ", "_")
    return next((a for a in allowed if w.startswith(a)), None)


def parse_answer(text: str, window: int) -> tuple[Label | None, str]:
    """Read a Label and the reason from model text. Returns (None, reason) when the verdict is
    missing or the parts contradict each other, which is scored as a failed answer."""
    verdict = _word(_VERDICT.search(text), VERDICTS)
    shape = _word(_SHAPE.search(text), SHAPES)
    m = _REASON.search(text)
    reason = m.group(1).strip().splitlines()[0] if m else ""
    if verdict is None:
        return None, reason
    if verdict == "normal":
        return Label("normal", "none"), reason
    where = _WHERE.search(text)
    if shape is None or shape == "none" or where is None or where.group(1).lower() == "none":
        return None, reason
    a = int(where.group(2) or where.group(4))
    b = int(where.group(3) or where.group(4))
    a, b = min(a, b), max(a, b)
    if a < 1 or b > window:
        return None, reason
    return Label(verdict, shape, a - 1, b - 1), reason
