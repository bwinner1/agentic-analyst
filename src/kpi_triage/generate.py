"""Build the triage examples: real backgrounds, injected events, notes, exact labels.

Verdicts are balanced (a quarter each), so always answering the most common verdict scores 25%.
Splits hold out whole groups (a country of the retailer, a CiK domain) and the test split also
uses held-out note paraphrases, so the test measures new series and new wording at once.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import events
from .backgrounds import Background
from .labels import VERDICTS, Label
from .notes import DATA_NOTE_FOR_INJECTOR, EVENT_NOTE_FOR_SHAPE, OPPOSITE_EVENT_NOTE, when_phrase, write_note

# Whole groups per split. Everything not listed is training data. "All countries" and the UK
# overlap heavily (most revenue is British), so both stay in training.
TEST_GROUPS = ("retail:Germany", "retail:France", "cik:electricity")
VAL_GROUPS = ("retail:EIRE",)
DEFAULT_SIZES = {"train": 4000, "val": 400, "test": 800}
MAX_TRIES = 200  # series and crops tried for one planned event before the plan is drawn again

WINDOW = {"trading-day": 20, "daily": 14, "hourly": 24, "weekly": 8}  # four trading weeks, two weeks, one day, eight weeks
CROP = {"trading-day": (80, 200), "daily": (96, 168), "hourly": (96, 168), "weekly": (56, 104)}

SHAPE_WORDS = {
    "spike": "short spike",
    "dip": "short dip",
    "level_up": "step up that persists",
    "level_down": "step down that persists",
    "flatline": "run of identical values",
}
SIGNATURE_REASON = {
    "zero_outage": "values fall to exactly zero, which points to missing data rather than real activity",
    "flatline": "the values repeat exactly for several points, which points to a stuck feed",
    "unit_change": "the level changes by a factor of a hundred or more at once, which points to a unit or definition change",
}


def split_of(bg: Background) -> str:
    if bg.group in TEST_GROUPS:
        return "test"
    if bg.group in VAL_GROUPS:
        return "val"
    return "train"


def _reason(verdict: str, shape: str, injector: str | None, note_kind: str) -> str:
    words = SHAPE_WORDS.get(shape, "")
    if verdict == "normal":
        tail = {
            "no_effect": ", and the note describes something that has not affected these figures",
            "irrelevant": ", and the note is unrelated to the figures",
        }.get(note_kind, "")
        return f"Nothing in the review window stands out from the usual pattern{tail}."
    if verdict == "business_change":
        if note_kind.startswith("event_"):
            return f"The window shows a {words}; the note describes a change in the other direction, so it does not explain it and it needs investigating."
        return f"The window shows a {words} that the note does not explain, so it is a real change to investigate."
    if verdict == "explained_event":
        return f"The {words} matches the event described in the note, so it is a real change with a known cause."
    if injector in SIGNATURE_REASON:
        confirm = "; the note confirms a data problem" if note_kind.startswith("data_") else ""
        return f"In the review window {SIGNATURE_REASON[injector]}{confirm}."
    return f"The {words} lines up with the data problem described in the note, so it reflects the data, not the business."


def _plan(rng, verdict: str) -> tuple[str, str | None, str]:
    """Draw the injector and the note plan for a verdict."""
    if verdict == "normal":
        return verdict, None, str(rng.choice(["none", "irrelevant", "no_effect"], p=[0.4, 0.3, 0.3]))
    if verdict == "business_change":
        return verdict, str(rng.choice(list(events.BUSINESS))), str(rng.choice(["none", "irrelevant", "opposite"], p=[0.45, 0.35, 0.2]))
    if verdict == "explained_event":
        return verdict, str(rng.choice(list(events.BUSINESS))), "matching_event"
    if rng.random() < 0.5:
        return verdict, str(rng.choice(list(events.DATA_SERIES_SIGNATURE))), str(rng.choice(["data", "none", "irrelevant"], p=[0.5, 0.3, 0.2]))
    return verdict, str(rng.choice(list(events.DATA_NOTE_ONLY))), "data"


def _crop(rng, bg: Background) -> tuple[list[str], np.ndarray]:
    lo, hi = CROP[bg.freq]
    n = min(len(bg), int(rng.integers(lo, hi + 1)))
    a = int(rng.integers(0, len(bg) - n + 1))
    return list(bg.timestamps[a:a + n]), bg.values[a:a + n].astype(float)


def make_example(rng, bg: Background, split: str, idx: int, plan: tuple[str, str | None, str]) -> dict | None:
    timestamps, y = _crop(rng, bg)
    w = WINDOW[bg.freq]
    if len(y) < w + 3 * (bg.period or 1):
        return None
    verdict, injector, note_plan = plan
    s = events.robust_scale(y[:-w], bg.period)
    if injector is None:
        values, label, params = y, Label("normal", "none"), {}
    else:
        inj = events.ALL[injector](y, w, s, rng, bg.freq)
        if inj is None:
            return None
        values, params = inj.values, inj.params
        label = Label(verdict, inj.shape, inj.start, inj.end)

    if note_plan in ("none", "irrelevant", "no_effect"):
        note_kind = note_plan
    elif note_plan == "matching_event":
        note_kind = EVENT_NOTE_FOR_SHAPE[label.shape]
    elif note_plan == "opposite":
        note_kind = OPPOSITE_EVENT_NOTE[label.shape]
    else:  # "data"
        note_kind = DATA_NOTE_FOR_INJECTOR[injector]
    when = "" if label.start is None else when_phrase(rng, timestamps, w, label.start, label.shape, bg.freq)
    note, template = write_note(rng, note_kind, split, when)

    return {
        "id": f"{split}-{idx:05d}",
        "split": split,
        "source": bg.source,
        "group": bg.group,
        "series_key": bg.key,
        "name": bg.name,
        "unit": bg.unit,
        "freq": bg.freq,
        "period": bg.period,
        "timestamps": timestamps,
        "values": [float(f"{v:.6g}") for v in values],
        "window": w,
        "note": note,
        "note_kind": note_kind,
        "note_template": template,
        "injector": injector,
        "params": params,
        "label": label.to_dict(),
        "reason": _reason(verdict, label.shape, injector, note_kind),
    }


def generate(backgrounds: list[Background], sizes: dict[str, int] | None = None, seed: int = 0) -> dict[str, list[dict]]:
    sizes = sizes or DEFAULT_SIZES
    by_split: dict[str, list[Background]] = {k: [] for k in sizes}
    for bg in backgrounds:
        if split_of(bg) in by_split:
            by_split[split_of(bg)].append(bg)
    out = {}
    for k, (split, n) in enumerate(sizes.items()):
        pool = by_split[split]
        if not pool:
            raise ValueError(f"no backgrounds for split {split!r}")
        rng = np.random.default_rng([seed, k])
        sources = sorted({b.source for b in pool})
        rows = []
        while len(rows) < n:
            # Verdicts cycle for an exact balance. The injector and note plan stay fixed while other
            # series are tried, so events that are harder to place are not quietly under-represented;
            # only a plan that fits nowhere in this split is drawn again.
            plan = _plan(rng, VERDICTS[len(rows) % len(VERDICTS)])
            for _ in range(MAX_TRIES):
                # Sources are drawn equally often, so the many CiK series do not crowd out retail.
                src = sources[int(rng.integers(len(sources)))]
                candidates = [b for b in pool if b.source == src]
                row = make_example(rng, candidates[int(rng.integers(len(candidates)))], split, len(rows), plan)
                if row is not None:
                    rows.append(row)
                    break
        out[split] = rows
    return out


def write(rows_by_split: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(rows: list[dict]) -> str:
    lines = [f"{len(rows)} examples"]
    for key in ("source", "group"):
        lines.append(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r[key] for r in rows).items())))
    lines.append("  verdict: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["label"]["verdict"] for r in rows).items())))
    lines.append("  injector: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(str(r["injector"]) for r in rows).items())))
    lines.append("  note: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["note_kind"] for r in rows).items())))
    return "\n".join(lines)
