"""Analyst notes: the short text that comes with a KPI ("IT says the load ran twice on Tuesday").

Each kind has several paraphrases. The last `HELD_OUT` paraphrases of every kind are used only in
the test split, so a system that memorised the training phrasings, or a keyword list built from
them, is caught out. `{when}` is filled with the event's timing: a date, or a relative phrase.
"""

from __future__ import annotations

from datetime import datetime

HELD_OUT = 2

NOTES: dict[str, tuple[str, ...]] = {
    # Nothing to do with the KPI's movement.
    "irrelevant": (
        "The client's fiscal year ends in March.",
        "The dashboard colours were updated last week.",
        "The next quarterly review is scheduled for the 15th.",
        "The KPI owner moved to another team.",
        "Please use the new slide template for the commentary.",
        "The board meeting moved to Thursday.",
        "Budget planning for next year starts soon.",
        "A new analyst joined the account team.",
    ),
    # Mentions a possible cause, but nothing that could have moved the series in this window.
    "no_effect": (
        "A promotion is planned for next month.",
        "IT re-ran last night's load as a precaution; the totals were unchanged.",
        "A competitor announced an opening for next year.",
        "Price changes are being discussed, but nothing has been decided.",
        "The sensors were recalibrated; readings were checked and unaffected.",
        "The holiday staffing plan for December was approved.",
        "A system upgrade is scheduled for the next quarter; no data is affected yet.",
        "Marketing is preparing a campaign that has not launched.",
    ),
    # Real events that push the KPI up for a short time.
    "event_spike": (
        "Marketing ran a one-day flash sale {when}.",
        "A large promotional email went out {when}.",
        "We were featured on national TV {when}, which brought a burst of demand.",
        "A major customer placed a one-off bulk order {when}.",
        "A big public event nearby drew unusual crowds {when}.",
        "A limited-time discount ran {when} only.",
    ),
    # Real events that push the KPI down for a short time.
    "event_dip": (
        "The site was closed for a public holiday {when}.",
        "A storm disrupted operations {when}.",
        "A staff strike stopped work {when}; operations resumed afterwards.",
        "A power cut shut the location {when}.",
        "The location closed for stock-taking {when}.",
        "A road closure blocked access to the site {when}.",
    ),
    # Real events that move the KPI up and keep it there.
    "event_level_up": (
        "A new pricing plan took effect {when}, and demand has been higher since.",
        "We opened a new sales channel {when}.",
        "A major new client went live {when}.",
        "A nearby competitor closed {when}, and we took over their customers.",
        "Capacity was expanded {when}.",
        "A permanent price cut was introduced {when}.",
    ),
    # Real events that move the KPI down and keep it there.
    "event_level_down": (
        "Prices were raised {when}, and volumes have been lower since.",
        "A competitor opened nearby {when}.",
        "We lost our largest client {when}.",
        "The best-selling product was discontinued {when}.",
        "Opening hours were cut back {when}.",
        "A new regulation limiting sales came into force {when}.",
    ),
    # Data problems: records missing or the feed down.
    "data_missing": (
        "The data feed from the source system failed {when}; some records were not loaded.",
        "IT reports an outage of the reporting pipeline {when}.",
        "Only part of the locations reported {when}; the rest will be backfilled.",
        "The nightly extract was incomplete {when}.",
        "Figures {when} are provisional: late-arriving data has not been loaded yet.",
        "The connection to the source database dropped {when}, so data is missing.",
    ),
    # Data problems: records counted twice.
    "data_duplicate": (
        "The load job ran twice {when}, so some records are duplicated.",
        "A reprocessing error double-counted records {when}.",
        "IT re-imported a file {when} without removing the first copy.",
        "Duplicate entries were found in the source table {when}.",
        "The pipeline retried after a timeout {when} and appended the same batch again.",
        "Finance flagged a double posting {when}.",
    ),
    # Data problems: the meaning or unit of the number changed.
    "data_definition": (
        "The reporting system was migrated {when}; the new system counts differently.",
        "The KPI definition changed {when}; figures before and after are not comparable.",
        "The data source was switched {when}, and its figures are calculated another way.",
        "The unit in the export changed {when}.",
        "A new CRM went live {when} and the extraction logic changed.",
        "Finance changed how the figure is reported {when}.",
    ),
    # Data problems: the value stopped updating.
    "data_stuck": (
        "The feed froze {when} and kept repeating its last value.",
        "Values {when} were carried forward because the source was unavailable.",
        "Monitoring raised an alert: no new data arrived {when}.",
        "The export kept re-sending the same value {when}.",
        "The meter stopped updating {when}.",
        "The dashboard showed a frozen feed {when}.",
    ),
}

# Which note kind explains which change. A business note in the opposite direction explains nothing.
EVENT_NOTE_FOR_SHAPE = {"spike": "event_spike", "dip": "event_dip", "level_up": "event_level_up", "level_down": "event_level_down"}
OPPOSITE_EVENT_NOTE = {"spike": "event_dip", "dip": "event_spike", "level_up": "event_level_down", "level_down": "event_level_up"}
DATA_NOTE_FOR_INJECTOR = {
    "zero_outage": "data_missing",
    "partial_load": "data_missing",
    "duplicate_load": "data_duplicate",
    "definition_change": "data_definition",
    "unit_change": "data_definition",
    "flatline": "data_stuck",
}


def paraphrases(kind: str, split: str) -> tuple[str, ...]:
    """Training and validation use the first paraphrases of a kind; the test split the held-out ones."""
    texts = NOTES[kind]
    return texts[-HELD_OUT:] if split == "test" else texts[:-HELD_OUT]


def _when_date(ts: str, freq: str) -> str:
    d = datetime.fromisoformat(ts)
    if freq == "hourly":
        return f"{d.day} {d.strftime('%B')} at {d.strftime('%H:00')}"
    return f"{d.day} {d.strftime('%B %Y')}"


def when_phrase(rng, timestamps, window: int, start: int, shape: str, freq: str) -> str:
    """How the note says when: usually the date the event started, sometimes a relative phrase."""
    ts = timestamps[len(timestamps) - window + start]
    sustained = shape in ("level_up", "level_down")
    if rng.random() < 0.7:
        return ("from " if sustained else "on ") + _when_date(ts, freq)
    return str(rng.choice(["during the review window", "in the latest period", "recently"]))


def write_note(rng, kind: str, split: str, when: str = "") -> tuple[str, str]:
    """(note text, template id). kind "none" gives an empty note."""
    if kind == "none":
        return "", "none"
    texts = paraphrases(kind, split)
    i = int(rng.integers(len(texts)))
    template_index = NOTES[kind].index(texts[i])
    return texts[i].replace("{when}", when).strip(), f"{kind}.{template_index}"
