"""Turn a benchmark question into model input, and a model's reply into a letter.

Two input forms, matching the two kinds of model in the OpenTSLM paper's comparison:
- text_prompt(): the numbers written out as text, for a plain LLM such as Llama 3.2 ("tokenized" baseline).
- opentslm_parts(): a pre-prompt, one (description, normalized series) pair per series, and a post-prompt,
  the shape OpenTSLM's PromptWithAnswer takes (see opentslm-cik/CiKQADataset.py for the same pattern).
"""
from __future__ import annotations

import re
from statistics import mean, pstdev

SYSTEM = "You are a business analyst at a consulting firm. Answer from the data shown."
INSTRUCTION = "Answer with the letter of the correct option only (A, B or C)."


def _options(item: dict) -> str:
    return "\n".join(f"{k}) {v}" for k, v in item["options"].items())


def text_prompt(item: dict) -> str:
    blocks = []
    for s in item["series"]:
        rows = "\n".join(f"{d}: {v:,}" for d, v in zip(s["dates"], s["values"]))
        blocks.append(f"{s['name'][:1].upper()}{s['name'][1:]}, monthly, {s['unit']}, not seasonally adjusted:\n{rows}")
    return f"{SYSTEM}\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {item['question']}\n{_options(item)}\n\n{INSTRUCTION}"


def opentslm_parts(item: dict) -> tuple[str, list[tuple[str, list[float]]], str]:
    series = []
    for s in item["series"]:
        m, sd = mean(s["values"]), pstdev(s["values"]) or 1.0
        desc = (f"{s['name']}, monthly from {s['dates'][0]} to {s['dates'][-1]}, {s['unit']}, "
                f"mean {m:,.0f}, std {sd:,.0f} (values normalized):")
        series.append((desc, [(v - m) / sd for v in s["values"]]))
    post = f"Question: {item['question']}\n{_options(item)}\n\n{INSTRUCTION}"
    return SYSTEM, series, post


_LETTER = re.compile(r"\b([ABC])\b")


def parse_letter(reply: str, item: dict | None = None) -> str | None:
    """The first standalone A, B or C; failing that, an option's text quoted in the reply."""
    head = reply.strip()
    m = re.match(r"^\(?([ABC])[)\.:\s]", head + " ")
    if m:
        return m.group(1)
    m = _LETTER.search(head)
    if m:
        return m.group(1)
    if item is not None:
        low = head.lower()
        hits = [k for k, v in item["options"].items() if v.lower() in low]
        if len(hits) == 1:
            return hits[0]
    return None
