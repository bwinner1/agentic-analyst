"""Run the engine on every store type and month, and write answers.json for the demo page.

The page only displays what the engine answered: it computes nothing itself.
    PYTHONPATH=src python prototypes/market-context/build_data.py
"""
import json
from pathlib import Path

from agentic_analyst import ask, load_census_retail
from agentic_analyst.answer_types.in_context import reference_changes
from agentic_analyst.sources.census_retail import CATEGORIES, MARKET_NAME

OUT = Path(__file__).parent / "answers.json"


def main() -> None:
    out = {"market": MARKET_NAME, "source": "U.S. Census Bureau, Monthly Retail Trade, via FRED. Snapshot of 22 Nov 2026. "
           "Not seasonally adjusted, current dollars.", "categories": []}
    for series_id, name in CATEGORIES.items():
        s = load_census_retail(series_id)
        answers = {}
        for i, d in enumerate(s.dates):
            month = f"{d:%B %Y}"
            a = ask(s, f"Is {month} normal?")
            if a.status != "answered":
                continue
            shown = {c.claim.measure: c.claim.shown for c in a.claims}
            answers[f"{d:%Y-%m}"] = {
                "text": a.text, "path": list(a.calculation_path), "shown": shown,
                "checked": [sum(c.passed for c in a.claims), len(a.claims)],
                "reference": [[y, round(c * 100, 1)] for y, c in reference_changes(s, i)],
            }
        out["categories"].append({"id": series_id, "name": name, "unit": s.unit,
                                  "dates": [f"{d:%Y-%m}" for d in s.dates], "values": list(s.values),
                                  "answers": answers})
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    n = sum(len(c["answers"]) for c in out["categories"])
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {n} answers)")


if __name__ == "__main__":
    main()
