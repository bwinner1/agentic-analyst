"""Write a short business explanation for every CiK instance, to train the "Explanation:" part.

CiK has no explanations, so a strong LLM writes one per instance once. It sees the context,
the recent history and the true future, and explains for a business reader what in the context
drives the future values. Spot-check ~30 by hand before training on them.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python make_rationales.py --out data/cik_rationales.jsonl            # all 355 instances
    python make_rationales.py --out data/cik_rationales.jsonl --limit 5  # try a few first

Cost: about 355 short requests, roughly a few dollars. The file is resumable: instances
already in --out are skipped.
"""

import argparse
import json
import os
import sys

import anthropic
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from opentslm.time_series_datasets.cik import cik_core  # inside the OpenTSLM repo
except ImportError:
    import cik_core  # standalone

MODEL = "claude-opus-5-5"

INSTRUCTIONS = """You write short explanations that a consulting partner would give a client about a forecast.

You will get the business context of a time series, its recent history and the values that actually happened next.
Write 2 to 3 plain sentences that explain what in the context and the history drives those future values:
the direction, roughly how big the change is, when it happens, and why. Write it as an explanation of a forecast,
not as hindsight: never say "actually", "the true values" or "the data shows". Do not list the numbers.
Reply with the explanation only."""


def summarize(values, max_points=60):
    v = np.asarray(values, dtype=float)
    tail = v[-max_points:]
    return ", ".join(f"{x:.3g}" for x in tail)


def build_request(row) -> str:
    past = cik_core.parse_frame(row["past_time"])
    future = cik_core.parse_frame(row["future_time"])
    parts = [f"Context:\n{cik_core.context_text(row) or '(none)'}"]
    for col in past:
        parts.append(f"History of '{col.name}' (last {min(60, len(col.values))} of {len(col.values)} points, "
                     f"{col.stamps[0]} to {col.stamps[-1]}):\n{summarize(col.values)}")
    for col in future:
        parts.append(f"What happened next for '{col.name}' ({len(col.values)} points, "
                     f"{col.stamps[0]} to {col.stamps[-1]}):\n{summarize(col.values, 200)}")
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=cik_core.HF_PARQUET)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    rows = cik_core.load_rows(args.data)
    done = set(cik_core.load_rationales(args.out).keys()) if os.path.exists(args.out) else set()
    todo = [r for r in rows if (r["name"], int(r["seed"])) not in done][: args.limit]
    print(f"{len(done)} already done, writing {len(todo)}")

    client = anthropic.Anthropic()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            try:
                response = client.beta.messages.create(
                    model=MODEL,
                    max_tokens=2000,
                    system=INSTRUCTIONS,
                    output_config={"effort": "low"},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",  # if the model declines, the API retries on a fallback model
                    messages=[{"role": "user", "content": build_request(row)}],
                )
            except anthropic.RateLimitError:
                print("Rate limited; rerun later to continue (finished rows are saved).")
                break
            except anthropic.APIStatusError as e:
                print(f"[{i}] {row['name']} seed {row['seed']}: API error {e.status_code}, skipped")
                continue
            if response.stop_reason == "refusal":
                print(f"[{i}] {row['name']} seed {row['seed']}: declined, skipped")
                continue
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if not text:
                continue
            f.write(json.dumps({"task_name": row["name"], "seed": int(row["seed"]), "rationale": text}) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {row['name']} seed {row['seed']}: {text[:90]}...")


if __name__ == "__main__":
    main()
