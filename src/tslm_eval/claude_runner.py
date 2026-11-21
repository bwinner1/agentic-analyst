"""Claude as a comparison system: the series as text in the prompt, with or without the CiK context.

Responses are written to a JSONL prediction file as they arrive (field "text"), so an interrupted
run resumes where it stopped and a finished run is never paid for twice. Scoring parses the text
like any other system's output (forecasts.parse_forecast_text).

Server-side model fallback is deliberately not enabled: a benchmark score must come from the one
model it names, so a refusal is recorded and scored as a failed forecast instead.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

MODEL = "claude-opus-5-5"
EFFORT = "medium"  # the model's default; set explicitly so it is recorded
MAX_TOKENS = 16000
PROMPT_VERSION = "v1"
PRICE_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50),
                  "claude-sonnet-5-5": (2.00, 10.00)}

SYSTEM_PROMPT = (
    "You are an expert forecaster. You are given the recent history of a time series and, when "
    "available, text describing the series and events that affect its future. Forecast the value "
    "at each requested future timestamp. When text is given, use it: it can describe events that "
    "change the future in ways the history alone does not show."
)


def _fmt(v: float) -> str:
    return format(float(v), ".6g")


def build_prompt(inst, with_context: bool = True) -> str:
    parts = []
    if with_context and inst.context_text():
        parts.append(f"<context>\n{inst.context_text()}\n</context>")
    history = "\n".join(f"{t},{_fmt(v)}" for t, v in zip(inst.past_times, inst.past))
    parts.append(f"<history>\ntimestamp,value\n{history}\n</history>")
    targets = "\n".join(inst.future_times)
    parts.append(f"Forecast the {inst.horizon} values at these timestamps:\n{targets}")
    parts.append(
        "Reply with the forecast inside <forecast></forecast> tags: one line per timestamp, "
        "formatted `timestamp,value`, in the order given, and nothing else inside the tags."
    )
    return "\n\n".join(parts)


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _ask(client, inst, with_context: bool, model: str, effort: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(inst, with_context)}],
        output_config={"effort": effort},
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return {
        "id": inst.id,
        "text": text,
        "stop_reason": response.stop_reason,
        "served_by": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _error_types():
    try:
        import anthropic
    except ImportError:  # only the fake client in tests
        return ()
    # Most specific first: a 400 will fail again on rerun; the others are worth retrying.
    return (anthropic.BadRequestError, anthropic.RateLimitError,
            anthropic.APIStatusError, anthropic.APIConnectionError)


def estimate_cost(n: int, model: str = MODEL, in_tok: int = 2500, out_tok: int = 3000) -> float:
    """Rough upper estimate in USD: ~2.5k prompt tokens and ~3k output tokens (thinking included)."""
    p_in, p_out = PRICE_PER_MTOK.get(model, (0.0, 0.0))
    return n * (in_tok * p_in + out_tok * p_out) / 1e6


def run(instances, out_path: Path, *, client=None, with_context: bool = True, model: str = MODEL,
        effort: str = EFFORT, workers: int = 8, log=print) -> dict:
    """Ask Claude for every instance not yet in out_path. Returns usage totals for this call."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [i for i in instances if i.id not in _done_ids(out_path)]
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=5)
    errors = _error_types()
    lock = threading.Lock()
    totals = {"done": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ask, client, inst, with_context, model, effort): inst for inst in todo}
        for fut in as_completed(futures):
            inst = futures[fut]
            try:
                rec = fut.result()
            except errors as exc:  # not written, so a rerun retries it
                totals["failed"] += 1
                log(f"  {inst.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if rec["served_by"] != model:
                log(f"  {inst.id}: served by {rec['served_by']}, not {model}", file=sys.stderr)
            with lock, out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            totals["done"] += 1
            totals["input_tokens"] += rec["input_tokens"]
            totals["output_tokens"] += rec["output_tokens"]
            if totals["done"] % 25 == 0:
                log(f"  {totals['done']}/{len(todo)} done")

    p_in, p_out = PRICE_PER_MTOK.get(model, (0.0, 0.0))
    totals["cost_usd"] = (totals["input_tokens"] * p_in + totals["output_tokens"] * p_out) / 1e6
    _write_meta(out_path, model, effort, with_context, totals)
    return totals


def _write_meta(out_path: Path, model: str, effort: str, with_context: bool, totals: dict) -> None:
    meta_path = out_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"runs": []}
    meta.update({"model": model, "effort": effort, "with_context": with_context,
                 "prompt_version": PROMPT_VERSION, "max_tokens": MAX_TOKENS})
    meta["runs"].append({"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **totals})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
