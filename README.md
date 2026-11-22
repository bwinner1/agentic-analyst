# Agentic Analyst

A Partner asks a Question about a KPI series and gets an Answer whose every Claim has been recomputed by code
(see `CONTEXT.md` for the terms and `docs/adr/0001-model-spots-code-measures.md` for the design).
The first series is the US trade deficit (Time-MMD `Economy.csv`).

## Run the tests

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

The tests need no downloaded data; they use `tests/fixtures/us_trade_2023.csv`, a small made-up series whose answers are known by construction.

## Try it

```
python3 prototypes/kpi-qa-PROTOTYPE/tools/fetch_data.py     # downloads Time-MMD data (not committed: no licence)
.venv/bin/python -m agentic_analyst.web prototypes/kpi-qa-PROTOTYPE/data/Economy.csv
```

Then open http://127.0.0.1:8766 and ask "What happened in March 2024?" or "What happened in the latest period?".
(`fetch_data.py` also fetches the Swiss series; the web shell needs only `Economy.csv`.)

## Layout

```
src/agentic_analyst/
  ask.py            the 'Ask a Question' seam: ask(series, question, drafter=None) -> Answer
  series.py         the standardized Series
  sources/          one loader per data source (time_mmd_economy.py: US trade deficit)
  period.py         finds the period a Question names ("March 2024", "2024-03", none = latest)
  routing.py        picks the answer type for a Question
  answer_types/     ONE MODULE PER ANSWER TYPE, auto-discovered (what_happened.py)
  measures.py       what code can compute and how it is written; answer types register their own
  check.py          the Check: recompute every Claim, correct and flag the ones that fail
  answer.py         Answer, Claim, Evidence, Draft
  web.py            thin chat box and chart (rendering only)
tests/              tests at the 'Ask a Question' seam, on fixtures
prototypes/         throwaway prototype (kept for reference)
```

## Hackathon evaluation

`src/tslm_eval/` and `evaluation/` score forecasting systems (baselines, Claude, pretrained and fine-tuned OpenTSLM) on Context is Key with the benchmark's own metric.
Start with `docs/evaluation-plan.md` and `evaluation/README.md`.

## Adding an answer type

Add a module to `src/agentic_analyst/answer_types/` with `PATTERN`, optional `PRIORITY` and `answer(series, i, question) -> Draft | None`
(see `answer_types/__init__.py`). Register the measures its Claims use with `measures.register_measure`.
No shared file needs editing. Add tests in a new file under `tests/`.
Loaders for further series go in `src/agentic_analyst/sources/`.
