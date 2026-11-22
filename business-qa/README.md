# Business question benchmark

OpenTSLM’s selling point is answering questions about a time series in text, not forecasting. The paper tests
it that way: TSQA, HAR-CoT, Sleep-CoT and ECG-QA-CoT are multiple-choice questions about a series, scored by
accuracy and macro-F1. This folder applies the same format to our use case: the questions a consulting partner
asks about sales data.

## The questions

759 multiple-choice questions (three options each) about US Census retail sales for four consumer goods store
types (clothing, shoes, health and personal care, grocery), monthly, 2014 to 2026, from `data/census_retail/`.

| Type | n | The question | Input |
|---|---|---|---|
| seasonal | 159 | Is this month’s change normal for the season, weaker than every one of the previous 10 years, or stronger? (2020 and 2021 left out) | Up to 10 years of the store type |
| market | 345 | Over the year, did this store type grow faster than, in line with (within 1.0 point), or slower than total retail? | 13 months of the store type and of total retail |
| trend | 255 | Against the same month last year, are sales up more than 2%, down more than 2%, or within 2%? | 13 months of the store type |

- **The answer key is computed by code**, through the same measures the analyst engine checks
  (`src/agentic_analyst/answer_types/in_context.py`), so it is correct by construction. Two answers are also
  checked by hand in `tests/test_business_qa.py`, and every trend answer is recomputed from the series shown.
- **Balanced:** each type has the same number of questions per answer, and option order is shuffled, so always
  picking one letter scores about a third (0.348 for always "A").
- A "peak month" type was dropped: the answer was December in 47 of 48 store-years, so a model could answer
  without reading the data.

## How to run a model

Plain LLM (the paper’s text-only baseline), on a GPU:

    pip install torch transformers accelerate
    huggingface-cli login            # Llama 3.2 is gated: accept its licence on Hugging Face first
    python business-qa/run_hf.py --model meta-llama/Llama-3.2-1B-Instruct --out business-qa/results/llama-3.2-1b.jsonl
    python business-qa/score.py business-qa/results/llama-3.2-1b.jsonl

OpenTSLM: `prompts.opentslm_parts(item)` returns the pre-prompt, one (description, normalized series) pair per
series, and the post-prompt, the same shape `opentslm-cik/CiKQADataset.py` builds for OpenTSLM’s
`PromptWithAnswer`. Write each reply as `{"id": ..., "reply": ...}` per line and score it with `score.py`.

Check the plumbing without a GPU: `python business-qa/run_hf.py --dry-run --limit 50 --out /tmp/dry.jsonl`.

## Status

The questions, scorer and runner are done and tested. **No model has been run yet**: the Llama 3.2 and OpenTSLM
rows need a GPU (Nebius). Rebuild the questions with `PYTHONPATH=src python business-qa/build_questions.py`.
