# Evaluation: forecasting with text context (Context is Key)

Scores every system with CiK's official metric (RCRPS) on the same instances. The plan, the systems and the open decisions are in [`docs/evaluation-plan.md`](../docs/evaluation-plan.md). The latest results are in [`results/results.md`](results/results.md).

## Run it

From the repository root:

```
python -m venv .venv && .venv/Scripts/pip install -e ".[eval,test]"   # macOS/Linux: .venv/bin/pip
python -m tslm_eval fetch          # pinned CiK parquet -> evaluation/data/ (sha256-checked, git-ignored)
python -m tslm_eval split          # proposed held-out test set -> evaluation/splits/ (already committed)
python -m tslm_eval baselines      # naive + seasonal-naive -> evaluation/runs/
python -m tslm_eval claude --split test            # prints a cost estimate; add --yes to call the API
python -m tslm_eval claude --split test --no-context --yes
python -m tslm_eval report         # scores everything in evaluation/runs/ -> evaluation/results/
pytest                             # 56 tests, no API key or GPU needed
```

The Claude run needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). It writes each answer as it arrives, so rerunning resumes instead of paying twice.

## Adding a system (OpenTSLM or anything else)

1. `python -m tslm_eval export --split test` writes `evaluation/inputs/cik-test-inputs.jsonl`. Each line has `id`, `context`, `past_times`, `past`, `future_times` and `horizon`, with no answers. `--no-context` blanks the text, and `--split train` writes the other 290 instances *with* their `future` values for training.
2. Run your model on each line and write `evaluation/runs/<system-name>.jsonl`, one line per instance, in any of these shapes:

   ```
   {"id": "ATMBuildingClosedTask/1", "forecast": [31.2, 28.0, ...]}   # exactly `horizon` numbers
   {"id": "ATMBuildingClosedTask/1", "text": "<raw model output>"}    # numbers are parsed out
   {"id": "ATMBuildingClosedTask/1", "samples": [[...], [...]]}       # several sampled paths
   ```

   For example:

   ```python
   import json
   with open("evaluation/inputs/cik-test-inputs.jsonl") as f, open("evaluation/runs/opentslm-finetuned-cik.jsonl", "w") as out:
       for line in f:
           x = json.loads(line)
           text = my_model_generate(x["context"], x["past"], x["horizon"])   # your code
           out.write(json.dumps({"id": x["id"], "text": text}) + "\n")
   ```
3. `python -m tslm_eval report`. A missing instance, wrong length or unreadable text counts as a failed forecast, scored 5 (the cap), and lowers that system's "valid outputs" share.

## What is committed

| Path | Content |
|---|---|
| `splits/test-holdout-groups.txt` | The 65 held-out instance ids. Don't train on these. |
| `runs/*.jsonl` | Each system's predictions. Claude runs are paid for, so keep them. |
| `runs/*.meta.json` | Model, effort, prompt version and cost of each Claude run |
| `results/` | `results.md`, CSVs and the PNG charts for the posts |
| `data/`, `inputs/` | Not committed (downloaded or regenerated) |

CiK data: ServiceNow, Apache-2.0, Hugging Face `ServiceNow/context-is-key` at commit `32da467`.
