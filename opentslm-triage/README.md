# OpenTSLM KPI triage

These scripts use the existing `src/kpi_triage/prompting.py` contract. They do not
need any files copied into the OpenTSLM checkout or any additional packages.

- `TriageQADataset.py` reads `train.jsonl`, `val.jsonl`, and `test.jsonl`, with
  history and review window as two native time series. Configure its class-level
  `data_dir` and optional per-split `max_samples` before use.
- `train_triage.py` trains the encoder, projector and PEFT LoRA adapters. It checks
  validation loss each epoch, stops after `--patience` epochs without improvement,
  and saves `best_model.pt` and `train_log.json`.
- `eval_triage.py` generates greedy answers, defaulting to the test split and
  96 new tokens. Use `--ckpt none` for the untuned model. A failed generation writes
  an empty answer and processing continues.
- `text_baseline.py` trains `meta-llama/Llama-3.2-1B` with PEFT LoRA using loss only
  on answer tokens, then generates on test. `--zero_shot` skips training.
  The default batch size is one with a 2048-token prompt budget, BF16 on supported
  GPUs and activation checkpointing during training. Long prompts lose oldest
  history values first; normalization, review values, note and question stay
  intact. If these protected parts alone exceed the budget, training raises an
  error and evaluation records an empty answer. Increase `--max_prompt_tokens`.
- `common.py` handles paths, JSONL input and seeding.
- `run.sh` runs the VM commands below using the existing CiK environment.
- `test_pipeline.py` checks prompt preservation, loss masks, failure recovery,
  checkpoint selection and early stopping on CPU without downloading a model.

## On the GPU VM

First run the existing CiK setup, if it has not already run. Keep the Hugging Face
login with access to `meta-llama/Llama-3.2-1B` from that setup. Copy this folder and
`src/kpi_triage/` to the VM, keeping them as siblings:

```text
$HOME/r2_work/opentslm-triage/...
$HOME/r2_work/src/kpi_triage/...
$HOME/r2_work/data/{train,val,test}.jsonl
```

The JSONL files come from `python -m kpi_triage generate`; generate locally and
copy the resulting folder to the VM. From `$HOME/r2_work`:

```bash
bash opentslm-triage/run.sh smoke     # train 16/split for one epoch; eval 4; print parsed answers
bash opentslm-triage/run.sh untuned   # OpenTSLM base, no adapters
bash opentslm-triage/run.sh train     # fine-tune with validation, then eval
bash opentslm-triage/run.sh text      # fine-tune text Llama, then eval
bash opentslm-triage/run.sh text0     # zero-shot text Llama
bash opentslm-triage/run.sh trainvalues                 # OpenTSLM, values also as text
OUT=$HOME/r2_work/results/frozen-encoder \
  bash opentslm-triage/run.sh trainvalues --lr_encoder 0  # the same, encoder frozen (best OpenTSLM so far)
bash opentslm-triage/val.sh train     # answer the validation split with a trained run
```

Results of every run are in `docs/round2/kpi-triage-plan.md`. The best OpenTSLM
setting so far is values as text with the encoder frozen: level with the text-only
Llama on test, above the rules. `--with_marks` (two marker series for repeated
values, zeros and unit changes) is built and tested on CPU but has not been run.

Do not replace `run.sh` on the VM while a run is using it: bash reads the script as it
goes, so the running job continues from the wrong place in the new file.

Environment defaults: `VENV=$HOME/cik_work/venv`,
`OPENTSLM_REPO=$HOME/cik_work/OpenTSLM`, `HF_HOME=$HOME/cik_work/hf_cache`,
`DATA=$HOME/r2_work/data`, and `OUT=$HOME/r2_work/results`. Set them to the actual
locations if the earlier setup used another `WORK` directory. Downloads use the
existing `HF_HOME`; results go under `OUT` and temporary/runtime caches under
`$HOME/r2_work`. The runner never copies code into OpenTSLM or `opentslm-cik`.

Extra arguments go to training for `train`/`smoke`, and to the corresponding
single script for other commands. Use the Python scripts directly for custom
evaluation options. For example:

```bash
DATA=/path/to/jsonl OUT=$HOME/r2_work/experiment bash opentslm-triage/run.sh train --epochs 2 --lr_lora 2e-5
python opentslm-triage/eval_triage.py --data_dir "$HOME/r2_work/data" \
  --ckpt "$HOME/r2_work/results/train/best_model.pt" --out "$HOME/r2_work/results/val" --split val
```

For direct Python commands, activate the CiK venv and set
`PYTHONPATH=$HOME/r2_work/src:$OPENTSLM_REPO/src`. The scripts also locate the
sibling `src` directory relative to their own file paths. `--max_samples N` limits
each used split. Both trainers accept `--device` and `--seed`.

## Outputs and scoring

Each run has its own directory under `OUT` (`smoke`, `untuned`, `train`, `text`,
`text0`). Evaluators write `predictions.jsonl`, one `{"id": ..., "text": ...}`
object per input row, plus `settings.json`. Text training also saves `adapter/`
and `train_log.json`. OpenTSLM evaluation gets the LoRA rank from the checkpoint's
neighboring training log; if you move only the checkpoint, pass `--lora_r` for a
nondefault rank.

Download the predictions and score locally with `python -m kpi_triage score`.
Use `python -m kpi_triage score --help` for the scoring CLI's current arguments.
These GPU scripts do not implement scoring.

## API references and verification

The SP API follows the working local `opentslm-cik/train_cik.py:109-119`
(load and enable LoRA), `:143-153` (optimizer groups), `:163` (loss),
and `:200` (save); evaluation follows `opentslm-cik/eval_cik.py:57-59` (load)
and `:79` (generation). Dataset formatting follows
`opentslm-cik/CiKQADataset.py:81-95`.

Upstream inspection confirmed that [OpenTSLMSP.py](https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/opentslm/model/llm/OpenTSLMSP.py)
uses PEFT (`:11-16`, `:90-112`), returns decoded strings from `generate`
(`:287-297`), and saves/restores LoRA with the encoder/projector (`:333-348`).
[QADataset.py](https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/opentslm/time_series_datasets/QADataset.py)
caches formatted splits on each subclass. The dataset invalidates that cache when
its configuration changes. [util.py](https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/opentslm/time_series_datasets/util.py)
pads series in place and accepts `normalize=False`; collation copies samples and
sets that explicitly so window level shifts remain visible. These upstream
details should be checked against the installed VM checkout if its version differs.

Run the local checks with `python -B opentslm-triage/test_pipeline.py`.
No GPU execution is claimed by the local syntax and CPU contract checks.
