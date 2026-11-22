# Training guide: OpenTSLM on Context is Key

Code version: **2026-11-21 extra-data v5** (also in the `VERSION` file; `train_cik.py` prints it first).

Code to train an OpenTSLM-SP model to read a time series plus its business context, then write a
forecast and a short business explanation. The data is the Context is Key (CiK) benchmark.

## What is in this folder

| File | What it does |
| --- | --- |
| `cik_core.py` | Reads CiK from TimeNet (or Hugging Face), turns rows into prompt/answer pairs, splits by task type, parses model output, scores forecasts. No GPU or torch needed. |
| `CiKQADataset.py` | OpenTSLM dataset class (same pattern as the repo's `M4QADataset`). |
| `cik_rationales.jsonl` | The ready-made business explanation for every one of the 355 CiK instances (the "why" the model learns to write). Training uses it by default. |
| `build_rationales.py` | The script that wrote `cik_rationales.jsonl` from each instance's context and true future. Re-run it if CiK changes. |
| `make_rationales.py` | Optional: rewrite the explanations with the Claude API instead (needs an API key). |
| `train_cik.py` | Finetunes the model and saves the best checkpoint. |
| `eval_cik.py` | Scores a checkpoint on held-out tasks against a seasonal-naive baseline. |
| `run_nuvolos.sh`, `NUVOLOS.md`, `NEBIUS.md` | One script (setup, smoke, train, pack) for any GPU machine with a terminal, and step-by-step guides for Nuvolos and Nebius (`full` = main run + no-context comparison). |
| `tests/test_cik_core.py` | Unit tests (`pytest tests`). |

Tested: the unit tests pass, a Colab T4 smoke test ran end to end, and the full runs (main model,
no-context comparison, extra-data run) ran on a Nebius L40S. Their outputs are in `results/`.

## Important: how the data is split

CiK has only 355 instances (71 task types x 5 seeds) and is meant as a test set. Many task types are
rewordings of the same problem on the **same series** (for example, all 7 heat-wave electricity tasks use
identical data). If one wording were in training and another in testing, the model would have already
seen the test answer. So the code first groups task types that share a series (21 groups), then puts
whole groups into one split: 43 task types / 215 instances for training, 7 / 35 for validation and
21 / 105 for testing. Test numbers therefore measure how well the model handles *new kinds* of problems.
Training is heavy on the Montreal fire-incident tasks (100 of 215 instances); keep that in mind when
reading results.

## Step by step

### 1. Get a GPU machine (Nebius)

1. Apply for the hackathon's Nebius voucher (see the hackathon page and Discord).
2. In the Nebius console create a VM with **1x H100** and an Ubuntu + CUDA image. Add your SSH public key.
3. `ssh <user>@<vm-ip>`. Run `nvidia-smi` to confirm the GPU shows up.
4. **Stop the VM whenever you are not using it.** Billing is per hour.

No GPU yet? Steps 2 to 4 also work on a free Google Colab GPU for the smoke test.

### 2. Install OpenTSLM and add these files

```bash
git clone https://github.com/StanfordBDHG/OpenTSLM.git
cd OpenTSLM
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv (Python package manager)
uv sync --all-groups && source .venv/bin/activate
pip uninstall -y torchao 2>/dev/null || true   # an old torchao breaks LoRA ("torchao >0.16 required")

mkdir -p src/opentslm/time_series_datasets/cik
touch src/opentslm/time_series_datasets/cik/__init__.py
cp /path/to/opentslm-cik/cik_core.py /path/to/opentslm-cik/CiKQADataset.py /path/to/opentslm-cik/cik_rationales.jsonl src/opentslm/time_series_datasets/cik/
cp /path/to/opentslm-cik/train_cik.py /path/to/opentslm-cik/eval_cik.py /path/to/opentslm-cik/make_rationales.py /path/to/opentslm-cik/VERSION .
```

### 3. Hugging Face access and data

1. Request access to `meta-llama/Llama-3.2-1B` on Hugging Face (approval is usually quick). Without
   it, use a Gemma base instead (see Troubleshooting).
2. Create a read token at huggingface.co/settings/tokens, then run `huggingface-cli login`.
3. Build CiK into TimeNet with your connector. Until PR 788 is merged, install TimeNet from your PR
   branch:

```bash
git clone https://github.com/OpenTSLM/TimeNet.git ../TimeNet
cd ../TimeNet && git fetch origin pull/788/head:cik && git checkout cik && cd -
pip install -e ../TimeNet/packages/timenet -e ../TimeNet/packages/timenet-connectors
timenet-build build servicenow/context-is-key
```

   This downloads CiK from Hugging Face, converts it with your connector, and stores it in
   `~/.timenet/registry`. The training scripts then read it through TimeNet (`--source auto`, the default).
   If TimeNet is not set up, they fall back to Hugging Face and print a message saying so. To skip
   TimeNet entirely, download the parquet and pass `--source hf --data <file>`:

```bash
huggingface-cli download ServiceNow/context-is-key data/test-00000-of-00001.parquet \
  --repo-type dataset --local-dir data/cik
```

4. Check the stack works: `python demo/huggingface/02_test_hf_m4.py` should print captions.

### 4. Explanations (already done)

`cik_rationales.jsonl` already holds a short business explanation for all 355 instances, written from
each instance's context and its true future values (for example: "An accident cuts traffic to 20% of its
usual level from 06:00 for 5 hours..."). Copying it in step 2 is all you need; training reads it
automatically. Open the file and read a few lines to see what the model will learn to write.

Optional: to regenerate them with the Claude API instead, set `ANTHROPIC_API_KEY` and run
`python make_rationales.py --out data/cik_rationales_api.jsonl`, then pass `--rationales data/cik_rationales_api.jsonl`.

### 5. Smoke test (minutes)

```bash
python train_cik.py --out results/smoke --max_samples 8 --epochs 1
```

The first lines should say `Loaded 355 CiK instances from TimeNet`. You should see a loss printed and `results/smoke/best_model.pt` saved.

### 6. Real training runs (about 20 to 60 minutes each)

```bash
python train_cik.py --out results/full
python train_cik.py --out results/nocontext --no_context   # the ablation
```

Watch the validation loss. With only 215 training examples the model starts memorizing quickly, so the
best checkpoint is usually from epoch 1 or 2. The defaults are set for that (4 epochs at most, stop after
2 epochs without improvement, gentle learning rates, a little LoRA dropout). Training loss falling to
near 0 while validation loss rises means memorizing; the saved `best_model.pt` is from before that.

### 7. Evaluate

```bash
python eval_cik.py --ckpt results/full/best_model.pt --out results/full
python eval_cik.py --ckpt results/nocontext/best_model.pt --no_context --out results/nocontext
python eval_cik.py --ckpt none --out results/untuned          # no finetuning
```

Each writes `metrics.json` (model vs seasonal naive) and `test_predictions.jsonl` (forecast, truth and
explanation per instance). The headline result is **full vs nocontext**: how much the business context
improves the forecast. Lower `mae` and `crps` are better. `parsed_ok_rate` is the share of outputs
that came back as the right number of values.

### 8. Save your work

Copy `results/` off the VM (`scp -r`) or upload `best_model.pt` to Hugging Face before deleting the VM.

## Troubleshooting

- **`torchao >0.16 required` when LoRA is switched on:** run `pip uninstall -y torchao`. We don't use it.
- **`ModuleNotFoundError: No module named 'timenet'` right after installing:** restart Python (in Colab:
  Runtime > Restart session; on Nebius: open a new terminal or re-run `source .venv/bin/activate`).
- **`parsed_ok` is False in the smoke test:** normal after 8 examples; the model has not learned the answer
  format yet. The eval prints what the model wrote. After a full run it should be close to 1.0.

- **Out of GPU memory:** add `--batch_size 1` and/or `--max_history 256`.
- **No Llama access:** `--base google/gemma-3-1b-pt` starts from the raw Gemma LLM (no pretrained
  OpenTSLM stage), or use an `OpenTSLM/gemma-3-1b-pt-m4-sp` checkpoint if one exists on the Hub. Pass the
  same `--base` to `eval_cik.py`.
- **Low `parsed_ok_rate`:** the model writes one numbered line per forecast step (`1: 0.12`). Open
  `test_predictions.jsonl` and read `raw_output` to see what went wrong. Long horizons (72 to 99 steps) are hardest.
- **Validation loss rises from epoch 2:** normal on this small dataset; try `--lr_lora 2e-5 --lora_dropout 0.1`.
- **Explanations:** the model learns the style from `cik_rationales.jsonl`. If you edit that file, keep one JSON object per line with `task_name`, `seed` and `rationale`.

## Numbers to report

`crps` here approximates CiK's RCRPS (region of interest weighted to half, scaled by `metric_scaling`).
For numbers comparable to the CiK paper, run the official `compute_rcrps_with_hf_dataset.py` from the
CiK Hugging Face repo on the same test instances.
