"""Evaluate a finetuned OpenTSLM-SP model on the held-out CiK test tasks.

    python eval_cik.py --ckpt results/cik_full/best_model.pt --out results/cik_full
    python eval_cik.py --ckpt results/cik_nocontext/best_model.pt --no_context --out results/cik_nocontext
    python eval_cik.py --ckpt none --out results/m4_untuned       # the base model, no finetuning

Writes <out>/test_predictions.jsonl (forecast, explanation and scores per instance) and
<out>/metrics.json, which also holds the seasonal-naive baseline on the same instances.
Use the same --no_context / --rationales flags as in training.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from opentslm.time_series_datasets.cik import cik_core
from opentslm.time_series_datasets.cik.CiKQADataset import CiKQADataset

from train_cik import DEFAULT_RATIONALES, add_data_args, collate, configure_dataset, load_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="best_model.pt from train_cik.py, or 'none' for the base model.")
    p.add_argument("--base", default="OpenTSLM/llama-3.2-1b-m4-sp")
    add_data_args(p)
    p.add_argument("--rationales", default=DEFAULT_RATIONALES)
    p.add_argument("--no_context", action="store_true")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--out", required=True)
    p.add_argument("--num_samples", type=int, default=10,
                   help="Sampled forecasts per instance (for CRPS). 1 = a single greedy forecast.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--max_history", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def max_tokens_for(meta) -> int:
    horizon = sum(len(t["values"]) for t in meta["targets"])
    return 10 * horizon + 250  # about 7 tokens per "12: -1.23" line plus room for the explanation


def main():
    args = parse_args()
    args.epochs = 0
    configure_dataset(args)
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.base, args.device, args.lora_r)
    if args.ckpt != "none":
        model.load_from_file(args.ckpt)
    model.eval()

    ds = CiKQADataset(args.split, EOS_TOKEN=model.get_eos_token())
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    print(f"Evaluating {len(ds)} instances")

    model_results, naive_results, n_valid, n_samples_ok, n_samples = [], [], 0, 0, 0
    pred_path = os.path.join(args.out, f"{args.split}_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f, torch.no_grad():
        for i, batch in enumerate(loader):
            meta = batch[0]["meta"]
            horizon = sum(len(t["values"]) for t in meta["targets"])
            # min_new_tokens stops the model from ending its answer (EOS) before writing the forecast lines;
            # an undertrained model sometimes does that and returns an empty string.
            gen_kwargs = {"max_new_tokens": max_tokens_for(meta), "min_new_tokens": 4 * horizon}
            if args.num_samples > 1:
                gen_kwargs.update(do_sample=True, temperature=args.temperature)
            texts, forecasts, valid_all, valids = [], [], True, []
            for _ in range(max(1, args.num_samples)):
                text = model.generate(batch, **gen_kwargs)[0]
                fc, valid, expl = cik_core.parse_answer(text, meta)
                texts.append(text)
                forecasts.append(fc)
                valid_all &= valid
                valids.append(bool(valid))
                n_samples_ok += int(valid)
                n_samples += 1
            n_valid += int(valid_all)
            samples = {t["name"]: np.stack([fc[t["name"]] for fc in forecasts]) for t in meta["targets"]}
            scores = cik_core.score_instance(meta, samples)
            naive = cik_core.score_instance(meta, {k: v[None, :] for k, v in cik_core.seasonal_naive(meta).items()})
            model_results.append((meta, scores))
            naive_results.append((meta, naive))
            _, _, explanation = cik_core.parse_answer(texts[0], meta)
            f.write(json.dumps({
                "task_name": meta["task_name"], "seed": meta["seed"],
                "forecast_median": {k: np.median(v, axis=0).tolist() for k, v in samples.items()},
                # every sample and whether it was readable, so ranges can be scored and unreadable ones dropped
                "forecast_samples": {k: v.tolist() for k, v in samples.items()},
                "samples_valid": valids,
                "truth": {t["name"]: t["values"] for t in meta["targets"]},
                "explanation": explanation, "raw_output": texts[0],
                "parsed_ok": valid_all, "scores": scores, "seasonal_naive_scores": naive,
            }) + "\n")
            print(f"[{i + 1}/{len(ds)}] {meta['task_name']} seed {meta['seed']}: "
                  f"mae {scores['mae']:.4f} (naive {naive['mae']:.4f}) parsed_ok={valid_all}")
            if not valid_all:
                print("    model wrote:", " ".join(texts[0].split())[:200])

    metrics = {
        "model": cik_core.aggregate(model_results),
        "seasonal_naive": cik_core.aggregate(naive_results),
        "parsed_ok_rate": n_samples_ok / max(1, n_samples),  # share of generated forecasts with every step present
        "all_samples_parsed_rate": n_valid / max(1, len(ds)),
        "settings": {k: v for k, v in vars(args).items() if k != "epochs"},
        "note": "crps is a sample CRPS weighted like CiK's RCRPS; use CiK's official script for paper-comparable numbers.",
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
