"""Finetune an OpenTSLM-SP model on Context is Key.

Run from the OpenTSLM repo root after copying the cik/ files in (see README.md):

    python train_cik.py --out results/cik_full
    python train_cik.py --out results/cik_nocontext --no_context      # ablation
    python train_cik.py --out results/smoke --max_samples 8 --epochs 1  # smoke test

What it does: loads a pretrained OpenTSLM-SP checkpoint (default: the M4 captioning one),
switches on a LoRA adapter on the LLM, and trains the time-series encoder, the projector and
the LoRA weights on the CiK train split. The LLM's own weights stay frozen. The checkpoint
with the lowest validation loss is saved to <out>/best_model.pt.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

# Let PyTorch reuse fragmented GPU memory (fewer out-of-memory crashes on 16 GB cards like the T4).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from opentslm import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.cik.CiKQADataset import CiKQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="OpenTSLM/llama-3.2-1b-m4-sp",
                   help="Pretrained OpenTSLM repo on Hugging Face (must end in -sp), or a raw LLM id "
                        "such as meta-llama/Llama-3.2-1B to start from scratch.")
    add_data_args(p)
    p.add_argument("--rationales", default=DEFAULT_RATIONALES,
                   help="JSONL of business explanations (task_name, seed, rationale). Default: the "
                        "cik_rationales.jsonl that ships with this code. Pass 'none' to train forecasts only.")
    p.add_argument("--no_context", action="store_true", help="Remove the text context (ablation).")
    p.add_argument("--extra_train", default=None, help="Extra generated training rows (parquet from cik_extra.py).")
    p.add_argument("--extra_rationales", default=None, help="Explanations for --extra_train (build_rationales.py).")
    p.add_argument("--no_baseline", action="store_true",
                   help="Don't show the seasonal-naive forecast in the prompt (default: show it).")
    p.add_argument("--out", default="results/cik")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=2, help="Stop after this many epochs without val improvement.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr_encoder", type=float, default=1e-4)
    p.add_argument("--lr_projector", type=float, default=5e-5)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05, help="Randomly drops part of the LoRA signal while training, against memorizing.")
    p.add_argument("--max_history", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None, help="Use only N rows per split (smoke test).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _default_rationales():
    """The cik_rationales.jsonl that ships with this code: in the cik/ dataset folder, or next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "src", "opentslm", "time_series_datasets", "cik", "cik_rationales.jsonl"),
        os.path.join(here, "cik_rationales.jsonl"),
        os.path.join(here, "data", "cik_rationales.jsonl"),
    ]
    return next((c for c in candidates if os.path.exists(c)), candidates[0])


DEFAULT_RATIONALES = _default_rationales()


def add_data_args(p):
    p.add_argument("--source", default="auto", choices=["auto", "timenet", "hf"],
                   help="Where to read CiK: TimeNet (needs `timenet-build build servicenow/context-is-key`), "
                        "Hugging Face, or auto = TimeNet first, Hugging Face if that fails.")
    p.add_argument("--timenet_ref", default="servicenow/context-is-key@1.0.0")
    p.add_argument("--timenet_registry", default=None, help="Local TimeNet registry folder (default ~/.timenet/registry).")
    p.add_argument("--data", default=None, help="Local CiK parquet for --source hf. Default: download from Hugging Face.")


def configure_dataset(args):
    CiKQADataset.source = args.source
    CiKQADataset.timenet_ref = args.timenet_ref
    CiKQADataset.timenet_registry = args.timenet_registry
    if args.data:
        CiKQADataset.parquet_path = args.data
    rationales = None if str(args.rationales).lower() == "none" else args.rationales
    if rationales and not os.path.exists(rationales):
        print(f"No explanations file at {rationales}; training forecasts only")
        rationales = None
    CiKQADataset.rationales_path = rationales
    CiKQADataset.include_context = not args.no_context
    CiKQADataset.extra_train_path = getattr(args, "extra_train", None)
    CiKQADataset.extra_rationales_path = getattr(args, "extra_rationales", None)
    CiKQADataset.show_baseline = not getattr(args, "no_baseline", False)
    CiKQADataset.max_history = args.max_history
    CiKQADataset.max_samples = args.max_samples


def load_model(base: str, device: str, lora_r: int, lora_dropout: float = 0.0):
    if base.endswith("-sp"):
        model = OpenTSLM.load_pretrained(base, device=device)
    else:
        model = OpenTSLMSP(llm_id=base, device=device)
    model.enable_lora(lora_r=lora_r, lora_alpha=2 * lora_r, lora_dropout=lora_dropout)
    return model


def collate(batch):
    return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)


def main():
    args = parse_args()
    vfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    print("Code version:", open(vfile).read().strip() if os.path.exists(vfile) else "unknown (no VERSION file)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    configure_dataset(args)

    model = load_model(args.base, args.device, args.lora_r, args.lora_dropout)
    eos = model.get_eos_token()
    train_ds = CiKQADataset("train", EOS_TOKEN=eos)
    val_ds = CiKQADataset("validation", EOS_TOKEN=eos)
    print(f"Train examples: {len(train_ds)}, validation examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True
    groups = [
        {"params": list(model.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(model.projector.parameters()), "lr": args.lr_projector},
    ]
    lora = model.get_lora_parameters()
    if lora:
        groups.append({"params": lora, "lr": args.lr_lora})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.03 * total_steps), total_steps)

    best_val, bad_epochs, history = math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, done, skipped = time.time(), 0.0, 0, 0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = model.compute_loss(batch)
                loss.backward()
            except torch.OutOfMemoryError:
                # One unusually long example did not fit on the GPU: skip it instead of crashing the run.
                loss = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                print(f"  epoch {epoch} step {step}: out of GPU memory, skipped this example ({batch[0]['meta']['task_name']})")
                scheduler.step()
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            done += 1
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        train_loss = running / max(1, done)
        if skipped:
            print(f"  ({skipped} examples skipped for memory this epoch; use --max_history 128 if this is many)")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for b in val_loader:
                try:
                    val_losses.append(model.compute_loss(b).item())
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  validation: out of GPU memory, skipped {b[0]['meta']['task_name']}")
        val_loss = sum(val_losses) / max(1, len(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f} ({time.time() - start:.0f}s)")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            model.store_to_file(os.path.join(args.out, "best_model.pt"))
            print(f"  saved new best model (val loss {val_loss:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Stopping early: no improvement for {args.patience} epochs")
                break

    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "history": history, "best_val_loss": best_val}, f, indent=2)
    print(f"Done. Best val loss {best_val:.4f}. Checkpoint: {os.path.join(args.out, 'best_model.pt')}")


if __name__ == "__main__":
    main()
