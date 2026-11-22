"""Fine-tune OpenTSLM-SP's encoder, projector and LLM LoRA on KPI triage."""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path
from typing import Any

from common import BASE, resolve_device, seed_everything, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--lr_projector", type=float, default=5e-5)
    parser.add_argument("--lr_lora", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--with_values", action="store_true", help="also give the series values as text")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.lora_r) < 1:
        parser.error("epochs, patience, batch_size and lora_r must be positive")
    return args


def load_model(base: str, device: str, lora_r: int | None = None, lora_dropout: float = 0.0) -> Any:
    """Use the model loading and adapter API exercised by train_cik.py."""
    from opentslm import OpenTSLM
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    model = OpenTSLM.load_pretrained(base, device=device) if base.endswith("-sp") else OpenTSLMSP(llm_id=base, device=device)
    if lora_r is not None:
        model.enable_lora(lora_r=lora_r, lora_alpha=2 * lora_r, lora_dropout=lora_dropout)
    return model


def collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pad both series while preserving the prompt contract's shared scale."""
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    # The upstream utility mutates samples; preserve QADataset's cached values.
    return extend_time_series_to_match_patch_size_and_aggregate(
        copy.deepcopy(batch), patch_size=PATCH_SIZE, normalize=False
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from TriageQADataset import TriageQADataset

    args.device = resolve_device(args.device)
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    TriageQADataset.data_dir, TriageQADataset.max_samples = args.data_dir, args.max_samples
    TriageQADataset.with_values = args.with_values
    model = load_model(args.base, args.device, args.lora_r, args.lora_dropout)
    train_ds = TriageQADataset("train", EOS_TOKEN=model.get_eos_token())
    val_ds = TriageQADataset("validation", EOS_TOKEN=model.get_eos_token())
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Training and validation splits must both contain rows")
    print(f"Train: {len(train_ds)}, validation: {len(val_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

    for module in (model.encoder, model.projector):
        for param in module.parameters():
            param.requires_grad = True
    lora = list(model.get_lora_parameters())
    if not lora:
        raise RuntimeError("OpenTSLM did not expose any trainable LoRA parameters")
    groups = [
        {"params": list(model.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(model.projector.parameters()), "lr": args.lr_projector},
        {"params": lora, "lr": args.lr_lora},
    ]
    parameters = [param for group in groups for param in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.03 * total_steps), total_steps)
    best_val, bad_epochs, history = math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, seen, skipped = time.monotonic(), 0.0, 0, 0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = None
            try:
                loss = model.compute_loss(batch)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}, step {step}")
                loss.backward()
            except torch.OutOfMemoryError:
                loss = None
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                skipped += len(batch)
                print(f"Epoch {epoch} step {step}: OOM, skipped {len(batch)} rows", flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(batch)
            seen += len(batch)
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        if not seen:
            raise RuntimeError("No training batches succeeded; reduce batch_size")
        model.eval()
        val_total, val_seen, val_skipped = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    value = model.compute_loss(batch).item()
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    val_skipped += len(batch)
                    print(f"Validation OOM: skipped {len(batch)} rows", flush=True)
                    continue
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                val_total += value * len(batch)
                val_seen += len(batch)
        if not val_seen:
            raise RuntimeError("No validation batches succeeded; cannot select a checkpoint")
        train_loss, val_loss = running / seen, val_total / val_seen
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_rows": seen, "val_rows": val_seen, "train_skipped": skipped,
                        "val_skipped": val_skipped})
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ({time.monotonic() - start:.0f}s)", flush=True)
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            model.store_to_file(str(out / "best_model.pt"))
            print("Saved best_model.pt", flush=True)
        else:
            bad_epochs += 1
        write_json(out / "train_log.json", {"args": vars(args), "history": history, "best_val_loss": best_val})
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement", flush=True)
            break


if __name__ == "__main__":
    main()
