"""Train a text-only Llama LoRA baseline, then generate test predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import add_repo_src, read_rows, resolve_device, seed_everything, write_json

BASE = "meta-llama/Llama-3.2-1B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.max_prompt_tokens, args.max_new_tokens, args.lora_r) < 1:
        parser.error("epochs, batch_size, token limits and lora_r must be positive")
    return args


def prompt_ids(row: dict[str, Any], tokenizer: Any, max_tokens: int) -> tuple[list[int], str, int]:
    """Use text_prompt verbatim, removing only oldest history values if needed."""
    add_repo_src()
    from kpi_triage.prompting import text_prompt

    prompt, answer = text_prompt(row)
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return ids, answer, 0
    # Keep the original header, scale, review window, note and question unchanged.
    # Rebuilding text_prompt from a shorter row would change its normalization.
    marker = "\nHistory before the review window: "
    prefix, found, remainder = prompt.partition(marker)
    history, review_marker, suffix = remainder.partition("\nReview window:")
    if not found or not review_marker:
        raise ValueError("text_prompt history layout changed; cannot safely truncate")
    values = history.split(", ") if history else []

    def encode_after_dropping(count: int) -> list[int]:
        shortened = prefix + marker + ", ".join(values[count:]) + review_marker + suffix
        return tokenizer.encode(shortened, add_special_tokens=True)

    if len(encode_after_dropping(len(values))) > max_tokens:
        raise ValueError(f"{row['id']}: review window and question exceed max_prompt_tokens={max_tokens}")
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if len(encode_after_dropping(middle)) <= max_tokens:
            high = middle
        else:
            low = middle + 1
    ids = encode_after_dropping(low)
    return ids, answer, low


def training_sample(row: dict[str, Any], tokenizer: Any, max_tokens: int) -> tuple[dict[str, list[int]], int]:
    """Mask all prompt tokens; supervise the complete answer and its EOS."""
    ids, answer, dropped = prompt_ids(row, tokenizer, max_tokens)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    return {"input_ids": ids + answer_ids, "labels": [-100] * len(ids) + answer_ids}, dropped


def collate_text(samples: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad training inputs and ignore every padding token in the loss."""
    import torch

    length = max(len(sample["input_ids"]) for sample in samples)
    return {
        "input_ids": torch.tensor([sample["input_ids"] + [pad_token_id] * (length - len(sample["input_ids"]))
                                   for sample in samples], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(sample["input_ids"]) + [0] * (length - len(sample["input_ids"]))
                                        for sample in samples], dtype=torch.long),
        "labels": torch.tensor([sample["labels"] + [-100] * (length - len(sample["labels"]))
                                for sample in samples], dtype=torch.long),
    }


def train(model: Any, tokenizer: Any, args: argparse.Namespace, out: Path) -> None:
    """Fit PEFT adapters with bounded sequences and activation checkpointing."""
    from functools import partial

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    rows = read_rows(args.data_dir, "train", args.max_samples)
    if not rows:
        raise ValueError("Training split is empty")
    samples, dropped_rows = [], 0
    for row in rows:
        sample, dropped = training_sample(row, tokenizer, args.max_prompt_tokens)
        if len(sample["input_ids"]) > model.config.max_position_embeddings:
            raise ValueError(f"{row['id']}: prompt plus full answer exceeds model context length")
        samples.append(sample)
        dropped_rows += int(dropped > 0)
    print(f"Text training: {len(samples)} rows; history shortened in {dropped_rows}", flush=True)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True,
                        collate_fn=partial(collate_text, pad_token_id=tokenizer.pad_token_id))
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.03 * total_steps), total_steps)
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    val_samples = [training_sample(row, tokenizer, args.max_prompt_tokens)[0]
                   for row in read_rows(args.data_dir, "val", args.max_samples)]
    if not val_samples:
        raise ValueError("Validation split is empty")
    val_loader = DataLoader(val_samples, batch_size=args.batch_size,
                            collate_fn=partial(collate_text, pad_token_id=tokenizer.pad_token_id))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    history, best_val, bad_epochs, best_state = [], float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, skipped = 0.0, 0, 0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss = None
            try:
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite text training loss")
                loss.backward()
            except torch.OutOfMemoryError:
                loss = None
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                skipped += batch["input_ids"].shape[0]
                print(f"Text epoch {epoch} step {step}: OOM, skipped batch", flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            count = batch["input_ids"].shape[0]
            running += loss.item() * count
            seen += count
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"Text epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}", flush=True)
        if not seen:
            raise RuntimeError("No text training batches succeeded; reduce batch_size or max_prompt_tokens")
        # Same checkpoint choice as train_triage.py: keep the epoch with the lowest validation loss.
        model.eval()
        val_total, val_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(args.device) for key, value in batch.items()}
                count = batch["input_ids"].shape[0]
                val_total += model(**batch, use_cache=False).loss.item() * count
                val_seen += count
        val_loss = val_total / val_seen
        history.append({"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss,
                        "train_rows": seen, "skipped": skipped})
        print(f"Epoch {epoch}: train={running / seen:.4f} val={val_loss:.4f} (text)", flush=True)
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(out / "adapter")
            tokenizer.save_pretrained(out / "adapter")
        else:
            bad_epochs += 1
        write_json(out / "train_log.json", {"args": vars(args), "history": history, "best_val_loss": best_val,
                                           "history_shortened_rows": dropped_rows})
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement (text)", flush=True)
            break
    set_peft_model_state_dict(model, best_state)
    model.gradient_checkpointing_disable()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.device = resolve_device(args.device)
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device)
    if args.max_prompt_tokens + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("max_prompt_tokens + max_new_tokens exceeds the model context length")
    if not args.zero_shot:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("PEFT is required for LoRA, as in OpenTSLMSP.enable_lora; use its setup environment") from exc
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        ))
        train(model, tokenizer, args, out)
    model.config.use_cache = True
    model.eval()
    write_json(out / "settings.json", {**vars(args), "base": BASE, "split": "test",
                                       "ckpt": "none" if args.zero_shot else str(out / "adapter")})
    rows = read_rows(args.data_dir, "test", args.max_samples)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(rows, 1):
            text, dropped = "", 0
            try:
                ids, _, dropped = prompt_ids(row, tokenizer, args.max_prompt_tokens)
                input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
                generated = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Text generation failed for {row['id']}: {type(exc).__name__}: {exc}", flush=True)
            handle.write(json.dumps({"id": row["id"], "text": text}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {row['id']}: {len(text)} characters, history values removed={dropped}", flush=True)


if __name__ == "__main__":
    main()
