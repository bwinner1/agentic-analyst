"""Generate one greedy OpenTSLM answer per row for local KPI triage scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import BASE, read_rows, resolve_device, seed_everything, write_json
from train_triage import collate, load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--base", help=f"Default: checkpoint training base, or {BASE}")
    parser.add_argument("--ckpt", required=True, help="best_model.pt, or none for the untuned base")
    parser.add_argument("--split", choices=("train", "val", "validation", "test"), default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--lora_r", type=int, help="Override adapter rank; otherwise read checkpoint's train_log.json")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("max_new_tokens must be positive")
    return args


def checkpoint_rank(ckpt: str, explicit: int | None) -> int:
    """Recover training rank without allocating the checkpoint twice."""
    if explicit is not None:
        return explicit
    log = Path(ckpt).parent / "train_log.json"
    if log.is_file():
        return int(json.loads(log.read_text(encoding="utf-8"))["args"]["lora_r"])
    return 16


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from TriageQADataset import TriageQADataset

    args.device = resolve_device(args.device)
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    untuned = args.ckpt.lower() == "none"
    if args.base is None:
        log = Path(args.ckpt).parent / "train_log.json"
        args.base = (json.loads(log.read_text(encoding="utf-8"))["args"]["base"]
                     if not untuned and log.is_file() else BASE)
    rank = None if untuned else checkpoint_rank(args.ckpt, args.lora_r)
    args.lora_r = rank
    write_json(out / "settings.json", vars(args))
    model = load_model(args.base, args.device, rank)
    if not untuned:
        model.load_from_file(args.ckpt)
    model.eval()
    TriageQADataset.data_dir, TriageQADataset.max_samples = args.data_dir, args.max_samples
    # Answer with the prompt the checkpoint was trained on.
    log = Path(args.ckpt).parent / "train_log.json"
    TriageQADataset.with_values = bool(not untuned and log.is_file()
                                       and json.loads(log.read_text(encoding="utf-8"))["args"].get("with_values"))
    print(f"Series values also as text: {TriageQADataset.with_values}", flush=True)
    split = "validation" if args.split == "val" else args.split
    dataset = TriageQADataset(split, EOS_TOKEN=model.get_eos_token())
    rows = read_rows(args.data_dir, split, args.max_samples)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(rows):
            text = ""
            try:
                batch = collate([dataset[index]])
                text = model.generate(batch, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens)[0]
                if not isinstance(text, str):
                    raise TypeError("OpenTSLM.generate did not return text")
            except Exception as exc:
                text = ""
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Generation failed for {row['id']}: {type(exc).__name__}: {exc}", flush=True)
            handle.write(json.dumps({"id": row["id"], "text": text}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(rows)}] {row['id']}: {len(text)} characters", flush=True)


if __name__ == "__main__":
    main()
