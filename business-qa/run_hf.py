"""Run a Hugging Face chat model on the business questions (the paper's text-only baseline, e.g. Llama 3.2).

    pip install torch transformers accelerate
    python business-qa/run_hf.py --model meta-llama/Llama-3.2-1B-Instruct --out business-qa/results/llama-3.2-1b.jsonl
    python business-qa/score.py business-qa/results/llama-3.2-1b.jsonl

Llama 3.2 is gated on Hugging Face: accept its licence there and log in with `huggingface-cli login` first.
Greedy decoding, a few new tokens: the answer is one letter. `--dry-run` skips the model and answers "A"
to every question, to check the plumbing without a GPU. Resumes: questions already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompts import text_prompt
from score import load_questions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N questions only, for a quick test")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    questions = list(load_questions().values())[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(todo)} questions to run ({len(done)} already in {out})")

    if a.dry_run:
        def reply(prompt: str) -> str:
            return "A"
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        def reply(prompt: str) -> str:
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

    with open(out, "a") as f:
        for k, q in enumerate(todo, 1):
            f.write(json.dumps({"id": q["id"], "reply": reply(text_prompt(q))}) + "\n")
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}")


if __name__ == "__main__":
    main()
