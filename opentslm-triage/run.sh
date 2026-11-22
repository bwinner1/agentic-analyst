#!/usr/bin/env bash
# Reuse the environment created by opentslm-cik/run_nuvolos.sh setup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/cik_work/venv}"
OPENTSLM_REPO="${OPENTSLM_REPO:-$HOME/cik_work/OpenTSLM}"
export HF_HOME="${HF_HOME:-$HOME/cik_work/hf_cache}"
DATA="${DATA:-$HOME/r2_work/data}"
OUT="${OUT:-$HOME/r2_work/results}"

case "${1:-}" in
  smoke|untuned|train|text|text0) command="$1"; shift ;;
  *) echo "Usage: bash opentslm-triage/run.sh {smoke|untuned|train|text|text0} [script flags]"; exit 2 ;;
esac

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing $VENV/bin/activate; set VENV to the environment from CiK setup." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$(dirname "$HERE")/src:$OPENTSLM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$HOME/r2_work/tmp"
export XDG_CACHE_HOME="$HOME/r2_work/cache"
export TORCH_HOME="$HOME/r2_work/torch"
export TRITON_CACHE_DIR="$HOME/r2_work/triton"
export TORCHINDUCTOR_CACHE_DIR="$HOME/r2_work/torchinductor"
mkdir -p "$OUT" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
OUT="$(cd "$OUT" && pwd)"
DATA="$(cd "$DATA" && pwd)"
# Relative outputs from libraries also land here, never in the OpenTSLM checkout.
cd "$HOME/r2_work"
for split in train val test; do
  [[ -f "$DATA/$split.jsonl" ]] || { echo "Missing $DATA/$split.jsonl" >&2; exit 1; }
done

case "$command" in
  smoke)
    python "$HERE/train_triage.py" --data_dir "$DATA" --out "$OUT/smoke" \
      --epochs 1 --max_samples 16 --batch_size 1 "$@"
    python "$HERE/eval_triage.py" --data_dir "$DATA" --out "$OUT/smoke" \
      --ckpt "$OUT/smoke/best_model.pt" --max_samples 4
    python - "$DATA" "$OUT/smoke/predictions.jsonl" <<'PY'
import json
import sys
from kpi_triage.labels import parse_answer

with open(sys.argv[1] + "/test.jsonl", encoding="utf-8") as handle:
    rows = {row["id"]: row for line in handle if line.strip() for row in [json.loads(line)]}
with open(sys.argv[2], encoding="utf-8") as handle:
    for line in handle:
        prediction = json.loads(line)
        label, reason = parse_answer(prediction["text"], rows[prediction["id"]]["window"])
        print(prediction["id"], "parsed:", label.to_dict() if label else None, "reason:", reason)
        print(prediction["text"])
PY
    ;;
  untuned)
    python "$HERE/eval_triage.py" --data_dir "$DATA" --out "$OUT/untuned" --ckpt none "$@"
    ;;
  train)
    python "$HERE/train_triage.py" --data_dir "$DATA" --out "$OUT/train" "$@"
    python "$HERE/eval_triage.py" --data_dir "$DATA" --out "$OUT/train" --ckpt "$OUT/train/best_model.pt"
    ;;
  text)
    python "$HERE/text_baseline.py" --data_dir "$DATA" --out "$OUT/text" "$@"
    ;;
  text0)
    python "$HERE/text_baseline.py" --data_dir "$DATA" --out "$OUT/text0" --zero_shot "$@"
    ;;
esac
