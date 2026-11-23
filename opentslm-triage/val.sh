#!/usr/bin/env bash
# Answer the validation split with a trained OpenTSLM checkpoint, to choose between variants
# without touching the test set. Usage: bash opentslm-triage/val.sh <run name under $OUT>
# Writes $OUT/val/<run name>/predictions.jsonl. Same environment as run.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/cik_work/venv}"
OPENTSLM_REPO="${OPENTSLM_REPO:-$HOME/cik_work/OpenTSLM}"
export HF_HOME="${HF_HOME:-$HOME/cik_work/hf_cache}"
DATA="${DATA:-$HOME/r2_work/data}"
OUT="${OUT:-$HOME/r2_work/results}"
RUN="${1:?usage: val.sh <run name under $OUT, e.g. train>}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$(dirname "$HERE")/src:$OPENTSLM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$HOME/r2_work/tmp" XDG_CACHE_HOME="$HOME/r2_work/cache" TORCH_HOME="$HOME/r2_work/torch"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$OUT/val/$RUN"
cd "$HOME/r2_work"
python "$HERE/eval_triage.py" --data_dir "$DATA" --split val --ckpt "$OUT/$RUN/best_model.pt" --out "$OUT/val/$RUN"
