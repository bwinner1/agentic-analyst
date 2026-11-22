#!/usr/bin/env bash
# Forecast demo for the finetuned OpenTSLM model. Works on a Mac (Apple chip) and on a Linux GPU machine.
#
#   bash demo/demo.sh setup     # once, about 10 minutes: installs everything and downloads Llama 3.2 1B
#   bash demo/demo.sh run       # starts the demo; open http://127.0.0.1:7860 in your browser
#   bash demo/demo.sh replay    # starts the demo without the model (shows the saved test answers)
#
# Extra options after `run` go to app.py, e.g.  bash demo/demo.sh run --share   (temporary public link)
#                                               bash demo/demo.sh run --ckpt some/other.pt
#                                               bash demo/demo.sh run --device cpu
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/.work}"
mkdir -p "$WORK"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export GRADIO_ANALYTICS_ENABLED=False
VENV="$WORK/venv"
say() { printf '\n=== %s ===\n' "$*"; }

case "${1:-}" in
setup)
  say "uv (Python installer)"
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv --version

  say "Python 3.12 environment in $VENV"
  [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  say "OpenTSLM and the demo's packages (a few minutes)"
  [ -d "$WORK/OpenTSLM" ] || git clone -q --depth 1 https://github.com/StanfordBDHG/OpenTSLM.git "$WORK/OpenTSLM"
  uv pip install -q -e "$WORK/OpenTSLM" "gradio>=5,<7" pyarrow
  uv pip uninstall -q torchao 2>/dev/null || true   # an old torchao breaks LoRA
  python -c "import torch; print('torch', torch.__version__, '| Apple GPU (MPS):', torch.backends.mps.is_available(), '| NVIDIA GPU:', torch.cuda.is_available())"

  say "Hugging Face login (paste the same READ token you used for training)"
  if ! python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"; then
    if command -v hf >/dev/null; then hf auth login; else huggingface-cli login; fi
  fi

  say "Download Llama 3.2 1B (about 2.5 GB, once)"
  python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-3.2-1B", allow_patterns=["*.json", "*.safetensors"])
print("Llama 3.2 1B is ready")
EOF
  ls "$HERE"/checkpoints/*.pt >/dev/null 2>&1 || echo "Note: no checkpoint in $HERE/checkpoints yet. Copy best_model.pt there."
  say "Setup done. Next: bash demo/demo.sh run"
  ;;

run)
  shift
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python "$HERE/app.py" "$@"
  ;;

replay)
  shift
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python "$HERE/app.py" --replay "$HERE/replay/test_predictions_full.jsonl" "$@"
  ;;

*)
  sed -n '2,10p' "$0"
  ;;
esac
