#!/usr/bin/env bash
# Run the CiK finetuning on a single-GPU Linux machine with a terminal (Nuvolos VS Code app, Nebius VM, ...).
#
#   bash run_nuvolos.sh setup    # once: install everything, log in to Hugging Face, build CiK in TimeNet
#   bash run_nuvolos.sh smoke    # ~5 min: train on 8 examples, score 3 (checks everything works)
#   bash run_nuvolos.sh train    # ~20-30 min on a T4: the real run (step 9), then scores all 105 test instances
#   bash run_nuvolos.sh full     # big GPU only (Nebius): main run + 'no context' comparison, about 1 hour
#   bash run_nuvolos.sh extra    # round 2: ~10x more training examples from CiK's generator, train + score
#   bash run_nuvolos.sh pack     # zips the results so you can download them
#
# Everything goes into $WORK (default: a folder "cik_work" next to this script). Change it with
#   WORK=/some/other/folder bash run_nuvolos.sh setup
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$(dirname "$HERE")/cik_work}"
mkdir -p "$WORK"
export HF_HOME="$WORK/hf_cache"               # model downloads go here, not into a small home folder
export PIP_NO_CACHE_DIR=1                     # don't keep a second copy of every download (saves several GB)
export TMPDIR="$WORK/tmp"; mkdir -p "$TMPDIR"
VENV="$WORK/venv"
REPO="$WORK/OpenTSLM"
TIMENET="$WORK/TimeNet"
REG="$WORK/timenet_registry"

say() { printf '\n=== %s ===\n' "$*"; }

activate() {
  if [ -f "$WORK/conda_env" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$(cat "$WORK/conda_env")"
  elif [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  else
    export PATH="$HOME/.local/bin:$PATH"   # plain --user install
  fi
  cd "$REPO"
}

make_env() {
  # 1) a venv without pip, then pip from get-pip.py (works without sudo and without python3-venv)
  if [ ! -x "$VENV/bin/pip" ]; then
    rm -rf "$VENV"
    if python3 -m venv --system-site-packages --without-pip "$VENV" 2>/dev/null \
       && { "$VENV/bin/python" -m pip install -q --ignore-installed pip 2>/dev/null \
            || { curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py" \
                 && "$VENV/bin/python" "$TMPDIR/get-pip.py" -q; }; }; then
      echo "Using a Python venv in $VENV"
    else
      rm -rf "$VENV"
    fi
  fi
  if [ -x "$VENV/bin/pip" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"; return
  fi
  # 2) conda, if the machine has it
  if command -v conda >/dev/null; then
    echo "venv failed; using conda instead"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    [ -d "$WORK/conda" ] || conda create -y -q -p "$WORK/conda" python=3.11 pip
    conda activate "$WORK/conda"; echo "$WORK/conda" > "$WORK/conda_env"; return
  fi
  # 3) last resort: install into the user's own Python folder
  echo "venv and conda not available; installing with pip --user"
  export PIP_USER=1 PATH="$HOME/.local/bin:$PATH"
  python3 -m pip --version >/dev/null 2>&1 || { curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py" && python3 "$TMPDIR/get-pip.py" --user -q; }
  pip() { python3 -m pip "$@"; }
}

copy_code() {
  mkdir -p src/opentslm/time_series_datasets/cik
  touch src/opentslm/time_series_datasets/cik/__init__.py
  cp "$HERE"/cik_core.py "$HERE"/CiKQADataset.py "$HERE"/cik_rationales.jsonl src/opentslm/time_series_datasets/cik/
  cp "$HERE"/train_cik.py "$HERE"/eval_cik.py "$HERE"/make_rationales.py "$HERE"/VERSION \
     "$HERE"/cik_extra.py "$HERE"/build_rationales.py .
  echo "Code version: $(cat VERSION)"
}

case "${1:-}" in
setup)
  say "GPU check"
  nvidia-smi || { echo "No GPU found. Pick a GPU (T4) size for this Nuvolos app and restart it."; exit 1; }
  say "Disk space in $WORK (need about 10 GB free)"
  df -h "$WORK" | tail -1

  say "Python environment"
  make_env
  python3 -c "import torch; print('Already installed: torch', torch.__version__, 'CUDA OK' if torch.cuda.is_available() else 'no CUDA')" 2>/dev/null || true

  say "OpenTSLM"
  [ -d "$REPO" ] || git clone -q https://github.com/StanfordBDHG/OpenTSLM.git "$REPO"
  pip install -q -e "$REPO"

  say "TimeNet with your Context is Key connector (PR 788)"
  if [ ! -d "$TIMENET" ]; then
    git clone -q https://github.com/OpenTSLM/TimeNet.git "$TIMENET"
    (cd "$TIMENET" && git fetch -q origin pull/788/head:cik && git checkout -q cik)
  fi
  pip install -q -e "$TIMENET/packages/timenet" -e "$TIMENET/packages/timenet-connectors"
  pip uninstall -y -q torchao 2>/dev/null || true   # an old torchao breaks LoRA ("torchao >0.16 required")

  say "Our code"
  cd "$REPO" && copy_code

  say "Hugging Face login (paste a READ token from huggingface.co/settings/tokens)"
  if command -v hf >/dev/null; then hf auth login; else huggingface-cli login; fi

  say "Build CiK into TimeNet"
  timenet-build build servicenow/context-is-key --no-isolation
  python3 -c "import timenet; ds = timenet.TimeNet().load('servicenow/context-is-key@1.0.0'); print(len(ds.get_all()), 'forecasting tasks in TimeNet (expect 355)')"
  say "Setup done. Next: bash run_nuvolos.sh smoke"
  ;;

smoke)
  activate; copy_code
  python3 train_cik.py --out results/smoke --max_samples 8 --epochs 1 --batch_size 1
  python3 eval_cik.py --ckpt results/smoke/best_model.pt --out results/smoke --max_samples 3 --num_samples 1
  python3 - <<'EOF'
import json
for line in open("results/smoke/test_predictions.jsonl"):
    p = json.loads(line)
    print(p["task_name"], "| parsed ok:", p["parsed_ok"])
    print("  wrote:", " ".join(p["raw_output"].split())[:300])
EOF
  say "Smoke test done. If it printed 'Code version: ... v4' and the model wrote numbered lines, run: bash run_nuvolos.sh train"
  ;;

train)
  activate; copy_code
  python3 train_cik.py --out results/t4 --batch_size 1 --max_history 256
  python3 eval_cik.py --ckpt results/t4/best_model.pt --out results/t4 --max_history 256 --num_samples 3
  cat results/t4/metrics.json
  say "Training and scoring done. Next: bash run_nuvolos.sh pack"
  ;;

full)
  # For a big GPU (Nebius L40S/H100): the main run plus the "no context" comparison, with full settings.
  activate; copy_code
  python3 train_cik.py --out results/full --batch_size 1 --max_history 512
  python3 eval_cik.py --ckpt results/full/best_model.pt --out results/full --max_history 512 --num_samples 5
  python3 train_cik.py --out results/nocontext --batch_size 1 --max_history 512 --no_context
  python3 eval_cik.py --ckpt results/nocontext/best_model.pt --out results/nocontext --max_history 512 --num_samples 5 --no_context
  python3 - <<'EOF'
import json
rows = [("with context", "results/full"), ("no context", "results/nocontext")]
print(f"{'':14s} {'model mae':>10s} {'naive mae':>10s} {'model crps':>11s} {'naive crps':>11s} {'parsed':>7s}")
for name, d in rows:
    m = json.load(open(f"{d}/metrics.json"))
    print(f"{name:14s} {m['model']['mae']:10.4f} {m['seasonal_naive']['mae']:10.4f} "
          f"{m['model']['crps']:11.4f} {m['seasonal_naive']['crps']:11.4f} {m['parsed_ok_rate']:7.2f}")
EOF
  say "All done. Next: bash run_nuvolos.sh pack"
  ;;

extra)
  # Round 2: generate more training examples with CiK's own generator, then train and score once.
  # Test and validation stay the same CiK benchmark instances, so scores compare with results/full.
  activate; copy_code
  X="$WORK/extra"; mkdir -p "$X"
  say "1/5 Plan: training task types and test-data fingerprints"
  python3 cik_extra.py plan --out "$X/plan.json"
  say "2/5 Install CiK's generator (separate environment)"
  [ -d "$WORK/cik_repo" ] || git clone -q --depth 1 https://github.com/ServiceNow/context-is-key-forecasting.git "$WORK/cik_repo"
  [ -x "$WORK/cikvenv/bin/pip" ] || python3 -m venv "$WORK/cikvenv"
  "$WORK/cikvenv/bin/pip" install -q "gluonts>=0.14.4" statsmodels causalchamber==0.1.1 huggingface_hub h5py \
     termcolor tenacity pandas numpy pyarrow scipy matplotlib datasets networkx einops requests
  say "3/5 Generate extra training examples (about ${GEN_MINUTES:-25} minutes)"
  (cd "$X" && "$WORK/cikvenv/bin/python" "$REPO/cik_extra.py" generate --plan plan.json \
     --cik_repo "$WORK/cik_repo" --out rows.parquet --per_group "${PER_GROUP:-100}" --minutes "${GEN_MINUTES:-25}")
  python3 build_rationales.py "$X/rows.parquet" "$X/rationales.jsonl"
  say "4/5 Train on benchmark + extra examples"
  python3 train_cik.py --out results/extra --batch_size 1 --max_history 512 --epochs "${EPOCHS:-1}" --patience 1 \
     --extra_train "$X/rows.parquet" --extra_rationales "$X/rationales.jsonl"
  say "5/5 Score on the same 105 test instances"
  python3 eval_cik.py --ckpt results/extra/best_model.pt --out results/extra --max_history 512 --num_samples 3
  python3 - <<'EOF'
import json, os
rows = [("round 1 (full)", "results/full"), ("round 2 (extra data)", "results/extra")]
print(f"{'':22s} {'model mae':>10s} {'naive mae':>10s} {'model crps':>11s} {'naive crps':>11s} {'parsed':>7s}")
for name, d in rows:
    if os.path.exists(f"{d}/metrics.json"):
        m = json.load(open(f"{d}/metrics.json"))
        print(f"{name:22s} {m['model']['mae']:10.4f} {m['seasonal_naive']['mae']:10.4f} "
              f"{m['model']['crps']:11.4f} {m['seasonal_naive']['crps']:11.4f} {m['parsed_ok_rate']:7.2f}")
EOF
  say "All done. Next: bash run_nuvolos.sh pack"
  ;;

pack)
  cd "$REPO"
  zip -q -r "$WORK/results.zip" results -x '*.pt'
  [ -f "$WORK/extra/rows.parquet" ] && zip -q -j "$WORK/results.zip" "$WORK/extra/rationales.jsonl"
  for d in t4 full extra; do
    if [ -f "results/$d/best_model.pt" ]; then
      cp "results/$d/best_model.pt" "$WORK/best_model_$d.pt" && zip -q -j "$WORK/results.zip" "$WORK/best_model_$d.pt"
    fi
  done
  ls -lh "$WORK/results.zip"
  echo "Download $WORK/results.zip (Nuvolos: right-click it in VS Code; Nebius: scp, see NEBIUS.md)."
  ;;

*)
  sed -n '2,15p' "$0"
  exit 1
  ;;
esac
