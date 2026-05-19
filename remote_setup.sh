#!/usr/bin/env bash
# Runs on the rented vast.ai box. Reads bench config from /root/.bench_* files
# staged via onstart-cmd, then builds llama.cpp and runs the requested binary.
set -euo pipefail

REPO_ID=$(cat /root/.bench_repo)
QUANT=$(cat /root/.bench_quant)
BIN=$(cat /root/.bench_bin)
PARAMS=$(cat /root/.bench_params)

echo "=== bench config ==="
echo "REPO_ID=$REPO_ID"
echo "QUANT=$QUANT"
echo "BIN=$BIN"
echo "PARAMS=$PARAMS"
nvidia-smi -L || true
echo

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build ccache \
    python3 python3-pip libcurl4-openssl-dev ca-certificates curl

pip install -q --break-system-packages huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /root
if [ ! -d llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
git pull --depth 1 origin master || true
HEAD_SHA=$(git rev-parse --short HEAD)
echo "llama.cpp @ $HEAD_SHA"

cmake -B build -G Ninja -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build -j --target llama-bench llama-cli llama-server

# Pick a GGUF file matching the requested quant; fall back to first .gguf.
python3 - "$REPO_ID" "$QUANT" <<'PY'
import os, sys
from huggingface_hub import HfApi, hf_hub_download
repo, quant = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf")]
if not files:
    sys.exit(f"no GGUF files in {repo}")
match = [f for f in files if quant.lower() in f.lower()] or files
path = hf_hub_download(repo, match[0], local_dir="/root/models")
print(path)
with open("/root/.bench_model", "w") as f:
    f.write(path)
PY

MODEL=$(cat /root/.bench_model)
echo "Model file: $MODEL"
echo

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 1 ]; then
    echo "no GPUs visible inside container" >&2
    exit 1
fi
SPLIT=$(python3 -c "print(','.join(['1']*$NGPU))")
cd /root/llama.cpp

PROMPT='Explain in detail how speculative decoding works in a transformer-based language model, covering draft models, verification, acceptance criteria, and the speedups it enables. Use technical language and concrete examples.'

# For llama-cli we need a prompt + -n; llama-bench has its own modes.
case "$BIN" in
    llama-cli)
        EXTRA_DEFAULTS=(-p "$PROMPT" -n 256 --no-conversation --no-warmup -s 42)
        ;;
    *)
        EXTRA_DEFAULTS=()
        ;;
esac

# Word-split $PARAMS intentionally so user-supplied flags reach the binary.
# shellcheck disable=SC2086
./build/bin/"$BIN" \
    -m "$MODEL" \
    -ngl 999 \
    --tensor-split "$SPLIT" \
    "${EXTRA_DEFAULTS[@]}" \
    $PARAMS \
    2>&1 | tee /root/bench.out

echo
echo "=== done; head/$HEAD_SHA ==="
