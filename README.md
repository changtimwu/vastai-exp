# vastai-exp

Hands-off llama.cpp benchmarking on rented vast.ai GPUs. Pick a GPU, point at a
GGUF on Hugging Face, get tokens/sec back — no manual SSH, no leftover instances.

## Setup

```bash
. venv/bin/activate
vastai set api-key <KEY>
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"   # any local key works
```

## Usage

```bash
./bench.py <hf-url> --gpu <type> --num-gpus <N> [--bin BIN] [--params "..."]
```

Example — Qwen3.6-27B with MTP speculative decoding on 2x RTX 5060 Ti:

```bash
./bench.py https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF \
    --gpu rtx-5060ti --num-gpus 2 \
    --bin llama-cli \
    --params "--spec-draft-p-min 0.75 --spec-type draft-mtp" \
    --max-hourly 0.4 --yes
```

What it does, end to end:

1. Searches vast.ai offers for the requested GPU/count, ordered by DLP/$.
2. Rents the top one (capped by `--max-hourly`).
3. SSHes in once the instance is running.
4. Builds latest llama.cpp from master with CUDA on the box.
5. `hf_hub_download`s a GGUF matching `--quant` (default `Q4_K_M`).
6. Runs `<bin> -m model -ngl 999 --tensor-split 1,1,... <params>`.
7. Pulls back `results/<id>.log` and `results/<id>.out`.
8. Destroys the instance — even on failure (in `finally`).

## Flags

| flag | default | notes |
|------|---------|-------|
| `--gpu` | required | e.g. `rtx-5060ti`, `rtx-4090`, `h100` (see `GPU_ALIASES` in `bench.py`) |
| `--num-gpus` | `1` | |
| `--bin` | `llama-bench` | use `llama-cli` for speculation flags |
| `--params` | `""` | passed verbatim to the binary |
| `--quant` | `Q4_K_M` | substring match against GGUF filenames |
| `--disk` | `120` GB | bump for Q6+ on a 30B+ model |
| `--image` | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` | Blackwell-capable |
| `--max-hourly` | unset | reject offers above this $/hr |
| `--yes` | off | skip the rent-confirm prompt |
| `--keep` | off | leave the instance running after the run |

## Layout

```
bench.py          orchestrator: search → rent → run → tear down
remote_setup.sh   runs on the rented box: build llama.cpp, fetch GGUF, bench
venv/             vastai CLI + its deps
results/          per-instance .log and .out from each run
```
