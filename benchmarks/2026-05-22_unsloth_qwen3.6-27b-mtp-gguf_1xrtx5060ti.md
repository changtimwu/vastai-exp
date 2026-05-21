# unsloth/Qwen3.6-27B-MTP-GGUF (Q3_K_S) on 1× RTX 5060 Ti

Replicates the report in #2. Plan in #5.

| | |
|---|---|
| Date         | 2026-05-22 |
| Model        | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) |
| Quant        | Q3_K_S (12.6 GB) |
| GPU          | 1× RTX 5060 Ti 16 GB |
| vast.ai host | machine 71705, North Carolina (AMD 128c, 31 GB host RAM) |
| llama.cpp    | master @ `b1-bb28c1f` |
| Image        | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` |
| Instance     | 37253365 |
| $/hr         | $0.137 |

## Command

```bash
./bench.py https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF \
    --gpu rtx-5060ti --num-gpus 1 \
    --bin llama-cli --quant Q3_K_S \
    --params "-c 32768 --cache-type-k q8_0 --cache-type-v q8_0 \
              --spec-type draft-mtp --spec-draft-p-min 0.75"
```

## Result

```
[ Prompt: 111.1 t/s | Generation: 29.0 t/s ]
```

## Comparison to #2 (same model + quant + GPU + MTP)

| run | gen t/s | notes |
|---|---|---|
| #2 commenter, MTP, cold start | 35 | |
| **us, MTP** | **29.0** | `-n 256`, single seed |
| #2 commenter, MTP, steady-state | 41 | "stays there" |
| #2 commenter, no MTP, cold | 25 | |
| #2 commenter, no MTP, steady | 23 | |

Our 29 t/s sits below their cold-start MTP number. They described it warming up
to 41 t/s; our `-n 256` likely doesn't run long enough to see the steady state
(the spec-decode acceptance rate typically climbs once the cache fills). Bumping
to `-n 1024+` next time would clarify.

Even at 29 t/s, MTP wins vs the reporter's no-MTP 23 t/s steady-state —
about 25% uplift.

## Comparison across our own runs

| our run | model/quant | GPU | gen t/s |
|---|---|---|---|
| 2026-05-19 | Qwen3.6-27B Q4_K_M  | 2× 5060 Ti | 24.5 |
| **2026-05-22** | **Qwen3.6-27B Q3_K_S** | **1× 5060 Ti** | **29.0** |
| 2026-05-20 | Qwen3.6-35B-A3B Q6_K_XL | 1× 3060   | 17.4 |

One card at Q3 beats two cards at Q4 here. The 2-card May-19 run hit the
cudaMalloc OOM on compute-buffer alloc and recovered with a smaller buffer,
which likely shrunk the draft batch. At Q3 there's plenty of headroom on a
single 16 GB card.

## Notes

- Single seed (42), single prompt — same as previous runs.
- The `head -c 5M` cap in `remote_setup.sh` worked: bench.out came back at
  5 MB instead of last run's 2 GB. Timing trailer was in the first 100 KB.
- No `--draft-max` tuning — used llama.cpp defaults.

## Raw output

`results/37253365.out` (not in git).
