# LTX self-attention on RTX 3090 / 4090: SageAttention 2.2.0 vs comfy-kitchen INT8 (2026-09-30)

`5380b4a` = SageAttention 2.2.0 (GitHub d1a57a5, built per box; `sageattn` runs its Triton INT8/fp16-PV kernel on
SM 8.6 and the 2++ CUDA fp8-PV fp32+fp16 kernel on SM 8.9); `4bec7d1` = comfy-kitchen 0.2.36 `int8_attention`.
`scripts/benchmark_ltx_denoise.py`, trial weights, INT8 distilled, 3 windows, median of 3, one shared compile cache,
interleaved over two rounds. Rented vast.ai boxes; the 3090 ran thermally limited (0x20, int8 200 of ~240 TOPS),
the 4090 at its 450 W cap.

| GPU | Canvas | SageAttention s/window | comfy-kitchen s/window | Change |
|---|---|---|---|---|
| RTX 3090 | 512 | 13.97 / 14.21 | 13.69 / 14.02 | −1.7 % |
| RTX 3090 | 768 | 44.13 / 44.17 | 42.34 / 41.58 | −5.0 % |
| RTX 4090 | 512 | 5.20 / 5.21 | 5.18 / 5.18 | −0.4 % |
| RTX 4090 | 768 | 14.25 / 14.26 | 14.40 / 14.40 | +1.0 % |

Peak VRAM is identical per canvas (12.75 / 13.57 GiB). Against cuDNN SDPA (no INT8 attention installed,
`benchmarks/2026-09-30_ltx_per_arch.md`, other boxes of the same GPU) comfy-kitchen is 13 % (3090, 512) to 36 % (4090, 768) faster.

Kernel alone (1 × 32 heads × tokens × 128, bf16), time / relative L2 error vs fp32 / extra VRAM:

| GPU | Tokens | cuDNN | SageAttention | comfy-kitchen |
|---|---|---|---|---|
| RTX 3090 | 8192 | 15.3 ms / 0.002 / 65 MiB | 9.1 ms / 0.013 / 192 MiB | 9.3 ms / 0.016 / 160 MiB |
| RTX 3090 | 18432 | 88.7 ms / 0.002 / 146 MiB | 52.7 ms / 0.013 / 432 MiB | 47.3 ms / 0.017 / 361 MiB |
| RTX 4090 | 8192 | 6.8 ms / 0.002 / 65 MiB | 2.9 ms / 0.039 / 224 MiB | 2.8 ms / 0.016 / 160 MiB |
| RTX 4090 | 18432 | 33.9 ms / 0.002 / 146 MiB | 13.0 ms / 0.039 / 505 MiB | 13.4 ms / 0.017 / 361 MiB |
