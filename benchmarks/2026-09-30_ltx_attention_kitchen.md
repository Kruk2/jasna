# LTX self-attention: SageAttention 2++ vs comfy-kitchen INT8 (2026-09-30, RTX 5090)

`5380b4a` = SageAttention 2.2.0 (`sageattn`, the 2++ kernel: INT8 QK, FP8 PV, fp32+fp16 accumulation);
`e8abe77` = comfy-kitchen 0.2.36 `int8_attention`. Both with the fused INT8 linears.

Denoise (`scripts/benchmark_ltx_denoise.py`, trial weights, INT8 distilled, 3 windows, median of 3, one shared
compile cache, runs interleaved over two rounds):

| Canvas | SageAttention 2++ s/window | comfy-kitchen s/window | Change |
|---|---|---|---|
| 512 | 3.90 / 3.94 | 3.95 / 3.96 | +1 % |
| 768 | 10.76 / 10.81 | 10.97 / 11.03 | +2 % |

Kernel alone (1 × 32 heads × tokens × 128, bf16; same speed with the pipeline's transposed inputs):

| Tokens | cuDNN SDPA | SageAttention 2 (fp32 acc) | SageAttention 2++ | comfy-kitchen INT8 |
|---|---|---|---|---|
| 8192 | 5.04 ms | 2.21 ms | 2.08 ms | 2.02 ms |
| 18432 | 25.67 ms | 11.05 ms | 10.32 ms | 10.19 ms |
| cross, 8192 × 1024 | 0.68 ms | 0.35 ms | 0.32 ms | 0.41 ms |

Error against fp32 attention on captured LTX q/k/v (6 self-attention calls, canvas 512 and 768): relative L2
0.013–0.025 for SageAttention 2 and 2++, 0.007–0.015 for comfy-kitchen. Extra VRAM per call at 18432 tokens:
504 MiB (2++) vs 361 MiB (comfy-kitchen). comfy-kitchen's sparse `sol_attn` is 3× faster but has 0.07–0.34 error.

Real clips (restored-pixel PSNR against the cuDNN SDPA run; same-code rerun = noise floor):

| Clip | cuDNN rerun (floor) | SageAttention 2++ | comfy-kitchen | 2++ self + 2++ cross |
|---|---|---|---|---|
| 1080p | 37.59 | 37.97 | 38.00 | 37.70 |
| 2160p | 39.61 | 39.66 | 39.57 | 39.58 |

All variants are within run-to-run noise on real output.
