# LTX self-attention: SageAttention 2.2.0 vs comfy-kitchen INT8 (2026-09-30)

`scripts/benchmark_ltx_denoise.py`: trial weights, INT8 distilled, 3 windows, median of 3, one compile cache, two interleaved rounds.
`5380b4a` SageAttention 2.2.0 (2++ fp8-PV kernel on 4090/5090, its Triton kernel on 3090), `e8abe77`/`4bec7d1` comfy-kitchen 0.2.36.
The 3090 ran heat-limited (int8 ~200 of ~240 TOPS).

| GPU | Canvas | SageAttention s/window | comfy-kitchen | Change |
|---|---|---|---|---|
| RTX 3090 | 512 | 14.09 | 13.86 | −1.7 % |
| RTX 3090 | 768 | 44.15 | 41.96 | −5.0 % |
| RTX 4090 | 512 | 5.21 | 5.18 | −0.4 % |
| RTX 4090 | 768 | 14.26 | 14.40 | +1.0 % |
| RTX 5090 | 512 | 3.92 | 3.96 | +1.0 % |
| RTX 5090 | 768 | 10.79 | 11.00 | +2.0 % |

Peak VRAM is the same. Against cuDNN SDPA (`2026-09-30_ltx_int8_kernels.md`), INT8 attention is 22 % (512) and 36 % (768) faster on the 4090.

Kernel alone, 32 heads × 18432 tokens × 128, time / relative L2 error vs fp32 / scratch VRAM:

| GPU | cuDNN | SageAttention | comfy-kitchen |
|---|---|---|---|
| RTX 3090 | 88.7 ms / 0.002 / 146 MiB | 52.7 ms / 0.013 / 432 MiB | 47.3 ms / 0.017 / 361 MiB |
| RTX 4090 | 33.9 ms / 0.002 / 146 MiB | 13.0 ms / 0.039 / 505 MiB | 13.4 ms / 0.017 / 361 MiB |
| RTX 5090 | 25.7 ms / 0.002 / 144 MiB | 10.3 ms / 0.013–0.025 / 504 MiB | 10.2 ms / 0.007–0.015 / 361 MiB |

5090 errors are on captured LTX q/k/v, the others on random inputs. Real clips (1080p and 2160p, restored-pixel PSNR vs an SDPA run)
land within the run-to-run floor for both kernels, and so do six full-frame synthetic-mosaic clips scored against ground truth.
comfy-kitchen's sparse `sol_attn` is 3× faster but reaches 0.07–0.34 error, so it is not used.
