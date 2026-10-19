# LTX decode: DiffVAE decoder internals (2026-10-19)

RTX 5090, real weights, one 121-frame window, median of 5; old (`c3dcae6`) and new interleaved in one
process. Measured on the working tree that became `d8d3b98`.

| Canvas, tile budget | old | new | change | peak VRAM |
|---|---|---|---|---|
| 512, 6 GiB (old segment budget) | 1.172 s | 0.943 s | −20 % | 4.09 → 3.72 GiB |
| 768, 8 GiB (old segment budget) | 2.975 s | 2.375 s | −20 % | 5.76 → 5.30 GiB |
| 512, 8 GiB (one tile) | 0.904 s | 0.730 s | −19 % | 5.15 → 4.64 GiB |
| 768, 16 GiB (one tile) | 1.987 s | 1.638 s | −18 % | 10.22 → 9.18 GiB |

66.6 dB against the old path. Kept: `context_proj` folded into the stage-4 upsample (−9 %), RoPE in one
pass (−5 %), gate|up in one GEMM, halo slab gathered in the compiled slab (−3 %).
Dropped: compiling stages 1-4 (+9 s first window), fewer W chunks (+1.5 GiB), fused QKV (slower).
