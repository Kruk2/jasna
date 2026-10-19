# LTX decode: DiffVAE decoder internals (2026-10-19)

RTX 5090, real weights, one 121-frame window, median of 5. Old = `c3dcae6`'s `transformer.py` loaded
side by side in the same process, interleaved with the change (uncommitted working tree on `c3dcae6`).

| Canvas, tile budget | old | new | change | peak VRAM |
|---|---|---|---|---|
| 512, 24 GiB (one tile) | 0.921 s | 0.744 s | −19 % | 5.15 → 4.64 GiB |
| 768, 24 GiB (one tile) | 2.001 s | 1.647 s | −18 % | 10.22 → 9.18 GiB |
| 512, 6 GiB (88-frame tiles) | 1.172 s | 0.943 s | −20 % | 4.09 → 3.72 GiB |
| 768, 8 GiB (480 px tiles) | 2.975 s | 2.375 s | −20 % | 5.76 → 5.30 GiB |
| 512, 8 GiB segment | 0.904 s | 0.730 s | −19 % | 5.15 → 4.64 GiB |
| 768, 16 GiB segment | 1.987 s | 1.638 s | −18 % | 10.22 → 9.18 GiB |

66.6 dB against the old path. First window with inductor caches off: 512 5.7 → 5.4 s, 768 7.4 → 8.7 s.

Before (512, one tile, 895 ms of GPU kernels): NA 27 %, context inject 14 % (two eager GEMMs, one of
them a 32x32 WMMA kernel on a strided W chunk), MLP 22 %, RoPE + its concat 8 %, halo slab copies 4 %,
stages 1-4 13 %. Steps at 512, one tile: context folded into one GEMM (`context_proj ∘ upsample.proj`)
−8 %, compiled with its shuffle + add −1.5 %; RoPE from per-axis cos/sin tables in one pass −5 %;
gate|up in one GEMM −1.5 %; halo slab gathered inside the compiled slab −3 %.
After (722 ms): NA 38 %, MLP 25 %, attention GEMMs 7 %, stages 1-4 17 %, context 5 %.

Dropped: compiling the stage-1-4 attention (−5 % per window, but +9 s first window), 2 W chunks
(−2 %, +1.5 GiB), 3 W chunks (no gain), fused QKV / K|V (V needs a strided copy for NA, slower),
bigger MLP token chunks (slower, L2 residency).
