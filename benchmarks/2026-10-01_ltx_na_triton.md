# LTX DiffVAE decode: Triton neighborhood attention (2026-10-01)

RTX 5090, real weights. Kernel: `scripts/benchmark_na3d.py` (`triton.testing.do_bench`, random Q/K/V on the
decoder's shapes, stage-5 queries on the slab's core columns only, as the decoder calls it). Decode:
`scripts/benchmark_ltx_decode.py --variants kitchen,shipped,fp8 --repeats 5`, one 121-frame window from a
random latent, median of 5. All rows at `85d690b` except the W-chunk rows (`329c6c9`). CSV: `2026-10-01_ltx_na_triton.csv`.
The INT8/FP8 experiment (`85d690b`, opt-in `JASNA_NA_FP8=1`, `fp8` benchmark variant) was not merged: its decode gain
was 1–2 % and its pixels fell slightly below the run-to-run floor.

## Kernel (ms per call)

| Shape (K/V, queries) | comfy-kitchen | NATTEN | Triton bf16 | Triton INT8 Q.K / FP8 P.V |
|---|---|---|---|---|
| stage 5, 512, W slab 42 (core 32) | 35.7 | 42.6 | 7.23 (93 TFLOPS) | 5.93 |
| stage 5, 512, W 138 (core 128) | 110.6 | 144.2 | 28.8 | 23.9 |
| stage 5, 768, W slab 58 (core 48) | 72.2 | 89.8 | 16.3 | 13.7 |
| stage 5, 768, W 202 (core 192) | 248.4 | 319.8 | 66.9 | 56.8 |
| stage 4, 512 (3x5x5) | 3.50 | 5.66 | 1.14 | 1.12 |
| stage 4, 768 (3x5x5) | 7.96 | 12.82 | 2.69 | 2.62 |

TFLOPS count useful work only. The kernel computes about twice that (each 4x4x4 query tile walks the union of
its windows, 14³ keys for 11³ needed), ~185 TFLOPS against ~230 for dense bf16 flash attention on this card.
Fixed configs: `4x4x4 n32 w2 s3` for 11³ windows, `2x4x4 n32 w2 s2` for 3x5x5 / 3x7x7 (sweep of ~280
tile/BLOCK_N/warps/stages combinations). Max |diff| vs kitchen 0.002 on stage 5 (kitchen vs NATTEN: 0.002–0.004).
Tried and dropped: masking variants (skipping the whole mask is only −10 %), splitting each tile's queries
into two T halves that each walk their own KV slices (slower: two M=32 dots), larger tiles.

## Decode (s per window)

| Canvas, tile budget | comfy-kitchen | Triton (shipped) | INT8/FP8 experiment |
|---|---|---|---|
| 512, 24 GiB (one tile) | 1.805 | 0.862 (−52 %) | 0.846 |
| 768, 24 GiB (one tile) | 3.800 | 1.915 (−50 %) | 1.876 |
| 512, 6 GiB (segment) | 2.382 | 1.121 (−53 %) | 1.122 |
| 768, 8 GiB (segment) | 6.297 | 2.831 (−55 %) | 2.771 |

Peak VRAM +0.1–0.4 GiB. Pixels vs kitchen: 63.4 dB (same seed and tiles), the same as kitchen vs the old
NATTEN path; the INT8/FP8 experiment 61.4–61.5 dB. Its kernel is ~18 % faster but quantizing K/V in the
compiled slab costs nearly all of that back: 1–2 % per decode.

## Stage-5 W chunks (not changed)

`DIFFUSION_W_CHUNKS` 4 → 2 → 1, Triton path (2 and 1: median of 3 at `329c6c9`, chunks patched by a wrapper; 4: the decode table above):

| Canvas, tile budget | 4 (shipped) | 2 | 1 |
|---|---|---|---|
| 512, 6 GiB | 1.121 s, 4.3 GiB | 1.068 s, 5.3 GiB | 1.008 s, 6.9 GiB (over budget) |
| 512, 24 GiB | 0.862 s, 5.3 GiB | 0.826 s, 6.8 GiB | 0.781 s, 9.1 GiB |
| 768, 8 GiB | 2.831 s, 6.2 GiB | 2.669 s, 7.2 GiB | 2.505 s, 9.2 GiB (over budget) |

With core-only queries the NA cost no longer depends on the chunk count; 2 chunks save ~5 % (less halo
QKV/RoPE work) but the tile-budget model does not count the larger slab, so it stays at 4.
