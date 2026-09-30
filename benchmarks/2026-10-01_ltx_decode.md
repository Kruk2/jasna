# LTX DiffVAE decode: comfy-kitchen na3d + compiled stage 5 (2026-10-01)

`scripts/benchmark_ltx_decode.py`: RTX 5090, real weights, one 121-frame window from a random latent, median of 3.
`1da6e40+patch` = the old path with the variant patched in by the script, `3ae528f` = the shipped change
(`3ae528f-draft` = the same with a graph break in the compiled slab, only used for the autotune rows).

| Canvas, tile budget | Old path (NATTEN, eager) | comfy-kitchen | compiled | both (shipped) |
|---|---|---|---|---|
| 512, 24 GiB (one tile) | 2.56 s | 2.36 s | 2.06 s | 1.79 s (−30 %) |
| 768, 24 GiB (one tile) | 5.43 s | 4.89 s | – | 3.71 s (−32 %) |
| 512, 6 GiB (segment, 2 tiles) | 3.43 s | – | – | 2.39 s (−30 %) |

Distilled INT8 denoise is ~3.9 s (512) and ~10.8 s (768) per window, so the decode was 50–85 % of it.
Pixels: 63 dB PSNR against the old path (same seed and tiles); the compile alone 66 dB. Peak VRAM +0.2 GiB.

Where the old 2.6 s went at 512: neighborhood attention 52 %, the rest of the attention slab
(QKV, fp32 RoPE, norms, halo copies) 29 %, MLP 9 %, stage-4 context upsample + projection 5 %, stages 1–4 5 %.
Every NA kernel reaches only 14–23 effective TFLOPS on this shape (11³ window, head_dim 64): kitchen 17–23,
NATTEN 15–17 with every cutlass tile config at or below its default; NATTEN flex-fna fails.

`max-autotune-no-cudagraphs` instead of the default mode: −1.4 % (512) and −2.9 % (768) for ~10 s extra
first compile per tile shape; not used. A later session ran ~6 % slower on the same code (old path 2.71 s),
so compare rows only within one session (label).
