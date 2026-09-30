# LTX compose: per-track feather reuse + histogram colour-match median (2026-10-01)

RTX 5090, `compose.composite_frame` on three 1080p, 124-frame test clips (one mosaic track each; 93 frames with
one live window, 31 with two overlapping windows). `offline` = the captured inputs of every frame replayed,
CUDA-synced, median of 5 passes; `in_pipeline` = the same function timed inside a full distilled-INT8 LTX run.
`80233a1-median` = 80233a1 with the old `torch.quantile` median swapped back in.

| Clip | Baseline 559968c | Feather reuse only | Shipped 80233a1 | In pipeline, whole clip |
|---|---|---|---|---|
| c0 | 2.45 ms/frame | 1.43 | 1.21 (−51 %) | 0.54 → 0.28 s |
| c1 | 5.85 ms/frame | 3.08 | 2.96 (−49 %) | 0.89 → 0.45 s |
| c2 | 2.56 ms/frame | 1.49 | 1.27 (−50 %) | 0.52 → 0.28 s |

Output is bit-identical to 559968c on all 372 frames (max abs diff 0). Whole run: compose was 2–3 % of a
~27 s clip; the saving is 0.25–0.44 s per clip (1–1.6 %), inside the run-to-run noise of a full render.

Where the baseline time went (c0 / c1): Gaussian-feathered ellipse dilation 67 % / 77 % (the union and every
candidate each feathered a mask; with one track they are the same mask), colour-match median 11 % / 7 %,
polygon rasterization 6 %, rest 15 %.

Colour-match median per call on the real rings (~210–250 k pixels): `torch.quantile` behind `ring.any()`
0.13 ms plus a host sync; a `scatter_add_` histogram over the whole region 0.64 ms on c1 (atomics pile onto the
zero-difference bin outside the crop), 0.14 ms over the crop box; `torch.histc` over the crop box 0.077 ms,
no host sync (shipped).
