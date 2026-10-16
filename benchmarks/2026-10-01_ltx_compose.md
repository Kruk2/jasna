# LTX compose: feather once per frame, histogram median (2026-10-01)

RTX 5090, `compose.composite_frame` on three 1080p 124-frame clips, replayed per frame (median of 5);
last column timed inside a full distilled INT8 run. Baseline `559968c`, change `80233a1`.

| Clip | baseline ms/frame | feather reuse only | 80233a1 | in a full run, per clip |
|---|---|---|---|---|
| c0 | 2.45 | 1.43 | 1.21 (−51 %) | 0.54 → 0.28 s |
| c1 | 5.85 | 3.08 | 2.96 (−49 %) | 0.89 → 0.45 s |
| c2 | 2.56 | 1.49 | 1.27 (−50 %) | 0.52 → 0.28 s |

Bit-identical output. Compose is 2–3 % of a run, so the whole run gains 1–1.6 %. Feathering was most of the
baseline cost (67–77 %); the colour-match median 7–11 % (`torch.histc`, no host sync, 0.08 ms per call).
