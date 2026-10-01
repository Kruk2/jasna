# LTX decode: comfy-kitchen na3d + compiled stage 5 (2026-10-01)

`scripts/benchmark_ltx_decode.py`, RTX 5090, real weights, one 121-frame window, median of 3.
Old path `1da6e40` (variants patched in by the script), change `3ae528f`.

| Canvas, tile budget | old (NATTEN) | kitchen | compiled | both (3ae528f) |
|---|---|---|---|---|
| 512, 24 GiB | 2.56 s | 2.36 s | 2.06 s | 1.79 s (−30 %) |
| 768, 24 GiB | 5.43 s | 4.89 s | – | 3.71 s (−32 %) |
| 512, 6 GiB segment | 3.43 s | – | – | 2.39 s (−30 %) |

63 dB against the old path, VRAM +0.2 GiB. At 512 the old decode was 52 % neighborhood attention, 29 % the
rest of the attention slab, 9 % MLP. `max-autotune` gained only 1.4–2.9 % and is not used.
