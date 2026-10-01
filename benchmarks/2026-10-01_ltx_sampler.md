# LTX sampler: no host syncs, shared prompt K/V (2026-10-01)

`scripts/benchmark_ltx_denoise.py`, RTX 5090, trial weights, INT8, back to back in one session.
`5534c5e`: no host sync per step. `6d2d2b7`: plus prompt K/V computed once per block and step.

| Model, canvas | 559968c s | 5534c5e s | 6d2d2b7 s | change |
|---|---|---|---|---|
| distilled, 512 | 11.82 | 11.83 | 11.69 | −1.1 % |
| distilled, 768 | 32.61 | 32.58 | 32.49 | −0.3 % |
| undistilled, 512 | 31.32 | 31.22 | 31.02 | −1.0 % |

The sync removal alone is within noise. The K/V gain grows with the number of windows per track.
Eager output is bit-identical; compiled output differs between runs of the same commit (~0.4 % rel. L2).
