# LTX VAE speed-ups per GPU (2026-10-19)

Old `c3dcae6`, new `d8d3b98`. Seconds per 121-frame window, decode at each version's segment budget.
`scripts/benchmark_ltx_encode.py` / `benchmark_ltx_decode.py`; trial weights on rented GPUs, real
weights on the 5090. RTX 3090 denoise is from `2026-10-01_ltx_gpus.md` (its hosts had no NVENC).

| GPU | Canvas | Encode | Decode | Denoise step | Window, 4 steps |
|---|---|---|---|---|---|
| RTX 5090 | 512 | 0.44 → 0.36 | 1.17 → 0.73 | 0.48 | 3.53 → 3.01 (−15 %) |
| RTX 5090 | 768 | 0.99 → 0.80 | 2.98 → 1.64 | 1.36 | 9.41 → 7.88 (−16 %) |
| RTX 4090 | 512 | 0.64 → 0.49 | 1.84 → 1.07 | 0.63 | 5.00 → 4.08 (−18 %) |
| RTX 4090 | 768 | 1.44 → 1.09 | 4.35 → 2.45 | 1.77 | 12.87 → 10.62 (−17 %) |
| RTX 5080 16 GB | 512 | 0.89 → 0.73 | 2.66 → 1.50 | 1.02 | 7.63 → 6.31 (−17 %) |
| RTX 5080 16 GB | 768 | 2.01 → 1.65 | 6.31 → 4.65 | 2.81 | 19.56 → 17.54 (−10 %) |
| RTX 3090 | 512 | 1.37 → 1.19 | 3.72 → 2.26 | 1.54 | 11.25 → 9.61 (−15 %) |
| RTX 3090 | 768 | 3.13 → 2.72 | 9.10 → 5.09 | 4.60 | 30.63 → 26.21 (−14 %) |
| RTX 4060 8 GB | 512 | 3.04 → 2.48 | 8.83 → 6.45 | 3.10 | 24.27 → 21.33 (−12 %) |

Window = encode + 4 denoise steps + decode; scan, compositing and video encoding are not included.
Whole trial jobs: −6 to −9 % (4090, 5080), −10 to −14 % (4060). No out-of-memory on any card; the
16 GB card keeps two 768 tiles (one tile left only ~0.5 GiB free during compose).
