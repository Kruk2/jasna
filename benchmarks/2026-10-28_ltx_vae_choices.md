# LTX: full, half and max VAE encoders and decoders (2026-10-28)

RTX 5090, CLI from source (`main` 2fd40c3e, plaintext models), torch 2.12.1+cu130. Encoders: `full` (the VAE's
own), `half` / `max` (distilled with 2,3,2,1,1 / 1,2,1,1,1 of the 4,6,4,2,2 residual blocks per stage). Decoders:
`full` (the tuned decoder, 8 stage-5 blocks), `half` / `max` (distilled, 4 / 2 blocks).

## Per window

Seconds per 121-frame window, median of 5 after a warm-up (decode at the segment budget). Encoder latents vs the
full encoder's on two real clips: half 6.8-7.5 %, max 9.3-9.9 % relative L2.

| | 512 | 768 |
|---|---|---|
| Encode full / half / max | 0.37 / 0.20 / 0.14 | 0.85 / 0.46 / 0.31 |
| Decode full / half / max | 0.53 / 0.31 / 0.21 | 1.21 / 0.72 / 0.48 |

Encode + decode per 512 window across versions (earlier rows from `2026-10-19_ltx_vae_gpus.md`,
`2026-10-26_ltx_decoder_na9.md`, `2026-10-27_ltx_decoder_fused_kernels.md`; about +-5 % between benchmarks):

| | Encode | Decode | Both |
|---|---|---|---|
| 0.11.2 | 0.44 | 1.17 | 1.61 |
| 0.11.3 | 0.36 | 0.73 | 1.09 |
| 0.12.0 | 0.36 | ~0.77 | ~1.13 |
| full + full (9x9x9 window, fused kernels) | 0.37 | 0.53 | 0.90 |
| 0.12.1 (full + half) | 0.37 | 0.31 | 0.68 |
| half + max | 0.20 | 0.21 | 0.41 |
| max + max | 0.14 | 0.21 | 0.35 |

## Whole jobs

Four real 1080p30 videos (p894-1 30 s, i564-1 15 s, s485-1 45 s converted from 60 fps, the first
20 s of m488-2 scaled from 4K60), seconds per job including ~8 s of startup; each arm ran a 2 s segment
first so that compiling is not timed. Per-run values with encode / denoise / compose in the `.csv`.

| Video, model, canvas | full + full | half + half | max + max | half + max |
|---|---|---|---|---|
| p894, Basic, 512 | 21.9 | 18.7 | 17.5 | 18.1 (-17 %) |
| m488, Basic, 512 | 17.7 | 15.3 | 14.9 | 15.3 (-14 %) |
| i564, Basic, 512 | 18.2 | 12.9 | 12.1 | 13.0 (-29 %) |
| s485, Basic, 512 | 39.1 | 32.6 | 30.4 | 31.3 (-20 %) |
| p894, Basic, 768 | 40.7 | 32.2 | 28.8 | 30.3 (-26 %) |
| p894, Good, 512 | 35.6 | 31.6 | 31.1 | 31.1 (-13 %) |
| m488, Good, 512 | 26.2 | 24.0 | 23.1 | 23.8 (-9 %) |
| i564, Good, 512 | 22.7 | 20.9 | 19.3 | 19.6 (-14 %) |
| s485, Good, 512 | 64.2 | 58.7 | 58.7 | 59.1 (-8 %) |

m488 has no mosaic large enough for the 768 canvas (its 768 run equals the 512 one).

## Quality and defaults

Human A/Bs (full + full vs half + max, all eight video/model pairs above): full is "slightly sharper, not much"
or "very similar" every time; on p894 with Good, each step full -> half -> max is "a bit softer". Earlier A/Bs on
val and real clips: half + max close to the 0.12.1 VAE, which is ~1 % sharper on fine hair.

The VAE is a large share of a Basic or Standard job and a small one of a Good, High or Top job, so Basic and
Standard default to half + max and Good, High and Top to full + full. `--ltx-vae-encoder` /
`--ltx-vae-decoder` (`full`, `half`, `max`) override either (the `JASNA_LTX_VAE_*` variables until 0.12.2).

Not measured: other GPUs (the distilled parts run the same kernels at the same shapes, fewer of them), AMD, and
release builds. The job outputs are near-lossless HEVC (constant QP 0): Jasna has no lossless output.
