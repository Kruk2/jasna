# AMD Radeon AI PRO R9700: Linux release builds, v0.12.3 vs the 0.12.3 hotfix (2026-11-14)

Radeon AI PRO R9700 (gfx1201, 32 GB) on a rented instance: Ryzen 9 5950X (16 cores), 121 GB RAM, Ubuntu 26.04
(kernel 7.0), AMF 26.20 for decoding and HEVC encoding. Both versions are Linux AMD release archives, unpacked as
shipped: v0.12.3 is the published `v0.12.3` release, the hotfix is the release build of `9801a766` (v0.12.3 plus
the change that keeps AMD compose on the Torch path). Both bundle PyTorch 2.14.0+rocm10.1.0 (HIP 7.16). Standard
LTX (2 steps) with the full VAE encoder and decoder, on the real model weights. Each version kept its own cache
folder; both used the same settings folder, which also holds the MIOpen kernel database. Job time is the whole
job, measured around the process. Each version first ran one discarded M284 job; then M284 (15 s 1080p30, 449
frames, 5 windows) ran in the order A B B A and S610 (1080p, 6056 frames, 143-144 windows) in the order A B A B.
The tables show the mean of the two runs; per-run values are in the `.csv`.

| Job | v0.12.3 | hotfix |
|---|---|---|
| M284 | 83.7 s | 76.1 s (-7.6 s, -9 %) |
| S610 | 1012.5 s | 1015.8 s (+3.4 s, +0.3 %) |

## Rates per step

| LTX step | v0.12.3 M284 | hotfix M284 | v0.12.3 S610 | hotfix S610 |
|---|---|---|---|---|
| scan | 190.4 frames/s | 193.6 frames/s | 171.1 frames/s | 172.1 frames/s |
| encode | 1.45 s/window | 1.46 s/window | 1.51 s/window | 1.52 s/window |
| denoise | 1.56 s/window-step | 1.56 s/window-step | 1.76 s/window-step | 1.76 s/window-step |
| decode (compose) | 1.75 s/window | 1.81 s/window | 1.72 s/window | 1.76 s/window |
| transformer blocks resident | 48/48 | 48/48 | 48/48 | 48/48 |

Scan, encode and denoise are the same in both builds. Decode is slightly slower with the hotfix: +0.04 s/window
on S610 (both runs: 1.71 and 1.73 against 1.75 and 1.76), about 5 s over S610's 143 windows. On M284 one hotfix
run decoded at 1.87 s/window and the other at 1.75, the same as v0.12.3.

## Where the time goes

| LTX pass | v0.12.3 M284 | hotfix M284 | v0.12.3 S610 | hotfix S610 |
|---|---|---|---|---|
| scan | 3.8 s | 4.9 s | 36.8 s | 36.6 s |
| encode | 18.8 s | 17.9 s | 216.1 s | 217.8 s |
| denoise | 25.2 s | 25.2 s | 489.3 s | 489.8 s |
| compose | 13.4 s | 13.6 s | 255.6 s | 256.9 s |
| LTX total | 61.1 s | 61.4 s | 997.7 s | 1001.0 s |
| outside LTX (start, detector, setup, finishing the file) | 22.6 s | 14.7 s | 14.8 s | 14.9 s |

The LTX passes take the same time in both builds; on S610 the hotfix is 3.3 s slower in LTX, mostly compose. The
M284 difference comes from outside LTX: v0.12.3 spent 20.2 s and 24.9 s there, the hotfix 15.0 s and 14.3 s; on
S610 both builds spent about 15 s, so that part varies from run to run rather than with the build.

## Other notes

- The v0.12.2 vs v0.12.3 result in `2026-11-14_amd_linux_release_0.12.2_vs_0.12.3.md` (S610 1144 s to 1037 s,
  -9 %, on another R9700 instance) holds for the hotfix: on S610 the hotfix is within 0.3 % of v0.12.3 here.
- Peak VRAM is the same: LTX worker 15.6 GB (15994 MiB), whole card 15.2-15.3 GB on M284 and 15.8 GB on S610.
- Spread between the two runs: M284 5.1 s (v0.12.3) and 1.5 s (hotfix); S610 13.7 s (v0.12.3) and 4.6 s
  (hotfix). Both builds planned 143 or 144 S610 windows from run to run.
- Discarded first runs on M284: 129.2 s (v0.12.3, with the LTX speed test) and 77.2 s (hotfix, which had run on
  this instance before).
- No run lost the GPU.
- Command, the same for both versions:
  `jasna --input <clip> --output <out>.mkv --restoration-model-name ltx --ltx-vae-encoder full
  --ltx-vae-decoder full --online --log-level debug --no-progress`

Not measured: other AMD GPUs (the hotfix targets RX 7800/7700 XT class cards), Windows, Basic quality.
