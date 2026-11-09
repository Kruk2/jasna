# MIOpen tuning on Windows: Radeon PRO V710 (gfx1101), 2026-11-09

The Windows ROCm 10.1 wheels ship no MIOpen system databases, and MIOpen on Windows looks for them beside
MIOpen.dll (`_rocm_sdk_libraries\bin\gfx1101_27.HIP.fdb.txt` here). So every Windows run searched the LTX
encoder's convs on first use and kept untuned kernels; the Windows LTX worker also keeps its MIOpen cache
for one session only. `31e7fcec` ships tuned encoder solvers for gfx1101 (made on this GPU with
`scripts/tune_miopen.py --backends prebuilt`), writes them under both MIOpen file names, and points
`MIOPEN_SYSTEM_DB_PATH` at `share/miopen/db` on Windows.

## LTX VAE encoder, release (prebuilt) path

Full encoder, production MIOpen settings, empty user cache; first window, then the median of four more:

| canvas | no system database | Jasna tuning |
|---|---|---|
| 512 | 119.2 s, then 3.457 s/window | 10.7 s, then 2.736 s/window (-21 %) |
| 768 | 254.3 s, then 8.468 s/window | 13.8 s, then 6.210 s/window (-27 %) |

## Decoder neighborhood-attention tiles on RDNA3

The 2x2x8 stage-5 tiles that made an R9700 decode 10 % faster are neutral here: full-decoder decode at 512
3.425 / 3.497 s per window with the old 4x2x4 kernel, 3.455 / 3.518 s with the new one (alternating runs,
within the 2 % run-to-run spread). RDNA3 keeps the shared kernel.

## Setup

Azure Standard_NV24ads_V710_v5 (one Radeon PRO V710 MxGPU, which reports 27 compute units), Windows 11
Enterprise 24H2, Microsoft's cloud driver 32.0.21017.2052. Python 3.12, torch 2.14.0+rocm10.1.0 with the
gfx1101 device pack, source tree, run as a scheduled task (no Visual Studio, so no Triton path).
Placeholder weights: tuning and timing depend on shapes only.
