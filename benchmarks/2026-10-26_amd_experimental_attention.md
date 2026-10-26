# AMD experimental attention flag comparison — 2026-10-26

The experimental-attention flag makes a substantial difference on this Windows V710 test. Toggling it has little effect on the Linux R9700 test. These are comparisons within each GPU and OS; the hardware and driver stacks differ between rows.

## Method

Frozen AMD 0.12.0, PyTorch 2.14.0+rocm10.1.0 and ROCm SDK 10.1.0. Linux uses the R9700/gfx1201 build from main `fec9a79f`, subsequently repackaged with an unchanged executable. Windows uses V710/gfx1101 and main `e8054db4`, with Microsoft Cloud driver 32.0.21017.2052. Executable identities are recorded in the CSV.

Both platforms use the identical bundled `test_clip1_1080p.mp4`: 300 frames, 1920×1080, 30 fps, SHA-256 `ba54003a7d197d8dbe3335d61ecaa88bb64426a3d7cebcf90a6df50140a2f0d1`. RF-DETR checkpoint hashes also match. Settings: FP16, detection batch 8, BasicVSR++, H.264 output, AMF hardware decode/encode, clip size 180, temporal overlap 15, no secondary restoration.

Each child process explicitly sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` to `0` or `1`. There is one discarded warm-up per setting and five measured runs per setting. Pair order alternates: 0/1, 1/0, 0/1, 1/0, 0/1. Each run starts a fresh application process. The Linux GPU lock is held for the sequence; no other Jasna workloads run in the Windows guest.

All 24 outputs pass complete software decoding and exact 300-frame 1080p counts. Warm-ups are retained in the CSV with their phase and excluded from medians. An initial Windows benchmark-controller attempt lost its process exit status; that attempt was preserved separately and excluded, then the complete sequence was rerun after successful exit-status self-checks.

## Medians of five measured runs

| Platform / GPU | Flag | Detection + tracking | Processing pipeline | Process wall | Logged peak VRAM |
|---|---:|---:|---:|---:|---:|
| Linux / R9700 | 0 | 3.2 s | 4.1 s | 13.25 s | 2926 MiB |
| Linux / R9700 | 1 | 3.2 s | 4.0 s | 13.14 s | 2812 MiB |
| Windows / V710 | 0 | 17.0 s | 17.3 s | 50.43 s | 5556 MiB |
| Windows / V710 | 1 | 5.4 s | 5.8 s | 38.95 s | 3126 MiB |

Detection/tracking and pipeline times are application stage logs, rounded to 0.1 seconds. Pipeline stages overlap; their durations must not be added together. Process wall includes startup and model loading. Peak VRAM is the application’s sampled value, not an instrumented hardware peak.

Windows flag 1 has a 3.15× faster median detection/tracking stage and 2.98× faster processing pipeline on this clip. Process wall falls 22.8%, from 50.43 to 38.95 seconds. Logged peak VRAM falls from 5,556 to 3,126 MiB.

Linux detection/tracking has the same 3.2-second median for both values. Pipeline medians differ by 0.1 seconds, with overlapping ranges; process wall differs by 0.11 seconds. A separate source-mode RF-DETR profiler probe using the same PyTorch stack and checkpoint selected `aten::_scaled_dot_product_flash_attention` with either flag value. This supplemental probe was separate from the frozen timing runs.

## Variation and limits

All measured samples, including outliers, are retained. Windows flag 0 detection/tracking spans 16.9–17.1 seconds. Flag 1 spans 5.4–22.7 seconds: its first measured run was slower, with a 24.5-second pipeline and 60.25-second process wall. The remaining four flag-1 detection/tracking runs were 5.4 seconds. The reported values are medians, not guarantees for every launch.

Linux detection/tracking ranges overlap: 3.2–3.4 seconds with flag 0 and 3.1–3.3 seconds with flag 1. The small Linux difference should not be presented as a meaningful performance gain.

This toggles the experimental eligibility flag and lets PyTorch choose its backend. It does not force MATH versus FLASH on both platforms. Windows kernel dispatch was not profiled. The workload includes tracking and restoration, so this is not an isolated attention-kernel benchmark. Floating-point changes can affect detected regions and downstream restoration work.

One short clip, two different GPUs, no LTX throughput comparison, no controlled visual-quality assessment, and no RX 9070 XT validation. User comparison copies include all warm-up and measured outputs plus the original input.
