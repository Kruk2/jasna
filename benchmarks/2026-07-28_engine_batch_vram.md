# Fixed-batch preprocess/upsample sub-engines; FP8 upsample removed (2026-07-28)

`scripts/benchmark_decode_backends.py`, source tree v0.9.1+, RTX 5090, driver 595.84, vali,
`--secondary-restoration none --temporal-overlap 15 --max-clip-size 180`. VRAM is per process, MiB.

TensorRT reserves scratch for the largest profile batch. The preprocess and upsample engines were built at the clip
size, so a 180-frame clip reserved 180 frames of scratch for per-frame work. They now use fixed batches
(`BASICVSRPP_PREPROCESS_BATCH = 60`, `BASICVSRPP_UPSAMPLE_BATCH = 30`): the engine set no longer depends on the clip
size, resident engine memory drops from 4560 to 916 MiB, and the upsample engine file from 452 to 78 MB.

E2E, H.264, median of 3 (`2026-07-28_engine_batch_vram.csv`; v0.9.1 rows from `2026-07-27_v0.9.1_e2e_cudagraphs_graphs_fp8.csv`):

| Clip | Wall v0.9.1 → new | VRAM med v0.9.1 → new | VRAM peak v0.9.1 → new |
|---|---:|---:|---:|
| 720p | 31.6 → 31.9 s | 7890 → **4312** (−45 %) | 7968 → **4720** (−41 %) |
| 1080p | 33.6 → 33.8 s | 8393 → **4867** (−42 %) | 9293 → **5779** (−38 %) |
| 2160p | 62.9 → 63.3 s | 10636 → **7326** (−31 %) | 13620 → **11176** (−18 %) |

Throughput 190.0 / 179.4 / 95.7 fps, within noise of v0.9.1. Upsample output is bit-exact; preprocess features are
bit-exact, SPyNet flows differ by ≤0.07 px (51.9 dB e2e), which is TensorRT tactic choice per profile, not batching.

## Supporting measurements (1080p h264 unless noted)

| What | Result |
|---|---|
| Upsample engine, 180 frames of work at batch 180 / 90 / 60 / 30 / 16 | 4672 / 2828 / 2106 / 1400 / 1118 MiB resident; 15.03 / 15.34 / 14.78 / 14.69 / 14.85 ms |
| v0.9.1 as shipped, b180: fp16 / fp16+graphs / fp8 / fp8+graphs (shipped) | 37.9 / 34.5 / 35.4 / 34.0 s; VRAM med 9979 / 9517 / 11809 / 11449, peak 10429 / 10417 / 12259 / 12035 |
| Upsample b30: fp16 / modelopt fp8 / shipped QDQ-ONNX fp8 | 748 / 560 / ~960 MiB; 15.06–15.11 / 13.56–13.94 / 15.36 ms per 180 frames; 74.42 / 64.21 / – dB |
| Upsample: fp16 b60 / fp16 b180 / modelopt fp8 b60 / QDQ-ONNX fp8 b180 | 1392 / 4422 / 1018 / ~5700 MiB; fp16 15.27 / 15.1 ms, fp8 b60 13.87–14.01 ms |
| E2E fp16 b30 vs fp8 b30 (two runs) | 33.97 / 33.77 vs 34.01 / 33.83 s; VRAM med 4919 / 5039 vs 4789 / 4705, peak 5689 / 5785 vs 5579 / 5583 |
| Preprocess batch 180 vs 60 (two runs) | 32.0 / 32.2 vs 32.9 / 32.5 s; VRAM med 5808 / 5685 vs 5077 / 4905, peak 6565 / 6533 vs 5789 / 5759 |
| Engine set loop_body ×4 / preprocess / upsample, b180 → fixed | 184 / 1292 / 3084 → 184 / 422 / 310 MiB |
| One 180-frame forward: propagate / upsample / preprocess | ~193 (88 %) / 14.5 (6.6 %) / 11.0 (5.0 %) of 218.5 ms |
| loop_body engines fp16 / fp8 / int8, graphs on (off) | 218.5 (255.1) / 414.4 (454.3) / 420.0 ms; 3214 / 3234 / 3234 MiB |
| `--no-compile-basicvsrpp`, clip 180 | 106.8 s / 4605 MiB vs 33.2 s / 4977 MiB compiled |

Clip-size sweep, overlap 8 (`2026-07-28_clip_size_sweep.csv`): 30 / 60 / 90 / 180 / 360 / 720 frames →
51.4 / 40.2 / 34.4 / 35.0 / 32.0 / 33.0 s, VRAM med 3411 / 3743 / 4095 / 5047 / 6307 / 8159, peak 3487 / 3907 / 4551 /
5591 / 7667 / 10957. Small clips are slower (overlap is reprocessed); past ~180 only VRAM grows.

FP8 upsample was the v0.9.1 VRAM regression (+1.8 GB): the shippable QDQ-ONNX route compiles to one Myelin node
with ~5.7 GB scratch at b180 (a `STRONGLY_TYPED` build: 945 MiB, 18.66 ms at b30). The better modelopt FP8 engine saves only ~230 MiB with no wall change and −10 dB,
and would need modelopt on every user machine, so FP8 is removed. Quantizing loop_body is ~1.9× slower.
The GUI clip-size slider now goes to 720. Not yet re-run on this code: 8K VR (3 repeats, ~2 min) and HUBLK-063 (~16 min; the earlier 15:40.8 / 11789 MiB
predates this change and is not a v0.9.1 number).

The release suite was cut to one H.264 input per resolution plus 8K VR (`docs/en/development.md`): codec variants
never disagreed in sign with H.264 across five releases (one magnitude outlier: v0.6.2 → v0.7.2 at 2160p, H.264
−0.3 %, AV1 +0.1 %, HEVC 10-bit −5.6 %), but they carry the decode signal (2160p: H.264 80.7 / 80.7 / 84.4 s vali /
pyav-hw / pyav-sw, AV1 and HEVC 10-bit spread 19 %) and 2.5–10.8 % VRAM spread at 4K; ~8 min instead of ~15.
