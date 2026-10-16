# BasicVSR++ sub-engine quantization: INT8 / FP8 / FP4 (2026-07-27)

RTX 5090 (SM 12.0), driver 595.84, TRT 10.16.1.11, torch-tensorrt 2.12.1, nvidia-modelopt 0.45.0.
Harnesses `scripts/quantize_basicvsrpp.py` and `scripts/loop_body_fusion.py`, both deleted afterwards (in git history).
48 synthetic mosaic clips (T=30, 256²; 32 calibration, 16 held out), quality vs the fp32 PyTorch model;
`max_clip_size=60` engines at T=30 (b180 engines at T=180), median of 20+ iterations.

| Config | PSNR mean/min dB | SSIM | Full fwd T=30 | Full fwd T=180 | VRAM T=30 / T=180 MiB |
|---|---|---|---|---|---|
| fp16 (shipping) | 73.9 / 69.4 | 1.0000 | 46.2 ms | 269.6 ms | 2886 / 7626 |
| fp16 + CUDA graphs | bit-exact | | 36.3 ms (−21 %) | 206.9 ms (−23 %) | 3012 / 8336 |
| fp8 upsample only | 63.9 / 59.2 | 0.9998 | 48.2 ms (≈ baseline, noise ±3) | 245.8 ms (−9 %) | 2512 / 6454 |
| fp8 upsample + CUDA graphs | 63.9 / 59.2 | 0.9998 | 35.4 ms (−23 %) | 199.4 ms (−26 %) | 2638 / 7212 |
| fp8 upsample + preprocess (SPyNet excluded) | 42.5 / 38.1 | 0.9790 | ≈ baseline | | |
| int8 all engines | 44.3 / 39.8 | 0.9846 | 71.5 ms | | 2756 / – |
| fp8 all engines | 42.3 / 37.7 | 0.9784 | 77.2 ms | | 2350 / – |

Full forward is the split forward in one process, VRAM is NVML process memory after warmup. An earlier pass
without VRAM sampling measured fp16 44.3, int8 77.3, fp8 73.5 ms, and torch-TRT CUDA graphs mode 34.7 ms (−22 %).

Per stage, fp16 → int8 / fp8: preprocess 2.07 → 2.71 / 2.81 ms, loop_body 0.332 → 0.648 / 0.529 ms per call,
upsample 2.45 → 3.00 / 2.18 ms (fp8 upsample alone 2.23 ms). FP4/NVFP4 does not apply: the net is all Conv2d.

loop_body alternatives (per step; fp16 engine 0.33 ms, 810 MiB): decomposed deform conv 0.35 ms; K=10 unrolled
engine 0.31 ms (852 MiB); TRT ILoop via ONNX `Scan` 0.30–0.31 ms (790 MiB at K=29, 1158 MiB at K=179);
cross-clip batch 4 / 8 0.24–0.25 / 0.23 ms per clip (878 MiB). Mixed clip lengths with graphs: 62.1 → 51.0 ms/clip.

INT8 and FP8 lose everywhere except FP8 upsample: TensorRT keeps the small batch-1 loop_body convs in fp16 and runs
the Q/DQ pairs as extra pointwise layers, and quantizing preprocess wrecks quality (its features feed the whole
recurrence). The forward is ~90 % sequential loop_body calls at a ~0.30 ms floor, so the levers are CUDA graphs
(shipped in v0.9.1) and cross-clip batching (needs a pipeline restructure). The prebaked FP8 upsample ONNX
(modelopt at dev time, SM 8.9+) shipped in v0.9.1 and was removed in `2026-07-28_engine_batch_vram.md`.
