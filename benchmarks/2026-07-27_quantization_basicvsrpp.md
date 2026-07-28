# BasicVSR++ sub-engine PTQ investigation (INT8 / FP8 / FP4)

Date: 2026-07-27. GPU: RTX 5090 (SM 12.0), TRT 10.16.1.11, torch-tensorrt 2.12.1,
nvidia-modelopt 0.45.0, driver 595.84. Harness: `scripts/quantize_basicvsrpp.py`
(removed once the investigation closed — `git log -- scripts/quantize_basicvsrpp.py`).
Calibration/eval: 48 synthetic mosaic clips (T=30, 256², from CAWD-166-1080p),
32 calib / 16 held-out. Quality vs fp32 PyTorch reference. `max_clip_size=60`
engines, timings at T=30, median of 20+ iters.

## Results

| config | PSNR mean/min (dB) | SSIM mean | preprocess | loop_body ×1 | upsample | full fwd T=30 |
|---|---|---|---|---|---|---|
| fp16 (prod baseline) | 73.9 / 69.4 | 1.0000 | 2.07 ms | 0.332 ms | 2.45 ms | 44.3 ms |
| int8 all engines | 44.3 / 39.8 | 0.9846 | 2.71 ms | 0.648 ms | 3.00 ms | 77.3 ms |
| fp8 all engines | 42.3 / 37.7 | 0.9784 | 2.81 ms | 0.529 ms | 2.18 ms | 73.5 ms |
| fp8 upsample only | 63.9 / 59.2 | 0.9998 | – | – | 2.23 ms | ≈ baseline |
| + fp8 preprocess (spynet excluded) | 42.5 / 38.1 | 0.9790 | 2.07 ms | – | – | ≈ baseline |

## Findings

- **INT8 is a net loss everywhere.** In the loop_body engines TensorRT's tactic
  search keeps the convolutions in fp16 and executes the Q/DQ pairs as fused
  pointwise arithmetic (`CastMulMinMaxRound…` Myelin layers) → 2× slower per
  call. In preprocess/upsample the convs do fuse to Int8, but the engines are
  memory-bound, so the extra Q/DQ bandwidth outweighs the int8 math: still
  ~20% slower than fp16.
- **FP8 fuses cleanly (per-tensor scales) and wins only in upsample**:
  2.45→2.23 ms (+10%). preprocess and loop_body get slower.
- **Quantizing preprocess destroys quality for zero gain**: feat_extract
  features feed the entire bidirectional recurrence, so PTQ error amplifies
  (42 dB even with spynet excluded). Never quantize preprocess.
- **fp8-upsample-only is the only quality-safe config** (63.9 dB / SSIM 0.9998)
  but upsample is ~5% of forward time → **<1% end-to-end**. Not worth a
  precision axis in the engine cache.
- **FP4/NVFP4 not applicable**: TRT FP4 targets GEMM/Linear; the net is 100%
  Conv2d.
- **Where the time actually is**: T=30 forward ≈ 120 sequential batch-1
  loop_body calls (~0.33 ms each ≈ 90% of forward). The engines are
  latency/launch-bound, not compute-bound — precision cannot help. The real
  levers are structural: CUDA-graphing the propagation loop, moving the
  per-frame Python loop into a single engine, or batching across independent
  clips.

## Structural follow-up (same date): loop_body sequential-call optimization

Tested with `scripts/loop_body_fusion.py` (also since removed; recover it from
git history) + scratch experiments:

| approach | full fwd T=30 | per loop_body step | notes |
|---|---|---|---|
| baseline fp16 engines | 44.3 ms | 0.33 ms | 4·(T−1)=116 sequential batch-1 calls |
| **torch-TRT CUDA graphs mode** | **34.7 ms (−22%)** | 0.32 ms | `torch_tensorrt.runtime.set_cudagraphs_mode(True)`, bit-exact (max|Δ|=0) |
| K-step unrolled engine (K=5/10) | – | 0.31 ms | needs deform_conv2d decomposed; no win |
| TRT ILoop (torch `scan` → ONNX `Scan` → OnnxParser) | – | 0.30 ms | builds fine (8 s, dynamic T); no win |
| cross-clip batching (batch=4 / 8) | – | 0.24 / 0.23 ms per clip | 1.4–1.5× on the dominant stage; needs pipeline restructure |

- Mixed clip lengths (alternating T=30/45): 62.1 → 51.0 ms/clip with CUDA graphs —
  dynamic-shape re-record costs ~12 ms vs constant-T but stays net-positive.
- ~0.30 ms/step is the hard floor: sequential dependent conv chain
  (~36 convs at 1×64×64×64) under-occupies the GPU; launch overhead is NOT the
  bottleneck, so loop-in-engine (unroll/ILoop) cannot help.
- `deform_conv2d` decomposes exactly into 9× grid_sample + mask + 1×1 conv
  (TRT/ONNX-native), which removes the
  per-call TRT→torch→TRT partition break and unlocks ONNX export of the whole
  loop (`torch._higher_order_ops.scan` → ONNX `Scan` → TRT ILoop) — kept as a
  reference even though it gives no speedup today.
- **Actionable:** enable torch-TRT CUDA graphs mode for the sub-engine path
  (−22% restoration forward, bit-exact); consider cross-clip batching later.

## Final tables (single process, NVML process VRAM after warmup, split full forward)

T=30 clip:

| config | full fwd | VRAM | quality vs fp32 ref |
|---|---|---|---|
| fp16 baseline | 46.2 ms | 2886 MiB | 73.9 dB / SSIM 1.0000 |
| fp16 + CUDA graphs | 36.3 ms (−21%) | 3012 MiB | bit-exact vs fp16 |
| **fp8 upsample + CUDA graphs** | **35.4 ms (−23%)** | **2638 MiB** | 63.9 dB / SSIM 0.9998 |
| fp8 upsample only | ≈ baseline (48.2 ms, run noise ±3) | 2512 MiB | 63.9 dB / 0.9998 |
| int8 all engines | 71.5 ms | 2756 MiB | 44.3 dB / 0.9846 |
| fp8 all engines | 77.2 ms | 2350 MiB | 42.3 dB / 0.9784 |

T=180 clip (b180 engines):

| config | full fwd | VRAM |
|---|---|---|
| fp16 baseline | 269.6 ms | 7626 MiB |
| fp16 + CUDA graphs | 206.9 ms (−23%) | 8336 MiB |
| fp8 upsample only | 245.8 ms (−9%) | 6454 MiB |
| **fp8 upsample + CUDA graphs** | **199.4 ms (−26%)** | **7212 MiB** |

Stage-level, loop_body backward_1 (process VRAM = engine only, no full pipeline):

| variant | per step | VRAM |
|---|---|---|
| prod fp16 engine ×1 call | 0.33 ms | 810 MiB |
| decomposed deform engine ×1 | 0.35 ms | 810 MiB |
| K=10 unrolled engine | 0.31 ms | 852 MiB |
| TRT ILoop (Scan) K=29 | 0.31 ms | 790 MiB |
| TRT ILoop (Scan) K=179 | 0.31 ms | 1158 MiB |
| batch=4 clips ×1 call | 0.25 ms/clip | 878 MiB |

CUDA graphs cost ~700 MiB extra pools at T=180 (+9%) but pairing with the fp8
upsample engine lands below the fp16 baseline VRAM while being fastest overall.
Min-VRAM option: fp8-upsample-only without graphs (6454 MiB, −9% time).

## Prebaked FP8 deployment path (validated)

Calibration can be baked at dev time so user machines never need modelopt:
plain fp32 ONNX export of `_UpsampleWrapper` → `modelopt.onnx.quantization.quantize(
quantize_mode="fp8", calibration_data=...)` (dev box, needs `nvidia-modelopt[onnx]`)
→ standard `QuantizeLinear/DequantizeLinear` FP8 ONNX (28 Q/DQ pairs, ~5 MB) →
user box compiles with the existing `compile_onnx_to_tensorrt_engine(...,
dynamic_batch=True)` — verified modelopt is never imported in the compiling
process. Caveats: FP8 needs SM 8.9+ (RTX 40xx/50xx; 20xx/30xx must keep fp16),
`BasicVSRPlusPlusNetSplit` would need a TrtRunner adapter for the upsample
call, and the measured win is small (~5-12% of the upsample stage = 1-3% e2e
without CUDA graphs; quality 63.9 dB). Baked asset ships as `model_weights/<stem>_upsample_fp8.onnx`
(regenerate with `scripts/export_upsample_fp8_onnx.py`).

## Reproduction (historical)

Both harnesses were deleted after this investigation closed and FP8 was dropped
from the product; restore them from git history first.

```bash
PY=~/.virtualenvs/model-opt/bin/python   # needs CC=gcc-15 CXX=g++-15 for modelopt cuda_ext JIT
$PY scripts/quantize_basicvsrpp.py synth --video <mosaic-src.mp4> --out calib
$PY scripts/quantize_basicvsrpp.py compile --data calib --precision fp8 --out engines_fp8
$PY scripts/quantize_basicvsrpp.py eval  --data calib --engines engines_fp8 --fallback <fp16 sub-engine dir>
$PY scripts/quantize_basicvsrpp.py bench --engines engines_fp8 --fallback <fp16 sub-engine dir>
```
