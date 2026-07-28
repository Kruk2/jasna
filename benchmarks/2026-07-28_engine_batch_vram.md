# Fixed-batch preprocess/upsample sub-engines; FP8 upsample removed (2026-07-28)

Source tree v0.9.1+, RTX 5090, driver 595.84. Canonical setup: vali,
`--secondary-restoration none --temporal-overlap 15 --max-clip-size 180`,
`scripts/benchmark_decode_backends.py`. VRAM is per-process
(`nvidia-smi --query-compute-apps`), not whole-GPU.

## Why

`2026-07-27_v0.9.1_e2e_cudagraphs.md` recorded the FP8 upsample sub-engine as
**−1 GB VRAM**. Re-measuring the shipped code showed the opposite: the FP8 path
*added* 1.8 GB. The study measured engines built by
`scripts/quantize_basicvsrpp.py compile`; what shipped is compiled from the
prebaked QDQ ONNX with `dynamic_batch=True` at the clip size, which is a
different artifact.

1080p h264, one run each, before this change:

| config | wall | vram med | vram peak |
|---|---:|---:|---:|
| fp16 b180, no graphs | 37.9 s | 9979 | 10429 |
| fp16 b180 + graphs | 34.5 s | 9517 | 10417 |
| fp8 b180, no graphs | 35.4 s | 11809 | 12259 |
| fp8 b180 + graphs (**shipped v0.9.1**) | 34.0 s | 11449 | 12035 |

CUDA graphs are fine: −9 % wall **and** −460 MiB. FP8 was the regression.

## Root cause: TensorRT scratch scales with the profile's max batch

`create_execution_context()` reserves the engine's `device_memory_size` up
front, and that is sized for the largest profile shape. The upsample engine was
built at the clip size, so a 180-frame clip reserved a 180-frame scratch — for a
stage whose math is strictly per-frame.

Upsample engine, 180 frames of work either way:

| batch | resident (load + I/O + warm) | median for 180 frames |
|---:|---:|---:|
| 180 | 4672 MiB | 15.03 ms |
| 90 | 2828 MiB | 15.34 ms |
| 60 | 2106 MiB | 14.78 ms |
| **30** | **1400 MiB** | **14.69 ms** |
| 16 | 1118 MiB | 14.85 ms |

Same work, same speed, −3.3 GB.

### FP8, properly re-tested at small batch

Two different FP8 build routes exist and they are not interchangeable:

| upsample engine | route | resident | ms/180 f (CUDA events) | full-model PSNR vs fp32 |
|---|---|---:|---:|---:|
| fp16 b30 (**shipping**) | torch-TRT | 748 MiB | 15.06 / 15.11 | 74.42 dB, SSIM 1.0000 |
| fp16 b60 | torch-TRT | 1392 MiB | 15.27 | 74.42 dB |
| fp16 b180 (old) | torch-TRT | 4422 MiB | 15.1 | 74.42 dB |
| fp8 b30 | modelopt + torch-TRT | **560 MiB** | **13.56 / 13.94** | 64.21 dB, SSIM 0.9998 |
| fp8 b60 | modelopt + torch-TRT | 1018 MiB | 13.87 / 14.01 | 64.21 dB |
| fp8 b30 | prebaked QDQ ONNX + OnnxParser (**what v0.9.1 shipped**) | ~960 MiB | 15.36 | — |
| fp8 b180 | prebaked QDQ ONNX + OnnxParser | ~5700 MiB | — | — |

So the modelopt/torch-TRT FP8 engine *is* genuinely better in isolation at
b30: **−188 MiB and −8 % on the upsample stage**. The ONNX route we can
actually ship is worse than fp16 on both counts — its
`_build_serialized_engine` sets `BuilderFlag.FP16` on an fp32-typed QDQ
network and TensorRT collapses the whole graph into one Myelin ForeignNode
(`ForeignNode[input_QuantizeLinear.../conv_last/Conv]`), whose scratch is the
5670 MiB. Rebuilding under a 4 GB workspace cap fails outright; building
`STRONGLY_TYPED` instead gives 945 MiB / 18.66 ms, still worse than fp16 b30.

E2E is what settles it. 1080p h264, clip 180, fp8 b30 wired in place of fp16
b30, two runs each:

| | wall | vram med | vram peak |
|---|---:|---:|---:|
| fp16 b30 | 33.97 / 33.77 s | 4919 / 5039 | 5689 / 5785 |
| fp8 b30 | 34.01 / 33.83 s | 4789 / 4705 | 5579 / 5583 |

**No wall-clock difference at all** (the stage is ~6 % of restoration and
restoration overlaps decode/encode), −232 MiB median, and −10 dB of model
accuracy. Getting even that would mean shipping nvidia-modelopt plus
calibration clips to every user, because the shippable ONNX route does not
produce this engine. **Not worth it — FP8 stays removed.**

Whole six-engine set, resident above the CUDA context:

| | loop_body ×4 | preprocess | upsample | total |
|---|---:|---:|---:|---:|
| clip-sized (b180) | 184 MiB | 1292 MiB | 3084 MiB | 4560 MiB |
| **fixed batch** | 184 MiB | 422 MiB (b60) | 310 MiB (b30) | **916 MiB** |

## Correctness

- **Upsample is bit-exact.** `_UpsampleWrapper` is a per-frame conv stack; the
  b180 engine and 6×b30 give `torch.equal == True` on the same input, and the
  full 1080p e2e run is byte-identical (PSNR `inf`).
- **Preprocess feature maps are bit-exact**; SPyNet flows differ by ≤0.07 px
  and e2e output by 51.9 dB PSNR. That is *not* the batching: calling the b60,
  b90 or b150 engine on the **same 60 frames** with no batching at all gives the
  same 0.04–0.07 px spread against b180. It is TensorRT tactic selection per
  profile, which already differed between clip-60 and clip-180 users.
- Propagation (`loop_body`) is untouched, so clip length, temporal receptive
  field and window seams are unchanged.

Preprocess batches overlap by one frame because SPyNet consumes consecutive
pairs: batch `[a, b)` yields pairs `a..b-2` and the pair `(b-1, b)` comes from
the next batch's first pair.

## Preprocess batch choice

1080p h264, two runs each:

| preprocess batch | wall | vram med | vram peak |
|---|---:|---:|---:|
| 180 (one call per clip) | 32.0 / 32.2 s | 5808 / 5685 | 6565 / 6533 |
| **60 (batched)** | 32.9 / 32.5 s | 5077 / 4905 | 5789 / 5759 |

−780 MiB for +1.8 % wall. Taken, because the default clip size is 90 (two
batches, not three) and because it is what makes the engine set independent of
the clip size.

## E2E, h264 sources, medians of 3 (warm)

See `2026-07-28_engine_batch_vram.csv`. Comparison rows are from
`2026-07-27_v0.9.1_e2e_cudagraphs_{baseline,graphs_fp8}.csv`, same script and
machine.

| clip | wall (v0.9.1 shipped) | wall (new) | vram med v0.9.1 → new | vram peak v0.9.1 → new |
|---|---:|---:|---:|---:|
| 720p h264 | 31.6 s | 31.9 s | 7890 → **4312** MiB (−45 %) | 7968 → **4720** MiB (−41 %) |
| 1080p h264 | 33.6 s | 33.8 s | 8393 → **4867** MiB (−42 %) | 9293 → **5779** MiB (−38 %) |
| 2160p h264 | 62.9 s | 63.3 s | 10636 → **7326** MiB (−31 %) | 13620 → **11176** MiB (−18 %) |

Throughput 190.0 / 179.4 / 95.7 fps — within run noise of the v0.9.1 default
(191.8 / 180.0 / 96.2), and still well ahead of the pre-CUDA-graphs baseline
(35.3 / 36.2 / 67.1 s). The entire FP8 VRAM bill is gone and roughly 3 GB more
with it.

## Consequences

- All six sub-engines are now clip-size independent
  (`BASICVSRPP_PREPROCESS_BATCH = 60`, `BASICVSRPP_UPSAMPLE_BATCH = 30` in
  `jasna/engine_paths.py`). Changing max clip size no longer recompiles
  anything, and the upsample engine file drops from 452 MB to 78 MB.
- Removed: the FP8 upsample path (`_Fp8UpsampleEngine`,
  `_compile_fp8_upsample_engine_if_supported`, `supports_fp8`, the bundled
  `<stem>_upsample_fp8.onnx` asset, `scripts/export_upsample_fp8_onnx.py`) and
  the `JASNA_UPSAMPLE_ENGINE_OVERRIDE` escape hatch.
- `EngineCompilationRequest.basicvsrpp_max_clip_size` is gone; nothing in the
  engine layer takes a clip size any more.
- The GUI max-clip-size slider now runs to 720 instead of 180 (the CLI never
  capped it); a long clip costs activation memory only.

## Clip size after the change (1080p h264, overlap 8)

`2026-07-28_clip_size_sweep.csv`. Engines are constant now, so this is purely
activation memory — and it corrects two claims the tuning docs had been making:

| clip size | wall | vram med | vram peak |
|---:|---:|---:|---:|
| 30 | 51.4 s | 3411 | 3487 |
| 60 | 40.2 s | 3743 | 3907 |
| 90 | 34.4 s | 4095 | 4551 |
| 180 | 35.0 s | 5047 | 5591 |
| 360 | 32.0 s | 6307 | 7667 |
| 720 | 33.0 s | 8159 | 10957 |

- Small clips are **slower**, not faster: overlap frames are reprocessed per
  clip, so clip 30 costs 1.5× the wall of clip 90. The docs said the opposite.
- Past ~180 the wall stops improving and only VRAM grows.
- `--no-compile-basicvsrpp` at clip 180 now measures 106.8 s / 4605 MiB versus
  33.2 s / 4977 MiB compiled. It used to be the recommended VRAM escape hatch;
  it now buys 372 MiB for 3.2× the runtime, so the docs no longer suggest it.
