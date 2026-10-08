# AMD MIGraphX execution provider on Windows (ONNX Runtime)

Host: MSI PRO B650M-P, Ryzen 7 7800X3D (integrated Radeon, `gfx1036`), **RX 7900 XT**
(20 GB, `gfx1100`), Windows 11 26100, Adrenalin 32.0.31041.5005.
EP package `MicrosoftCorporationII.WinML.AMD.GPU.EP.1.8` v1.8.64.0 (AMD GPU EP
7.2.2609.19, installed on demand by Windows), `windowsml==1.8.2192` with
`onnxruntime-windowsml 1.23.5`; the newer rail (`windowsml[with-ort]==2.4.89`,
ORT 1.27.1) was tried first and cannot run it (see below).

Question: can the RF-DETR detector run on the AMD GPU through **MIGraphX (ROCm)**
rather than DirectML, and is it faster than the existing torch path?

## Result

Same batch (8), steady state, warm caches, RX 7900 XT. torch numbers are the production
path (`RfDetrTorchRunner`, fp16 autocast, flash SDP); ONNX numbers are MIGraphX through
the classic provider path below. The shipped `rfdetr-v6.onnx` is fixed at 576; the 480
ONNX files were exported from the same checkpoint for a like-for-like comparison.

| Path | 480 ms/frame | 576 ms/frame |
| ---- | -----------: | -----------: |
| torch + ROCm, fp16, flash SDP (current default) | **9.63** | 13.85 |
| ONNX Runtime + MIGraphX, fp32 | 11.26 | 17.1 – 17.8 |
| ONNX Runtime + MIGraphX, **fp16** (all-half weights) | **5.04 – 6.76** | not measured |
| ONNX Runtime, CPU reference (fp32) | 428 – 444 | 428 – 444 |

So a plain fp32 MIGraphX export is *slower* than torch at both resolutions, but a full
fp16 export is **~1.5-1.9x faster than the production torch path** (RDNA3 fp16 rate +
MIGraphX fusion). Model size halves too (130.5 -> 65.6 MB). MIGraphX first-run cost is
substantial: **131 – 152 s** to compile per model/shape, dropping to **0.1 – 4 s** on the
next process once the `.mxr` cache exists.

### fp16 correctness (real frames from the test clip, torch fp16 as reference)

| Check | Result |
| ----- | ------ |
| score max abs diff | 0.0405 (scores up to 0.92) |
| queries above 0.35 | identical, 600/600 query decisions |
| top-1 query per frame | 3/3 identical; top-1 box diff **0.0003** (normalised) |
| masks, binarised, all 200 queries | IoU 0.614 (9256 vs 9222 on-pixels) |

The mask IoU is computed over every query including the ~197 low-score ones whose masks
are noise in both engines; the raw mask-logit deviation of the all-fp16 export is real
(mean label diff 0.18 vs 0.0004 for the fp32 export, which matches torch to 0.0000 on
dets/labels). Before shipping an fp16 export, either use a mixed-precision conversion
(fp32 weights for norm/softmax/attention-sensitive ops, i.e. what autocast does) or
validate visually on real material.

### fp32 correctness (480 export vs ORT CPU, random batch)

dets/labels max diff 0.0000; masks max diff 0.1959 on 120x120 logits (sign flips at zero
crossings only).

Outputs are equivalent for practical purposes (same random batch, vs CPU):

| Output | max abs diff | mean abs diff | post-processing agreement |
| ------ | -----------: | ------------: | ------------------------- |
| `dets` | 0.2741 | 0.0002 | — |
| `labels` | 0.3820 | 0.0004 | score max diff **0.0019**, top-1 query identical |
| `masks` | 24.99 (logits) | 0.0161 | binarised IoU **0.9951** |

## End-to-end A/B (`--detection-engine torch` vs `migraphx`)

Clip: 20 s / 602 frames, `rfdetr-v6`, batch 8, `--secondary-restoration none`, second
of two runs each (the first is warm-up):

| Engine | detect-track | wall | progress-meter speed |
| ------ | -----------: | ---: | -------------------: |
| torch | 15.5 / 15.9 s | 30.8 / 32.1 s | 32 fps |
| **migraphx** | **12.2 / 12.5 s (-21 %)** | 30.7 / 30.8 s | **37.5 fps (+17 %)** |

Wall time does not move because the pipeline overlaps detection with restoration and
encode; the detection stage itself is 21 % cheaper, matching the model-level delta
(~5 ms/frame). Outputs are visually identical: PSNR migraphx-vs-torch **52.7 dB**
(average, min 46.9), same clip encoded independently.

## Jasna integration (`--detection-engine migraphx`)

`jasna --detection-engine migraphx` (AMD only; default `torch`) runs the detector through
`RfDetrMosaicDetectionModel(engine="migraphx")` -> `RfDetrMigraphxRunner`
(`jasna/mosaic/rfdetr_migraphx_runner.py`):

* on first use it exports the checkpoint to `model_weights/rfdetr-v6.migraphx.r480.b8.fp16.onnx`
  (static shape = engine batch, fp16, ~1 min) and compiles the MIGraphX program into
  `model_weights/migraphx-cache/` (~2.5 min); later runs load the cache in seconds,
* batches are padded to the engine batch (the export is static-shape), like the
  fixed-batch TensorRT engines,
* needs `windowsml==1.8.2192[with-ort]` installed (its ORT 1.23.5 replaces
  `onnxruntime-gpu`; nothing in jasna imports onnxruntime, so that is safe on AMD),
* the engine can also be selected with `JASNA_DETECTION_ENGINE=migraphx` (no GUI change
  required); `--detection-engine` is wired in `main.py` via
  `detection_registry.set_detection_engine`.

Verified end to end with the release pipeline on the 3 s test clip (both engines
produce output; the MIGraphX run processed at 18.7 fps vs 7.9 fps on the torch path in
the progress meter) and by unit tests in `tests/test_detection_engine.py`.

## The configuration that works

```python
os.add_dll_directory(ep_dir)                       # EP package ExecutionProvider folder
for dll in every *.dll in ep_dir:                  # siblings are loaded by bare name
    LoadLibraryExW(dll, None, LOAD_WITH_ALTERED_SEARCH_PATH)
shutil.copy2(ep_dir / "onnxruntime_providers_migraphx.dll", <ort>/capi/)  # ORT loads it from its own folder
session = ort.InferenceSession(model, providers=[
    ("MIGraphXExecutionProvider",
     {"device_id": "0", "migraphx_model_cache_dir": cache_dir}),   # cache_dir is NOT accepted here
    "CPUExecutionProvider"])
```

* Registering `onnxruntime_providers_migraphx.dll` (the library the 1.8 rail reports)
  through ORT's own `register_execution_provider_library` is not enough: the provider
  bridge resolves `onnxruntime_providers_<name>.dll` next to ORT's `onnxruntime.dll`, so
  the file has to be copied there and its siblings preloaded.
* **Do not set `HIP_VISIBLE_DEVICES`.** The plugin routes on the HIP device it sees:
  unset / `0` / `0,1` / `ROCR_VISIBLE_DEVICES=0` → `arch=gfx1100` → **MIGraphX**;
  `HIP_VISIBLE_DEVICES=1` → `arch=gfx1036` → **DirectML**. Routing can be traced with
  `ORT_AMDGPU_TRACE_ROUTING=1`, which prints e.g.
  `[amdgpu-routing] arch="gfx1100" model_arch=(none) model_fw=(none) -> MIGraphX`.
* Only `device_id` is accepted by the classic provider; `cache_dir`, `force_recompile`,
  `exhaustive_tune`, `static_pad_*`, `pinned`, `profile` belong to the plugin wrapper
  (`amdgpu-ep.dll`), where `Unknown provider option` silently falls back to CPU.

## What does not work

* **Plugin EP route** (`amdgpu-ep.dll` via `register_execution_provider_library` +
  `add_provider_for_devices`, the way the EP catalogue is meant to be used): sessions are
  created and the device is accepted, but inference then fails with
  `BatchOrCopyMLValue: allocator != nullptr was false. Failed to find allocator for
  device Device:[DeviceType:1 MemoryType:0 VendorId:4098 DeviceId:2]`. Passing more than
  one device is rejected outright (`INVALID_ARGUMENT: only supports selection for a
  single device when using None`) and ORT silently retries on CPU.
* **Its DirectML backend** crashes the process on a real model
  (`directml-backend.dll`, access violation) because it is built against the ORT 1.17
  plugin ABI while ORT is 1.27:
  `The requested API version [24] is not available, only API versions [1, 17] are supported.
  Current ORT Version is: 1.17.1`. This is why `HIP_VISIBLE_DEVICES=1` (which routes to
  DirectML) looks like it "runs on the EP" for a tiny conv but aborts on RF-DETR.
* `migraphx_model_cache_dir` / `cache_dir` are silently ignored where they do not apply,
  and the older names (`migraphx_save_compiled_model` &co.) are gone; a cache miss after
  changing EP version, GPU arch, compute mode or input shape is expected.

## Files

Scripts used (outside the repo, `D:\Strata-data\`): `bench_v6_classic2.py`
(copy provider + preload, one session), `bench_v6_mgx_final.py` (cold/warm + CPU check),
`check_v6_equiv.py` (post-processing equivalence), `winml_migraphx_run.py` (probe).
The EP package exposes `onnxruntime_providers_migraphx.dll`, `amdgpu-ep.dll`,
`migraphx*.dll`, `amdhip64_7.dll`, `amd_comgr*.dll` and `directml-backend.dll`; the
MIGraphX half is the ROCm stack, the DirectML half is only a fallback for GPUs MIGraphX
does not support.

## Restoration model (BasicVSR++): exportable with the community symbolic, but no win

The detector switch only covers `--detection-engine migraphx`. The main restoration model
(`BasicVSRPlusPlusGanNet` via `generator_ema`) was tried with `torch.onnx.export` on a
T=4 / 256x256 clip, opset 20:

* the recurrent `propagate`/`upsample` loops **trace fine** (they unroll; `grid_sampler`
  converts), and `F.affine_grid` needs **opset 20**,
* `torchvision::deform_conv2d` (`SecondOrderDeformableAlignment`, vendored
  `deformconv.py`) has no ONNX mapping in torch 2.14 - solved by the community package
  **`deform_conv2d_onnx_exporter`** (PyPI), which registers a symbolic that decomposes
  DCNv2 into standard ONNX ops. It needs a 5-line patch for torch 2.14
  (`torch.onnx._type_utils` moved to
  `torch.onnx._internal.torchscript_exporter._type_utils`; applied to the installed copy),
* with that: **export succeeds** in ~3 s (fp32 56.2 MB / fp16 28.6 MB).

Results on MIGraphX (classic provider path, T=4 batch 1):

| Variant | vs torch (same precision) | MIGraphX steady | torch steady |
| ------- | ------------------------- | --------------: | -----------: |
| fp32 | **max diff 0.00000 (bit-exact)** | 71.9 ms/call | 76.3 ms/call |
| fp16 | max diff 0.0088 (fp16 noise) | 48.8 ms/call | 47.4 ms/call |

The fp16 run initially aborted inside the opset-20 `AffineGrid` function expansion
(If/Split subgraphs that ORT cannot fold in fp16 - no CPU half kernels). Fixed by
never emitting `AffineGrid`: `flow_warp` always passes an identity 2x3 theta, so the
base grid is a constant and can be built from plain arithmetic ops (export-time
monkeypatch of `F.affine_grid`; the graph then contains 0 AffineGrid / 0 If / 0 Split).
With that, the fp16 path runs end to end on MIGraphX with fp16-level numerical
agreement - but its speed is **parity with torch**, not a win: the heavy part is the
grid_sample-based deformable decomposition itself, which MIGraphX cannot fuse better
than torch's CUDA kernels. Combined with one export+compile (~5.5 min) per clip length
T, the restoration model stays on the torch path.

## Community research + root cause (2026-10-07 evening): why restoration can't win on this EP build

### Root cause: kernel-launch bound, not compute bound

`MIGRAPHX_TRACE_EVAL=1` on the warm fp16 model (`basicvsrpp_t4.onnx`, T=4/256x256):

* the compiled program executes **135,447 instructions per call**, of which
  **41,929 are `code_object` launches** (real GPU kernels); the rest are load/slice/
  reshape_lazy bookkeeping;
* 41,929 launches per 48.8 ms call = **~1.2 µs of wall time per kernel** — the GPU is
  starved by launch overhead, identical to the launch-bound regime that motivated
  CUDA-graph replay for the TensorRT sub-engines on NVIDIA;
* the ONNX graph itself is already huge (**6,181 nodes**, 689 Conv, 1,843 Constant)
  because `deform_conv2d_onnx_exporter` decomposes every DCNv2 into
  grid_sample + gather/concat chains, and MIGraphX additionally lowers every
  GridSample to **concat + GatherND index materialisation** (its only implementation
  before the native kernel landed).

### Confirmed with a variant matrix (each a full recompile, fresh cache)

| Variant | steady ms/call | verdict |
| ------- | -------------: | ------- |
| baseline fp16 (warm cache) | 48.8 | reference |
| `MIGRAPHX_NSTREAMS=2` (no recompile) | **46.6** | ~5%, only real gain |
| `MIGRAPHX_ENABLE_NHWC=1` | 46.8 | none; diff 0.096 vs 0.0088, +5 min compile |
| MLIR input+reduce+GEG fusion | 48.6 | none |

Kernel-side switches cannot help: they improve per-kernel efficiency, and the cost is
in the *number* of kernels. `ORT_MIGRAPHX_EXHAUSTIVE_TUNE=1` was not run for the same
reason (kernel *selection*, not kernel *count*; hours of compile on a 6k-node graph).

### What the binary actually supports (strings enumerated from the DLLs)

The EP package's MIGraphX 7.2.2609.19 has **no HIP-graph capture** and **no
`MIGRAPHX_ENABLE_CK`**; available perf switches are `MIGRAPHX_ENABLE_NHWC`,
`MIGRAPHX_ENABLE_MLIR_{INPUT,REDUCE,GEG}_FUSION`, `MIGRAPHX_NSTREAMS`,
`MIGRAPHX_USE_FAST_SOFTMAX`, MLIR tuning vars, `MIGRAPHX_PROBLEM_CACHE`.
The ORT provider exposes `migraphx_exhaustive_tune` as a session option and honours
`ORT_MIGRAPHX_EXHAUSTIVE_TUNE` / `ORT_MIGRAPHX_MODEL_CACHE_PATH` env overrides on the
classic provider path (unlike the session-option dict, which only accepts `device_id`).

Also observed: ORT logs `Unsupported nodes: Resize` — one Resize falls back to CPU
with D2H/H2D copies each call. Fixing it (export-time replace with Gather-based
bilinear) is worth at most a few ms and cannot change the launch-bound verdict.

### The community fix that changes the picture: native GridSample kernel

* ROCm/AMDMIGraphX #4017 "GridSample generates >15M literal instructions" -> fixed by
  PR #4067 (literals tamed, decomposition kept);
* #5140 "GridSample is slow (InternImage < 1 fps)": the concat+GatherND lowering bakes
  ~327 MB of index literals per op (78 ms each on RX 7800 XT); **PR #5139** adds a
  native `gridsample` op + fused GPU JIT kernel (bilinear; nearest/bicubic still
  decomposed; `MIGRAPHX_DISABLE_GRIDSAMPLE_OP=1` reverts). Merged into develop
  **2026-09-30**; author-measured **5.9-8.6x** on GridSample-heavy models
  (internimage_640 263 -> 45 ms; warp_pyramid 10.5 -> 1.2 ms).
* Verified by binary string search: our EP build (7.2.2609.19) does **not** contain it.

**Conclusion for BasicVSR++:** the moment the Windows ML AMD GPU EP ships a MIGraphX
with #5139 (next EP package drop), the 14+ GridSample ops per forward collapse from
tens of thousands of launch-bound bookkeeping kernels into single fused kernels —
that is the realistic 2-3x path that would finally beat torch. Until then every
engine-level acceleration is capped by launch overhead: keep restoration on torch,
and re-run `bench_bv_variant.py` against `basicvsrpp_t4.onnx` when the EP updates
(a cache miss per shape/EP change is expected).

## Conclusion

The detector: `--detection-engine migraphx` (AMD) works end to end and is ~1.5-1.9x
faster per frame than the torch path with an fp16 export (fp32 export is slower than
torch). The restoration model cannot move to MIGraphX without surgery on the vendored
deformable-alignment op and stays on the torch path.
