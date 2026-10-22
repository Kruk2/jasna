# RX 9060 XT validation, 2026-10-22

The AMD optimizations carry over to the RX 9060 XT. With the same Torch environment,
RF-DETR + BasicVSR++ increases from 19.89 to 54.30 pipeline fps (2.73x). Against the
official Linux AMD 0.10.0 binary, the increase is 13.15 to 54.30 fps (4.13x).
YOLO v4 increases from 29.71 to 59.51 fps against that release (2.00x).
The tested frozen 0.11.3 candidate cannot complete either detector path. Source
LTX trial completes at 512 px; encrypted LTX is outside this validation.

Raw rows: [2026-10-22_amd_rx9060xt_validation.csv](2026-10-22_amd_rx9060xt_validation.csv).
All timed repeats and final output checks are complete. Follow-up packaging fixes
were made after validation; their rebuilt frozen package remains untested.

## Scope and environment

- One RX 9060 XT, gfx1200, 16 GB; Core i5-6600K, four CPU cores, 16 GB host RAM,
  35 GB swap, Ubuntu 24.04.5. GPU work runs exclusively under one validation lock.
- Archived source: before the AMD push `3a50f3e83f32d0d98aa21ba1ad88169d22349f59`;
  optimized `48f8bf8f66dc53442664bfe036279497a086f866`. Later local commits and
  uncommitted changes are not included in these numbers.
- Both source versions use Python 3.12.15, Torch 2.12.0+rocm10.0.0,
  torchvision 0.27.0+rocm10.0.0, PyAV 18.1.0 and matching gfx1200 device packs.
- The actual [official v0.10.0 Linux AMD release](https://github.com/Kruk2/jasna/releases/tag/v0.10.0)
  was downloaded and both split parts checked against GitHub's SHA256 digests.
  It bundles Torch 2.9.1+rocm7.2.0. Its launcher retains its original MIOPEN FAST
  default; source uses HYBRID. This pod needs the per-process
  `LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib` to resolve the old binary's external
  ROCm libraries. Therefore the release comparison includes runtime/default changes;
  the source comparison above isolates code changes in one environment.
- AMF 26.10.1 (`26.10.499-1`) and FFmpeg/ffprobe 8.1.2 are used. AMF 26.20
  cannot encode on this host: its VCN firmware is 1.12, while that runtime requires
  1.13. No host firmware update was attempted.

Performance validation used unmodified archived source and unchanged frozen
payloads, with scratch scripts and environment/dependency setup on the 9060 XT
pod. The R9700 pod was used read-only. Follow-up fixes below do not change the
code versions attached to these measurements. No changes were pushed.

## End-to-end method and results

Owner clip: H.264, 1920x1080, 30 fps, 177 seconds, 5310 frames, AAC audio.
One discarded warm-up and two timed runs per comparison, with idle checks and
other-process CPU usage recorded. Shared settings:

```sh
--secondary-restoration none --batch-size 4 --max-clip-size 90 \
--temporal-overlap 8 --codec hevc --no-progress --log-level info
```

Detection model is `rfdetr-v6` or `lada-yolo-v4`. Restoration is BasicVSR++ with
the same bundled checkpoint. HEVC quality stays at 25 in both versions. H.264
and AV1 defaults changed between releases, so they are not used for the timing
comparison. Wall time includes process startup and shutdown; pipeline time runs
from the Processing log through encoder flush/mux. FPS is frames / pipeline time.

| Version | Detector | Mean wall s | Mean pipeline s | Mean fps | Mean peak VRAM MiB |
|---|---|---:|---:|---:|---:|
| official 0.10.0 | RF-DETR v6 | 411.3 | 403.8 | 13.15 | 4810 |
| source before AMD push, 3a50f3e | RF-DETR v6 | 275.4 | 267.0 | 19.89 | 3301 |
| optimized source, 48f8bf8 | RF-DETR v6 | 105.4 | 97.8 | 54.30 | 2868 |
| official 0.10.0 | YOLO v4 | 185.0 | 178.8 | 29.71 | 2715 |
| optimized source, 48f8bf8 | YOLO v4 | 95.3 | 89.3 | 59.51 | 2377 |

RF-DETR wall time improves 2.61x against the previous source and 3.90x against
official 0.10.0. In optimized RF-DETR run 2, detect/track takes 94.9 s and primary
restoration 42.2 s; detection now limits this clip.
YOLO wall time improves 1.94x against official 0.10.0. Its optimized run 2 spends
87.8 s in detect/track and 26.1 s in primary restoration; it is also limited by detection.

## BasicVSR++ alone

Actual restoration checkpoint, deterministic synthetic uint8 input, fp16, one
warm-up and three timed repetitions, GPU synchronized. Values are medians.
These are model-only timings, not whole-video throughput.

| Clip frames | Before, ms/frame | Optimized, ms/frame | Speedup |
|---|---:|---:|---:|
| 16 | 17.106 | 4.854 | 3.52x |
| 90 | 17.060 | 4.508 | 3.78x |

The 16-frame old/new float outputs have PSNR 62.48 dB and maximum absolute
difference 0.02490. The focused fused-kernel tests also compare actual gfx1200
results with FP32 references.

## Frozen 0.11.3: fails qualification

The final completed candidate is based on main `c6860d4`, with an 18,382,283,239-byte payload.
A checksum rsync comparison with the finished local dist was empty, and all four
final archive parts pass their generated SHA256 manifest. The older archive
tested earlier is retained separately in raw evidence; the failures below were
reproduced on the final candidate.

`--version`, bundled Python loading, and an Xvfb GUI font probe pass. Video
processing does not:

| Check | Observed result |
|---|---|
| Official AMF installed in `/opt/amf`, registered in the system loader cache | Frozen preflight raises `ImportError: unverified runtime library`. |
| Add a standard system-library symlink to that installed AMF library | Preflight passes; BasicVSR++ `.half()` fails with `hipErrorInvalidImage`. |
| Prefix the existing signed Torch gfx1200 archive through `ROCM_KPACK_PATH_PREFIX` | BasicVSR++ loads; RF-DETR fails importing Hugging Face with partially initialized `file_download` missing `_CACHED_NO_EXIST`. |
| Same Torch-only prefix, YOLO v4 | Reaches Torchvision NMS and fails with `hipErrorInvalidDeviceFunction`. |

`ROCM_KPACK_DEBUG=1` confirms that the relocated root `libtorch_hip.so` searches
for Torch's archive outside the install folder, although its byte-identical
gfx1200 archive exists under `torch/.kpack`. Torchvision's `.kpack` archives are
absent altogether; debug lookup failures and the NMS traceback confirm that
second packaging defect. The frozen Hugging Face failure is an initialization
problem, not a confirmed missing package; the ordinary import succeeds in the
unchanged build image. Its exact reentrant import edge remains unproved.

The environment diagnostics did not modify the frozen release files. No frozen
performance number is reported. The AMF symlink and working 26.10.1 runtime
remain installed on the validation pod.

## Source LTX trial

CLI `--restoration-model-name ltx --ltx-trial`, 512 px, HEVC input and H.264 output:
all 300 input frames produce a 10-second 1920x1080 output. This test input has no
audio. Placeholder weights exercise the real passes but do not restore image quality.

The cold speed test takes 523.1 s for 209 frames / 4 windows, with an application
reported peak of 15422.7 MiB. The subsequent full-video pass takes 218.0 s for
300 frames / 15 tracks / 15 windows, reporting peak 11825.1 MiB. During that pass,
40 of 48 transformer blocks stay resident and 8 stream from RAM. No OOM occurs.
These timings describe this trial, not licensed model load or restoration quality.

The generic BasicVSR benchmark harness cannot form an LTX timing row because LTX
does not emit its Processing marker. The application exits successfully and logs
its final memory/end markers; LTX timings above come from its own stage logs.

Read-only review finds that the GUI still gates LTX on NVIDIA, so this AMD trial
is validated through the CLI only. Frozen/encrypted LTX, 768 px, Windows AMD,
dual-GPU affinity and interactive GUI/player/streaming/smart-render paths are
not qualified by this run.

## Correctness and media checks

146 distinct focused tests pass: 95 model/media-kernel tests and 51 AMF,
private-stream, seek and mux tests. Three expected AV1 width-padding failures
remain. Six initial subprocess import failures were rerun successfully with the
archived source root on `PYTHONPATH`; both initial and retry logs are retained.

Functional renders complete for H.264; 4K HEVC Main10 (303 frames, YOLO); AV1
8-bit and 10-bit inputs (1800 frames each, RF-DETR, AV1 output). Direct AMF
H.264/HEVC/AV1 encode-and-decode probes also pass with runtime 26.10.1.
Non-multiple-of-eight AV1 output widths can still be padded; this is not fixed.

All 16 timed and functional outputs fully decode: 57903 frames, expected sizes,
strictly increasing timestamps and exact expected frame counts. All 13 outputs
with audio preserve their source audio packets byte for byte. Redundant warm-up
outputs are excluded from this full check. Both downloaded RF-DETR videos also
contain 5310 frames and preserve all 8298 AAC packets against the local source.

AMF versus forced-software decoding matches RGB bytes and timestamps exactly for
32 frames each of H.264, 4K HEVC Main10, AV1 8-bit and AV1 10-bit (128 sampled
frames total). Both decoder paths use the same separately tested YUV converter;
this comparison isolates actual codec reconstruction and layout/depth handling.

Every image comparison below covers all 5310 frames; none falls below the 35 dB
per-frame inspection threshold.

| Comparison | PSNR dB | SSIM | Worst frame dB |
|---|---:|---:|---:|
| RF-DETR official 0.10 r1 vs optimized r1 | 47.998 | 0.993563 | 46.68 |
| RF-DETR optimized r1 vs r2 | 48.122 | 0.993679 | 46.65 |
| YOLO official 0.10 r1 vs optimized r1 | 48.128 | 0.993675 | 46.53 |
| YOLO optimized r1 vs r2 | 48.910 | 0.994573 | 47.45 |

Agreement is not a ground-truth restoration-quality measurement. The owner also
inspected the downloaded outputs and reported them OK.

## Follow-up fixes and remaining qualification

Follow-up packaging fixes add support for the standard AMF installation,
correct Torch/Torchvision kernel-archive lookup, and retain the Hugging Face
dependencies required by the release. These fixes were made after the
measurements above and require a fresh release build and GPU retest.

Regression checks reproduced the failures before the fixes. The resulting CPU
suite passes 492 tests, with 15 skipped. A small actual ELF fixture verifies the
Torch library/archive location. A CPU-only smoke test in the existing AMD build
image imports Hugging Face Hub 1.33.0 and constructs a tiny DINOv2 AutoBackbone
with metadata restricted to the intended bundled list. Read-only review passes.

These checks do not qualify a rebuilt frozen release. Next: rebuild from the
committed source revision, verify the archive, and run unchanged YOLO
and RF-DETR renders on the 9060 XT without the diagnostic symlink/environment
workarounds. The frozen Hugging Face import must be checked explicitly.

Raw logs, artifact identities and output-validation records are retained locally.
