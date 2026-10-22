# CLI Reference

The command line does almost everything the GUI does. Only the Video Player,
LTX seed previews, mask suggestions and interactive image mode are GUI-only.
`jasna --help` always shows all options. This page explains them and gives
examples.

The first start after an install or update takes longer, because Jasna checks
its files first.

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image (routes to SD 1.5 automatically)
jasna --input photo.png --output restored.png

# Whole folder (images first, then videos)
jasna --input input_folder --output output_folder
```

On Windows the CLI is the same file as the app: `jasna.exe --input ...`.

## Windows temporary files

Jasna can keep its temporary files on a RAM disk, for example ImDisk (NTFS).
This spares your SSD. Set `TMP` and `TEMP` before starting Jasna:

```powershell
$env:TMP = 'Z:\Temp'
$env:TEMP = 'Z:\Temp'
.\jasna.exe --version
```

Jasna uses `TMP` first, then `TEMP`. `TMPDIR` is not used.
`--working-directory` sets the folder for range and LTX files separately.

## General

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--version` | — | Print the Jasna version and exit. |
| `--input` | — | Video, image, or folder. |
| `--output` | — | Output file, or output folder when `--input` is a folder. |
| `--output-pattern` | `{original}_out` | Filename template for folder input. `{original}` is the input stem. Images keep their source extension; videos use the template extension when provided. Jasna checks planned outputs before processing and errors out if two inputs map to the same file. |
| `--device` | `cuda:0` | GPU selection. AMD cards use the same `cuda:N` names through ROCm. |
| `--batch-size` | `4`, AMD `8` | Detection batch size. Legacy `rfdetr-v5` always uses 4. |
| `--fp16` / `--no-fp16` | on | Half precision: less VRAM, often faster, no visible quality loss. |
| `--log-level` | `error` | `debug`, `info`, `warning`, `error`. |
| `--no-progress` | off | Disable the progress bar. |

## Restoration

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--restoration-model-name` | `basicvsrpp` | Video restoration model: `basicvsrpp` (fast), `ltx` (Standard Quality LTX: more detail, much slower; recommended), `ltx-4step` (Good Quality: a bit more detail than `ltx`, about 1.5x slower), `ltx-8step` (High Quality: more detail than `ltx`, about 2.5x slower) or `ltx-undistilled` (Top Quality: slightly better, about 5x slower than `ltx`). See [Choosing models](models.md#restoration-model). |
| `--restoration-model-path` | `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth`, or `model_weights/ltx-restore` for LTX | Restoration weights (a folder for LTX). |
| `--ltx-fast` / `--no-ltx-fast` | off | LTX: about 1.4x faster, slightly less detail. RTX 50-series only; other GPUs stop with an error. |
| `--ltx-large-canvas` / `--no-ltx-large-canvas` | off | LTX: more detail on large mosaics, about 3x slower on those. Needs about 12 GB of free VRAM; with less, those mosaics are restored normally. |
| `--experimental-ltx-less-overlap` / `--no-experimental-ltx-less-overlap` | on | LTX, experimental: about 13% faster when a mosaic stays on screen for more than 4 seconds. Turn it off for the older mode with more overlap. |
| `--ltx-seed` | `20260923` | LTX: another seed gives a slightly different result; the same seed repeats it. |
| `--ltx-trial` / `--no-ltx-trial` | off | LTX: test the speed without the model or a license. The video is not restored. |
| `--ltx-auto-split` | — | LTX: big mosaics (at least this percent of the video height, e.g. `12`) with LTX, smaller ones with BasicVSR++. Works on folders. Cannot be combined with `--segments` or `--fmp4`. See [Segments](segments.md#cli---ltx-auto-split). |
| `--compile-basicvsrpp` / `--no-compile-basicvsrpp` | on | Prepares BasicVSR++ for your GPU: about 3x faster, almost no extra VRAM. The first run takes 15–60 minutes. NVIDIA only. See [Tuning](tuning.md). |
| `--max-clip-size` | `90` | Frames restored together. Bigger is steadier and faster but uses more VRAM. Use the biggest that fits. |
| `--temporal-overlap` | `8` | Frames shared by neighboring clips, to hide flicker where they meet. `8`–`20` works well; must be less than half of `--max-clip-size`. |
| `--enable-crossfade` / `--no-enable-crossfade` | on | Cross-fade clip boundaries using already-processed frames; no extra GPU cost. |
| `--denoise` | `none` | Removes noise in restored areas: `low`, `medium`, `high`. |
| `--denoise-step` | `after_primary` | Remove noise before secondary restoration (`after_primary`) or after it (`after_secondary`). |

The `--ltx-*` options need `--restoration-model-name ltx`, `ltx-4step`, `ltx-8step` or `ltx-undistilled`.
`--no-compile-basicvsrpp` works only with `basicvsrpp`.

## Detection

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--detection-model` | `rfdetr-v6` | Installed models are discovered from `model_weights/`; `rfdetr-v6` (fast), `rfdetr-vr-v1` (VR180) and `lada-yolo-v4` are bundled; `rfdetr-v6-large` and `zelefans-vr-yolo-v2` are optional downloads. See [Models](models.md). |
| `--detection-model-path` | auto | Defaults to `model_weights/<detection-model>` with the right file type for your card: RF-DETR uses `.onnx` on NVIDIA and `.pt` on AMD; YOLO always uses `.pt`. |
| `--detection-score-threshold` | auto | From `0` to `1`. Defaults to the model's recommended value (`rfdetr-v6`: 0.35, `rfdetr-v6-large` and `rfdetr-vr-v1`: 0.40, YOLO models: 0.25). Lower it when mosaics are missed; raise it when normal areas get falsely detected. |
| `--max-detection-gap` | `2` | Fill detection dropouts up to N frames when the mosaic reappears at the same spot. `0` disables. |
| `--min-detection-duration` | `2` | Drop detections shorter than N frames as false positives; those frames stay unrestored. `0` disables. |
| `--scene-detection` | on | Detect hard scene cuts and end all tracked mosaic clips at the cut, so no clip spans two shots. Disable with `--no-scene-detection`. |

## Secondary restoration

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--secondary-restoration` | `none` | `unet-4x`, `tvai`, or `rtx-super-res`. See [Models](models.md). |
| `--rtx-scale` | `4` | RTX Super Res upscale factor (`2` or `4`). |
| `--rtx-quality` | `high` | `low`, `medium`, `high`, `ultra`. |
| `--rtx-denoise` | `medium` | `none`, `low`, `medium`, `high`, `ultra`. |
| `--rtx-deblur` | `none` | `none`, `low`, `medium`, `high`, `ultra`. |
| `--tvai-ffmpeg-path` | Topaz default install path | Path to Topaz Video `ffmpeg.exe` (Windows only). |
| `--tvai-model` | `iris-2` | e.g. `iris-2`, `prob-4`, `iris-3`. |
| `--tvai-scale` | `4` | Output size is `256*scale`; `1` = no scale. |
| `--tvai-args` | see `--help` | Extra `tvai_up` parameters. |
| `--tvai-workers` | `2` | Parallel TVAI ffmpeg workers. |
| `--tvai-denoise` | off | Remove noise before the Topaz model runs. Slower, uses more VRAM. |

## SD 1.5 image restoration

Still images route here automatically; `--restoration-model-name` is video-only.

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--image-restoration-model-name` | `sd-15-jav` | Only current value. |
| `--sd15-steps` | `25` | How many times the picture is refined. More can add detail, but is slower. |
| `--sd15-strength` | `0.6` | How much the mosaic area is redrawn, up to `0.7`. Higher fixes heavy mosaics but may look less like the original. |
| `--sd15-freeu` / `--no-sd15-freeu` | on | Often makes shapes sharper. Turn off if you see harsh edges. |
| `--sd15-seed` | `0` | Seed. The same seed and settings repeat the same result. |
| `--sd15-variants` | `1` | Saves N versions as `name_v1` … `name_vN` (seeds `seed` … `seed+N-1`). Pick the best one yourself. |

## VR

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--vr-mode` | `auto` | `auto`, `off`, `sbs`, `sbs-fisheye`. See [VR180](vr180.md). |

## Continuing LTX jobs

After Ctrl+C, an error or closing the app, run the same LTX command again to continue
from saved work. This also works with `--segments` and `--ltx-auto-split`. The unfinished
track may run again, and writing the output starts over. If the input, settings or app
version changed, Jasna asks before replacing saved work; without a terminal, it stops
and keeps the saved work.

## Encoding

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--codec` | `hevc` | `hevc`, `h264`, or `av1` for offline output. HLS streaming always uses H.264. |
| `--cq` | GPU/codec-specific | Video quality. Lower is better quality and a larger file. NVIDIA defaults: H.264 25, HEVC 28, AV1 35. AMD defaults: H.264 24, HEVC 25, AV1 16. |
| `--encoder-settings` | — | Advanced settings as a JSON object or comma-separated `key=value` pairs, e.g. `{"rc-lookahead":32}` or `rc-lookahead=32,bf=4`. See below. |
| `--lut` | — | `.cube` color LUT (1D or 3D) applied as the video is written. Also available in the GUI's Encoding section. |
| `--sharpen` | `0` | Sharpen the picture before encoding, from `0` (off) to `1` (strongest). Matches ffmpeg's `cas` filter, so no second pass is needed. See [Advanced processing](advanced_processing.md). |
| `--retarget-high-fps` | off | 60 → 30 FPS (and 59.94 → 29.97) by keeping every second frame. Other rates unchanged; audio stays in sync. Not available with `--stream` or `--segments`. |
| `--fmp4` | off | Play `.mp4`/`.mov` output while it is still being made; it also survives an interrupted job. Not available with `--stream`, `--segments` or `--ltx-auto-split`. See [Advanced processing](advanced_processing.md). |
| `--segments` | — | Restore only these ranges of one video, in seconds or `mm:ss`, e.g. `10-25,01:10-01:30.5`. Cannot be combined with `--stream`, `--retarget-high-fps`, `--fmp4` or `--ltx-auto-split`, or with image or folder input. See [Segments](segments.md). |
| `--working-directory` | output dir | Where temporary files and saved LTX progress are written. See [Segments](segments.md). |

### Choosing a codec

- **`hevc`** (default): good quality for the file size, 10-bit. Plays on most
  modern devices. Use this if you are not sure.
- **`h264`**: plays almost everywhere (old TVs, browsers, video editors).
  8-bit only, bigger files at the same quality. Streaming always uses it.
- **`av1`**: smallest files at the same quality, 10-bit. Needs a GPU that can
  encode AV1 (NVIDIA RTX 40-series, AMD RX 7000 or newer) and a newer player.

With `--segments`, the output always has the codec of the input. A different
`--codec` stops with an error.

### Encoder settings

`--cq` is the main quality setting. Lower means better quality and a bigger
file. Jasna sends the number to the encoder as it is (only AMD AV1 multiplies
it by 5). When you change the codec, the number is not converted.

| GPU | H.264 default | HEVC default | AV1 default | Accepted range |
| --- | ---: | ---: | ---: | --- |
| NVIDIA | 25 | 28 | 35 | 1–51 for H.264/HEVC; 1–63 for AV1 |
| AMD | 24 | 25 | 32 | H.264/HEVC: 0–51; AV1: 1–51 |

On NVIDIA, CQ 0 means "automatic", so Jasna needs a value of 1 or more. The
GUI keeps a separate CQ value for each codec.

`--encoder-settings` changes other encoder options. Jasna checks every key.
A key the encoder doesn't know stops the job with an error that lists the
keys it accepts:

```bash
# Higher quality (bigger file): lower CQ.
jasna --input in.mp4 --output out.mkv --cq 22

# CQ plus advanced keys
jasna --input in.mp4 --output out.mkv --cq 22 --encoder-settings "rc-lookahead=32,bf=4"
```

`cq=22` inside `--encoder-settings` still works when you don't use `--cq`.
Setting CQ in both places is an error. In the GUI, CQ can be set only with the
CQ control, not in **Custom Args**.

#### NVIDIA (NVENC) keys, all codecs

| Key | What it does |
| --- | ------------ |
| `cq` | Target quality for VBR. Lower = better quality and bigger file. Range 1–51 for H.264/HEVC (defaults 25/28), 1–63 for AV1 (default 35). The automatic size ceiling can make nearby values behave alike. |
| `preset` | Speed/quality trade-off, `p1` (fastest) to `p7` (best). Default `p5`. |
| `tune` | `hq` (default), `ll`, `ull`, or `lossless`. |
| `rc` | Rate-control mode: `vbr` (default), `cbr`, `constqp`. |
| `qmin` / `qmax` | Quality floor/ceiling for VBR. Defaults 17/34 (H.264/HEVC only; AV1 uses a different 0–255 QP scale and leaves these unset). |
| `init_qpI` / `init_qpP` / `init_qpB` | Initial quantizer per frame type. Default 17 (H.264/HEVC). |
| `g` | Keyframe interval in frames. Default 250. Smaller = better seeking, bigger file. |
| `bf` | Max consecutive B-frames. Default 4. |
| `b_ref_mode` | Use B-frames as references: `disabled`, `each`, `middle` (default). |
| `b_adapt` | Adaptive B-frame placement. |
| `nonref_p` | Non-reference P-frames, enabled by default. |
| `spatial_aq` / `spatial-aq` | Spatial adaptive quantization: spends more bits where the eye notices. On by default. AV1 accepts only the hyphenated spelling. |
| `temporal-aq` | Temporal adaptive quantization. On by default. |
| `aq-strength` | AQ aggressiveness, 1–15. Default 8. |
| `rc-lookahead` | Frames analyzed ahead for rate control. Default 32. |
| `lookahead_level` | Lookahead quality, 0–3. HEVC/AV1 only. On H.264 it is ignored with a warning. |
| `maxrate` / `bufsize` | Bitrate cap and VBV buffer size, in bits per second. Jasna sets these automatically from the source bitrate (see below); setting `maxrate` yourself replaces that. |
| `multipass` | Two-pass encoding: `disabled`, `qres`, `fullres`. |
| `weighted_pred` | Weighted prediction. NVENC supports it only together with `bf=0`; otherwise (and always on AV1) it is ignored with a warning. |
| `tf_level` | Temporal filtering level. |

Per-codec extras:

| Codec | Extra keys |
| ----- | ---------- |
| `hevc` | `profile` (`main`, `main10` — default `main10`), `tier` |
| `h264` | `profile` (`baseline`, `main`, `high` — default `high`), `coder` (`cabac`/`cavlc`) |
| `av1` | `tier`, `tile-rows`, `tile-columns` (parallelize decode of large frames) |

#### Automatic output size ceiling

`cq` aims for a fixed quality, no matter how the source was encoded. A source
with low bitrate would then come out several times bigger. To stop that, Jasna
sets `maxrate` from the bitrate of the source video, and `bufsize` to twice
that:

| Case | Ceiling |
| ---- | ------- |
| NVIDIA H.264 output | 2.0 x source video bitrate |
| Other output from an HEVC source | 1.25 x source video bitrate |
| Other combinations | 1.0 x source video bitrate |

NVIDIA H.264 gets more room, because it needs more bits to keep the restored
detail. The limit matters only for sources with low bitrate. A source with high
bitrate stays below it anyway. When the limit is hit, close CQ values can give
the same file size.

Set your own `maxrate` to replace the limit, or a very high one to turn it off.
If the source has no bitrate information, Jasna logs a warning and uses no
limit. AMD uses constant QP by default and has no limit.

#### AMD (AMF) keys, all codecs

| Key | What it does |
| --- | ------------ |
| `cq` | Quality value for AMF, sent as constant QP (AV1 uses a finer internal scale, so CQ is multiplied by 5). With `rc=qvbr` on H.264 it sets the QVBR level instead. Lower = better. Defaults: H.264 24, HEVC 25, AV1 16. |
| `qvbr_quality_level` | Legacy CQ alias in CLI advanced settings when `--cq` is omitted; not accepted in the GUI custom-args field. |
| `usage` | Encoder usage profile. Default `high_quality`. |
| `quality` | Speed/quality preset: `speed`, `balanced`, `quality` (default). |
| `rc` | Rate-control mode. Default `cqp`. |
| `preset` | AMF preset. |
| `g` | Keyframe interval in frames. Default 250. |
| `bf` | Max consecutive B-frames. |
| `preanalysis` | Pre-analysis pass. Off by default. |
| `maxrate` / `bufsize` | Bitrate cap and VBV buffer size, in bits per second. Set automatically from the source bitrate except in CQP mode. |
| `profile` / `level` | Codec profile and level. |

Per-codec extras:

| Codec | Extra keys |
| ----- | ---------- |
| `hevc` | `tier`, `bitdepth` (default 10), `vbaq` (off by default) |
| `h264` | `coder`, `bf_ref` (B-frame references), `pa_adaptive_mini_gop` (adaptive B-frame placement), `vbaq` (off by default) |
| `av1` | `bitdepth` (default 10), `aq_mode` (off by default) |

## Streaming

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--stream` | off | HLS streaming mode, no file output. See [Streaming](streaming.md). |
| `--stream-port` | `8765` | HTTP port. |
| `--stream-segment-duration` | `4.0` | HLS segment length in seconds. |
| `--no-browser` | off | Don't open a browser window. |

## Post-export

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--post-export-action` | `none` | Runs after all videos finish. `shutdown` powers off the PC right away (no countdown, unlike the GUI); `command` runs `--post-export-command`. |
| `--post-export-command` | — | Shell command for `--post-export-action command`. |
| `--post-export-video-command` | — | Shell command run after each successful video. Supports `{input}`, `{output}`, `{output_dir}`, `{output_stem}`, and `{output_suffix}`. |

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
jasna --input folder_in --output folder_out --post-export-video-command "ffmpeg -i {output} -map 0 -map_metadata 0 -map_chapters 0 -c copy -movflags +faststart {output_dir}/{output_stem}_remuxed{output_suffix}"
```

The last example keeps the original output and writes a second, remuxed file.
To replace the file, use a script with a temporary file. `ffmpeg` must be on
`PATH`, or write its full path.

## Activation

| Option | Notes |
| ------ | ----- |
| `--machine-key` | Print the Machine key and exit. No GPU or GUI startup is needed. |
| `--license-code CODE` | Activate online with your License code before GPU setup. With no processing options, activate and exit. |
| `--cloud-workspace` | Create or reuse a cloud identity with `--machine-key` or `--license-code`. CLI only: the official Linux release must run inside a supported container, with cloud turned on for your License code. |
| `--activation-key KEY` | Save the Activation key for this machine. With no processing options, activate and exit. |

Online activation:

```bash
jasna --license-code "YOUR_LICENSE_CODE"
jasna --license-code "YOUR_LICENSE_CODE" --input input.mp4 --output output.mkv
```

Manual activation: print the Machine key, enter it with your License code on
the [activation page](https://jasna-activation.kruk2.workers.dev) from any
device, then save the Activation key it returns:

```bash
jasna --machine-key
jasna --activation-key "YOUR_ACTIVATION_KEY"
```

Don't combine `--license-code` with `--machine-key` or `--activation-key`.

- Rented GPUs (RunPod, Vast): add `--cloud-workspace`, and set
  `XDG_CONFIG_HOME` to a private folder that you back up. It holds your
  activation, so use the same folder every run:

  ```bash
  export XDG_CONFIG_HOME=/root/jasna-config
  ./jasna --license-code "YOUR_LICENSE_CODE" --cloud-workspace
  ```

  Don't use RunPod's `/workspace`, which ignores private file permissions.
  Full steps: [Cloud GPUs](cloud.md).
- Device slots, privacy and what to do when activation fails:
  [Activate supporter models](gui.md#activate-supporter-models).

## Benchmark

| Option | Notes |
| ------ | ----- |
| `--benchmark` | Run benchmarks instead of processing. |
| `--benchmark-filter` | Only benchmarks whose name contains this string. |
| `--benchmark-video` | Video path for the benchmark; can be repeated. |
