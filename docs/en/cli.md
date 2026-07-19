# CLI Reference

Jasna's CLI mirrors the GUI. `jasna --help` always shows the full, current list of options; this page adds context and examples.

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image (routes to SD 1.5 automatically)
jasna --input photo.png --output restored.png

# Whole folder (images first, then videos)
jasna --input input_folder --output output_folder
```

On Windows the CLI is the same file as the app: `jasna.exe --input ...`.

## General

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--version` | — | Print the Jasna version and exit. |
| `--input` | — | Video, image, or folder. |
| `--output` | — | Output file, or output folder when `--input` is a folder. |
| `--output-pattern` | `{original}_out` | Filename template for folder input. `{original}` is the input stem. Images keep their source extension; videos use the template extension when provided. Jasna checks planned outputs before processing and errors out if two inputs map to the same file. |
| `--device` | `cuda:0` | GPU selection. On AMD Linux this also selects the MIGraphX GPU. |
| `--batch-size` | `4` | Detection batch size. |
| `--fp16` / `--no-fp16` | on | FP16 where supported (restoration + TensorRT). Lowers VRAM, may improve speed. |
| `--log-level` | `error` | `debug`, `info`, `warning`, `error`. |
| `--no-progress` | off | Disable the progress bar. |

## Restoration

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--restoration-model-name` | `basicvsrpp` | Video restoration model (only `basicvsrpp` for now). |
| `--restoration-model-path` | `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth` | Restoration weights. |
| `--compile-basicvsrpp` / `--no-compile-basicvsrpp` | on | TensorRT compilation: big speed boost, more VRAM. See [Tuning](tuning.md). |
| `--max-clip-size` | `90` | Maximum tracked clip length in frames. Main VRAM lever. |
| `--temporal-overlap` | `8` | Overlap+discard margin at clip splits; reduces boundary flicker. |
| `--enable-crossfade` / `--no-enable-crossfade` | on | Cross-fade clip boundaries using already-processed frames; no extra GPU cost. |
| `--denoise` | `none` | Spatial denoising of restored crops: `low`, `medium`, `high`. |
| `--denoise-step` | `after_primary` | Apply denoising before secondary (`after_primary`) or right before blend (`after_secondary`). |

## Detection

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--detection-model` | `rfdetr-v5` | Installed models are discovered from `model_weights/`; `zelefans-vr-yolo-v2` is bundled. See [Models](models.md). |
| `--detection-model-path` | auto | Defaults to `model_weights/<detection-model>.onnx` (RF-DETR) or `.pt` (YOLO). |
| `--detection-score-threshold` | `0.25` | Lower it when mosaics are missed; raise it when normal areas get falsely detected. |

## Secondary restoration

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--secondary-restoration` | `none` | `unet-4x`, `tvai`, or `rtx-super-res`. See [Models](models.md). |
| `--rtx-scale` | `4` | RTX Super Res upscale factor (`2` or `4`). |
| `--rtx-quality` | `high` | `low`–`ultra`. |
| `--rtx-denoise` | `medium` | `none` disables. |
| `--rtx-deblur` | `none` | `none` disables. |
| `--tvai-ffmpeg-path` | Topaz default install path | Path to Topaz Video `ffmpeg.exe`. |
| `--tvai-model` | `iris-2` | e.g. `iris-2`, `prob-4`, `iris-3`. |
| `--tvai-scale` | `4` | Output size is `256*scale`; `1` = no scale. |
| `--tvai-args` | see `--help` | Extra `tvai_up` parameters. |
| `--tvai-workers` | `2` | Parallel TVAI ffmpeg workers. |

## SD 1.5 image restoration

Still images route here automatically; `--restoration-model-name` is video-only.

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--image-restoration-model-name` | `sd-15-jav` | Only current value. |
| `--sd15-steps` | `25` | Diffusion steps. |
| `--sd15-strength` | `0.6` | SDEdit denoise strength, clamped to `<= 0.7`. |
| `--sd15-freeu` / `--no-sd15-freeu` | on | FreeU UNet tweak. |
| `--sd15-seed` | `0` | Base seed. |
| `--sd15-variants` | `1` | Generate N variants with seeds `seed..seed+N-1`; keep the best. |

## VR

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--vr-mode` | `auto` | `auto`, `off`, `sbs`, `sbs-fisheye`. See [VR180](vr180.md). |

## Encoding

| Option | Default | Notes |
| ------ | ------- | ----- |
| `--codec` | `hevc` | `hevc`, `h264`, or `av1` for offline output. HLS streaming always uses H.264. |
| `--encoder-settings` | — | JSON object or comma-separated `key=value` pairs, e.g. `{"cq":22}` or `cq=22,rc-lookahead=32`. See below. |
| `--lut` | — | `.cube` color LUT (1D or 3D) applied on GPU before encoding. Also available in the GUI's Encoding section. |
| `--retarget-high-fps` | off | 60 → 30 FPS (and 59.94 → 29.97) by processing every second frame. Other rates unchanged; audio timing preserved. |
| `--segments` | — | Restore only selected ranges, e.g. `10-25,01:10-01:30.5`. Cannot be combined with `--stream` or `--retarget-high-fps`. See [Segments](segments.md). |
| `--working-directory` | output dir | Where segment temp files are written. See [Segments](segments.md). |

### Choosing a codec

- **`hevc`** (default): best balance of quality and file size, encodes in
  10-bit. Plays on all modern devices and players. Use this unless you have
  a reason not to.
- **`h264`**: maximum compatibility (older TVs, browsers, editing software),
  8-bit only, larger files at the same quality. Also the codec used for
  streaming.
- **`av1`**: best compression — smallest files at the same quality, 10-bit.
  Needs a GPU generation that provides AV1 encoding (NVIDIA RTX 40-series or
  newer) and a reasonably modern player.

With `--segments`, the codec is locked to the input video's codec and
`--codec` does not apply.

### Encoder settings

`--encoder-settings` fine-tunes the hardware encoder. Keys are validated
against the active encoder — an unsupported key fails with a clear error
listing what the encoder accepts. You rarely need more than `cq`:

```bash
# Higher quality (bigger file): lower cq. Default is 25 (HEVC), 24 (H.264), 32 (AV1).
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22"

# Multiple keys
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22,rc-lookahead=32,bf=4"
```

#### NVIDIA (NVENC) keys — all codecs

| Key | What it does |
| --- | ------------ |
| `cq` | Target quality for VBR. **The main quality knob.** Lower = better quality and bigger file. Scale 0–51 for H.264/HEVC (defaults 24/25), 0–63 for AV1 (default 32). |
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
| `spatial_aq` / `spatial-aq` | Spatial adaptive quantization — spends bits where the eye notices. On by default. AV1 accepts only the hyphenated spelling. |
| `temporal-aq` | Temporal adaptive quantization. On by default. |
| `aq-strength` | AQ aggressiveness, 1–15. Default 8. |
| `rc-lookahead` | Frames analyzed ahead for rate control. Default 32. |
| `lookahead_level` | Lookahead quality, 0–3. HEVC/AV1 only — on H.264 it is ignored with a warning (the encoder can't use it). |
| `maxrate` / `bufsize` | Bitrate cap and VBV buffer size, for when you need a hard bitrate limit. |
| `multipass` | Two-pass encoding: `disabled`, `qres`, `fullres`. |
| `weighted_pred` | Weighted prediction. NVENC supports it only together with `bf=0`; otherwise (and always on AV1) it is ignored with a warning. |
| `tf_level` | Temporal filtering level. |

Per-codec extras:

| Codec | Extra keys |
| ----- | ---------- |
| `hevc` | `profile` (`main`, `main10` — default `main10`), `tier` |
| `h264` | `profile` (`baseline`, `main`, `high` — default `high`), `coder` (`cabac`/`cavlc`) |
| `av1` | `tier`, `tile-rows`, `tile-columns` (parallelize decode of large frames) |

#### AMD (AMF) keys — all codecs

| Key | What it does |
| --- | ------------ |
| `cq` | Portable quality knob, automatically translated to AMF's `qvbr_quality_level`. Lower = better. Defaults 24 (H.264), 25 (HEVC), 32 (AV1). |
| `qvbr_quality_level` | The native AMF quality level, if you prefer to set it directly. |
| `usage` | Encoder usage profile. Default `high_quality`. |
| `quality` | Speed/quality preset: `speed`, `balanced`, `quality` (default). |
| `rc` | Rate-control mode. Default `qvbr`. |
| `preset` | AMF preset. |
| `g` | Keyframe interval in frames. Default 250. |
| `bf` | Max consecutive B-frames. |
| `preanalysis` | Pre-analysis pass, enabled by default. |
| `vbaq` | Variance-based adaptive quantization, enabled by default. |
| `maxrate` / `bufsize` | Bitrate cap and VBV buffer size. |
| `profile` / `level` | Codec profile and level. |

Per-codec extras:

| Codec | Extra keys |
| ----- | ---------- |
| `hevc` | `tier`, `bitdepth` (default 10) |
| `h264` | `coder` |
| `av1` | `bitdepth` (default 10) |

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
| `--post-export-action` | `none` | `shutdown` or `command`, run after all exports finish. |
| `--post-export-command` | — | Shell command for `--post-export-action command`. |

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```

## License

| Option | Notes |
| ------ | ----- |
| `--license-email` | Supporter email tied to your key (unlocks unet-4x and SD 1.5). |
| `--license-key` | License key issued for that email. |

The GUI stores these after first entry; the CLI flags exist for scripted use.

## Benchmark

| Option | Notes |
| ------ | ----- |
| `--benchmark` | Run benchmarks instead of processing. |
| `--benchmark-filter` | Only benchmarks whose name contains this string. |
| `--benchmark-video` | Video path for the benchmark; can be repeated. |
