# Advanced Processing

Optional features for special cases. Everything here works in both the GUI
(look for the matching setting, each has a tooltip) and the CLI.

## Denoising

Restored regions can carry noise artifacts. The Denoising setting
(`--denoise low|medium|high`) applies gentle spatial denoising to the
restored regions only — the rest of the frame is untouched. Start with `low`
and raise it only if artifacts remain.

By default it runs before secondary restoration;
`--denoise-step after_secondary` moves it right before blending.

## Detection stability filtering

Detection is not perfect frame-to-frame: a mosaic can vanish for a frame or
two (cutting one clip into several, with a visible seam and an unrestored
frame), and a single-frame false detection triggers a needless restore.

- **Max Detection Gap** (`--max-detection-gap`, default `2`) fills dropouts
  up to N frames when the mosaic reappears at the same spot, keeping the
  clip continuous.
- **Min Detection Duration** (`--min-detection-duration`, default `2`) drops
  detections shorter than N frames as false positives; those frames stay
  unrestored.

Keep both small so genuine fast appear/disappear moments are unaffected.
`0` disables either.

## 60 FPS to 30 FPS export

For 60 (or 59.94) FPS input, **Reduce 60 FPS to 30 FPS**
(`--retarget-high-fps`) processes every second frame and writes 30 (or
29.97) FPS output — half the processing work. Audio timing and playback
speed are preserved. Other frame rates are unchanged:

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

Cannot be combined with [segment processing](segments.md).

## Color LUT

Apply a `.cube` color LUT (1D or 3D) to the output — for color grading or
matching a house look. Set it in the GUI's Encoding section or with
`--lut path/to/look.cube`. The LUT is applied on the GPU just before
encoding, so it costs almost nothing.

## Custom encoder settings

The **Encoder custom args** field (`--encoder-settings`) fine-tunes the
hardware video encoder — quality level, bitrate caps, keyframe interval, and
more. The main knob is `cq` (lower = better quality, bigger file):

```bash
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22"
```

Every accepted key for every codec is documented in the
[CLI reference](cli.md#encoding).

## Post-export actions

Run something when the whole queue finishes: **Shutdown PC** or a **custom
command** (for example, a notification script). Set it in the GUI's
Post-export section or via CLI:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```
