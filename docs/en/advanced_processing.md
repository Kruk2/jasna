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

## Scene cut detection

Without it, a mosaic tracked across a hard scene cut can end up in one clip
spanning two different shots, and the restorer blends content across the
cut. **Scene Cut Detection** (`--scene-detection`, default on) detects hard
cuts and ends every tracked clip at the boundary, so each clip stays within
a single shot. It runs on the GPU with negligible cost.

Disable with `--no-scene-detection` (or the switch in the GUI's Advanced
section) only if you see clips being split where there is no real cut.

## 60 FPS to 30 FPS export

For 60 (or 59.94) FPS input, **Reduce 60 FPS to 30 FPS**
(`--retarget-high-fps`) processes every second frame and writes 30 (or
29.97) FPS output — half the processing work. Audio timing and playback
speed are preserved. Other frame rates are unchanged:

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

Cannot be combined with [segment processing](segments.md).

## Playable while processing (fragmented MP4)

A normal MP4 can only be opened once the job is finished. **Playable while
processing** (`--fmp4`) lets you play the output while it is still being made —
handy for checking quality without waiting — and the file stays playable if a
job is interrupted:

```bash
jasna --input input.mp4 --output output.mp4 --fmp4
```

The video grows in steps of a few seconds, and players may show the wrong length
until the job ends. Only `.mp4` and `.mov` output is affected. Not available with
streaming or [segment processing](segments.md).

## Color LUT

Apply a `.cube` color LUT (1D or 3D) to the output — for color grading or
matching a house look. Set it in the GUI's Encoding section or with
`--lut path/to/look.cube`. The LUT is applied on the GPU just before
encoding, so it costs almost nothing.

## Sharpening

Restored video can look a little soft. **Sharpening** in the GUI's Encoding
section (`--sharpen`) makes edges and fine detail crisper as the video is
written, so you don't need a second pass through another tool.

```bash
jasna --input in.mp4 --output out.mkv --sharpen 0.5
```

`0` turns it off, `0.2`–`0.5` is a gentle boost, and `1` is the strongest and
can look harsh. A sharper picture needs a bigger file, so if the result looks
worse rather than better, lower the CQ value as well. The effect is not shown
in the preview.

## Custom encoder settings

The **Encoder custom args** field (`--encoder-settings`) fine-tunes the
hardware video encoder — quality level, bitrate caps, keyframe interval, and
more. The main knob is `cq` (lower = better quality, bigger file). Jasna also
limits output size, so nearby CQ values can give the same result when that limit
is reached:

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
