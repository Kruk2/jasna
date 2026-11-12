# Advanced Processing

Extra features for special cases. Everything here works in the GUI (each
setting has a tooltip) and in the CLI.

## Denoising

Restored areas can have some noise. **Denoise Strength**
(`--denoise low|medium|high`) cleans only the restored areas. The rest of the
frame is not changed. Start with `low`, and go higher only if the noise is
still visible.

Not used with LTX.

## Detection stability

Detection is not perfect from frame to frame. A mosaic can disappear for one
or two frames. Then the clip is cut in pieces, with a visible seam and one
frame not restored. A false detection on a single frame starts a restore that
is not needed.

- **Max Detection Gap** (`--max-detection-gap`, default `2`): fills gaps up to
  N frames when the mosaic comes back at the same place. LTX always fills gaps
  of up to 3 frames; values above 3 apply to LTX too.
- **Min Detection Duration** (`--min-detection-duration`, default `2`): drops
  detections shorter than N frames. Those frames stay as they are. Works the
  same with LTX.

Keep both small, so real short mosaics are not affected. `0` turns either one
off.

## Scene cut detection

Without it, a mosaic can be tracked across a hard cut, and the restorer mixes
two different shots. **Scene Cut Detection** (`--scene-detection`, on by
default) finds hard cuts and ends every clip there. It runs on the GPU and
costs almost nothing.

Turn it off (`--no-scene-detection`, or the switch in the **Detection** tab, **Expert** view) only if
clips are split where there is no real cut.

## 60 FPS to 30 FPS

For 60 (or 59.94) FPS video, **Reduce 60 FPS to 30 FPS**
(`--retarget-high-fps`) processes every second frame and saves 30 (or 29.97)
FPS. That is half the work. Audio and speed stay correct. Other frame rates
are not changed.

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

Cannot be used with [selected ranges](segments.md) or streaming.

## Playable While Processing (fMP4)

A normal MP4 opens only after the job is done. With **Playable While
Processing** (`--fmp4`) you can play the output while it is still being made,
for example to check quality early. If the job stops in the middle, the file
still plays.

```bash
jasna --input input.mp4 --output output.mp4 --fmp4
```

The video grows a few seconds at a time, and players can show the wrong
length until the job ends. Works only for `.mp4` and `.mov` output. Not
available with streaming or [selected ranges](segments.md).

## Color LUT

Apply a `.cube` color LUT (1D or 3D) to the output, for color grading. Set
**Color LUT** in the **Output** tab, or use `--lut path/to/look.cube`.
It is applied as the video is written, so it is almost free. With
[selected ranges](segments.md), the parts between them are encoded again
instead of copied, so the LUT covers the whole video.

## Sharpening

Restored video can look a bit soft. **Sharpening** in the **Output** tab
(`--sharpen`) makes edges and small details sharper while the video is saved.
No second tool is needed.

```bash
jasna --input in.mp4 --output out.mkv --sharpen 0.5
```

`0` is off, `0.2`–`0.5` is a small boost, `1` is the strongest and can look
too hard. A sharper picture needs more bits. If the result looks worse, also
lower CQ. The preview does not show sharpening. With
[selected ranges](segments.md), the parts between them are encoded again
instead of copied, so sharpening covers the whole video.

## Upscale the whole video

**Upscale Video** in the **Upscale** tab (`--upscale`) makes the whole saved
video bigger and sharper, not only the restored areas. Pick the output height
(`--upscale-height`: 1080p, 1440p or 4K); the width keeps the picture's shape.
Videos that are already this tall are not upscaled.

- **SPAN** (`span`): a fast AI upscaler. NVIDIA and AMD. On NVIDIA it is
  prepared once for each new video size, which takes about half a minute.
- **RealPLKSR** (`realplksr`): a slower AI upscaler that keeps more fine detail
  and cleans compression. NVIDIA and AMD. About 5 FPS from 1080p to 4K on an
  RTX 5090, so a long video takes a while; preparing it takes about a minute.
- **RTX Super Res** (`rtx-super-res`): NVIDIA only, very fast. It can also
  remove noise or blur first (`--upscale-rtx-denoise`, `--upscale-rtx-deblur`).
- **DLSS 5** (`dlss5`, command line only, Windows with an RTX 30-series or newer
  GPU; RTX 30 is slow): NVIDIA DLSS 5 also reworks lighting and surfaces, so
  colors and skin can change. Jasna doesn't include it: the first time, Jasna
  offers to download the unofficial DLSS 5 Visual Enhancer v3.0 (467 MB) from
  GitHub and keeps its runtime in `model_weights/dlss5`. `--upscale-dlss5-preset` picks the upscaling model
  (`l` and `m` rebuild more texture), `--upscale-dlss5-style` the look and
  `--upscale-dlss5-intensity` (0–2) how strong the rework is.

```bash
jasna --input in.mp4 --output out.mp4 --restoration-model-name ltx --upscale span --upscale-height 2160
```

**Strength** (`--upscale-strength`, 0–1) mixes in a plain resize for a softer,
more natural look. 4K needs a bigger file, so Jasna raises its bitrate limit
to match. Works with BasicVSR++ and LTX. The whole video is restored, so it is
not available with [selected ranges](segments.md), and **LTX only for big
mosaics** is skipped. Not available with streaming. The preview does not show it.

## Encoder quality and custom settings

**CQ** (`--cq`) sets the quality. Lower means better quality and a bigger
file. Jasna also limits the output size, and when that limit is hit,
close CQ values can give the same file.

```bash
jasna --input in.mp4 --output out.mkv --cq 22
```

**Custom Args** (**Expert** view, `--encoder-settings`) is for other encoder options,
for example bitrate limits or keyframe interval. CQ can't be set there in the
GUI; use the CQ control. All keys, ranges and defaults are in the
[CLI reference](cli.md#encoding).

## Settings saved in the video

Jasna writes the settings it used into the output's **Comment** tag: the
model, the LTX seed, the detection and encoder settings, and so on. It is one
line of text. You can see it in MediaInfo, in VLC (**Tools → Media Information**)
or with:

```bash
ffprobe -v error -show_entries format_tags=comment out.mkv
```

A comment the source video had is kept in that line as `source_comment`.

## After the queue is done

Run something when the whole queue is finished: **Shutdown PC** or a
**Custom Command** (for example a script that sends you a message, **Expert**
view only). Set it with **When Queue Is Done** in the **Output** tab or in the CLI:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```

You can change or turn off this action while jobs are running. Before
**Shutdown PC**, the GUI counts down 60 seconds. Press **Cancel** to keep the
PC on. The CLI shuts down right away.

To run a command after each finished video, use **Command After Each Video**
(**Expert** view) or `--post-export-video-command`. Jasna waits until it ends. If the command
fails, that video is marked as failed. The command runs in the output folder.

Placeholders: `{input}`, `{output}`, `{output_dir}`, `{output_stem}`,
`{output_suffix}`. They are already quoted. There is an example in the
[CLI reference](cli.md#post-export).

In the GUI, `{log}` is a file with that video's log, saved next to the output
as `<output name>.log`. Copy or move it wherever you want, for example
`copy {log} D:\logs`.
