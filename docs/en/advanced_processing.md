# Advanced Processing

Extra features for special cases. Everything here works in the GUI (each
setting has a tooltip) and in the CLI.

## Denoising

Restored areas can have some noise. **Denoise Strength**
(`--denoise low|medium|high`) cleans only the restored areas. The rest of the
frame is not changed. Start with `low`, and go higher only if the noise is
still visible.

By default it runs before secondary restoration.
`--denoise-step after_secondary` moves it to the end, just before the
restored area goes back into the frame.

Not used with LTX.

## Detection stability

Detection is not perfect from frame to frame. A mosaic can disappear for one
or two frames. Then the clip is cut in pieces, with a visible seam and one
frame not restored. A false detection on a single frame starts a restore that
is not needed.

- **Max Detection Gap** (`--max-detection-gap`, default `2`): fills gaps up to
  N frames when the mosaic comes back at the same place.
- **Min Detection Duration** (`--min-detection-duration`, default `2`): drops
  detections shorter than N frames. Those frames stay as they are.

Keep both small, so real short mosaics are not affected. `0` turns either one
off.

## Scene cut detection

Without it, a mosaic can be tracked across a hard cut, and the restorer mixes
two different shots. **Scene Cut Detection** (`--scene-detection`, on by
default) finds hard cuts and ends every clip there. It runs on the GPU and
costs almost nothing.

Turn it off (`--no-scene-detection`, or the switch in **Advanced Processing**) only if
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
**Color LUT** in the **Encoding** section, or use `--lut path/to/look.cube`.
It is applied as the video is written, so it is almost free.

## Sharpening

Restored video can look a bit soft. **Sharpening** in the **Encoding** section
(`--sharpen`) makes edges and small details sharper while the video is saved.
No second tool is needed.

```bash
jasna --input in.mp4 --output out.mkv --sharpen 0.5
```

`0` is off, `0.2`–`0.5` is a small boost, `1` is the strongest and can look
too hard. A sharper picture needs more bits. If the result looks worse, also
lower CQ. The preview does not show sharpening.

## Encoder quality and custom settings

**CQ** (`--cq`) sets the quality. Lower means better quality and a bigger
file. Jasna also limits the output size, and when that limit is hit,
close CQ values can give the same file.

```bash
jasna --input in.mp4 --output out.mkv --cq 22
```

**Custom Args** (`--encoder-settings`) is for other encoder options,
for example bitrate limits or keyframe interval. CQ can't be set there in the
GUI; use the CQ control. All keys, ranges and defaults are in the
[CLI reference](cli.md#encoding).

## After the queue is done

Run something when the whole queue is finished: **Shutdown PC** or a
**Custom Command** (for example a script that sends you a message). Set it in
the **Post-Export Action** section or in the CLI:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```

You can change or turn off this action while jobs are running. Before
**Shutdown PC**, the GUI counts down 60 seconds. Press **Cancel** to keep the
PC on. The CLI shuts down right away.

To run a command after each finished video, use **Command After Each Video**
or `--post-export-video-command`. Jasna waits until it ends. If the command
fails, that video is marked as failed. The command runs in the output folder.

Placeholders: `{input}`, `{output}`, `{output_dir}`, `{output_stem}`,
`{output_suffix}`. They are already quoted. There is an example in the
[CLI reference](cli.md#post-export).
