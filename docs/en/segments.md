# Restoring Only Parts of a Video

You don't have to restore the whole video. Choose only the scenes with mosaics.
Jasna copies the rest without changes, so it is much faster and the other
parts keep their original quality. The output is still the full video.
With a [Color LUT](advanced_processing.md#color-lut) or
[Sharpening](advanced_processing.md#sharpening), the rest is encoded again
instead, so they cover the whole video. This takes longer.

## Segment Editor (GUI)

<img width="822" alt="Segment editor" src="https://github.com/user-attachments/assets/c67939a9-37de-46ae-b722-20c6a0933df7" />

Open it with the scissors button on a video in the queue.

- Select the ranges to restore, accurate to the frame. With no ranges, the
  whole video is restored.
- **Restore with** sets the model for each range: BasicVSR++ or
  [LTX](models.md#restoration-model).
- **Restore preview** shows the restored picture before you start the job.
  - BasicVSR++ ranges show it at once.
  - LTX ranges must be rendered first. Click **Render LTX seed N**; the button
    shows how long it takes (often a few minutes). Jasna asks before long
    renders.
  - To compare LTX seeds: choose another seed and render again. Tried seeds
    are listed under **Tried**. Click one to see it, then **Use this seed** to
    keep it.
- Zoom with **− / +** or the mouse wheel, and drag the picture to move it.
  **Reset view** goes back.
- With ranges selected, the output keeps the codec of the source video. The
  **Codec** setting in **Encoding** is not used, because the parts between
  ranges are copied. If another codec is chosen there, its CQ and custom
  encoder settings are not used either. If a video can't be processed in
  parts, the editor tells you at once.
- A BasicVSR++ range and an LTX range need some space between them. If they
  are too close, the editor tells you.

### Mosaic scanning

The editor can find the mosaic scenes for you.

1. Choose a detection model and how often to check: every frame (most
   accurate) or every 0.25–2 seconds (faster). On an RTX 5090 a scan runs at
   about **2,000 FPS**.
2. Click **Scan video**. Found ranges show in amber on the timeline.
3. Move the **Detection threshold** slider to change what counts as found.
4. **Add all ranges** turns the amber ranges into purple ranges, which are
   restored.

- **Show masks in preview** draws the found mosaics in red, with their size
  (longest side as % of the video height).
- **LTX only for big mosaics** appears after a scan if LTX can run on your
  PC. It makes LTX ranges where a mosaic is big and BasicVSR++ ranges for the
  rest, so the slow LTX runs only where it helps most. "Big" means the longest
  side is at least the slider value, as % of the video height. 12% is a good
  start. Check the ranges and change the ones you don't like.

The detection model and detection threshold are saved for this video and used for the
final job. Other videos keep their own settings.

### Suggest a better mask

If a detection is wrong, you can help train better models. Pause on the
frame, click **Suggest better mask** and draw around each mosaic. The editor
shows you how. If the mosaic has soft or blurry edges, include them too.

Jasna uploads the frame and your mask **anonymously**. The data is encrypted
on your PC before upload. Only the app version, detection model name and
frame size are sent with it. No file names, no times, nothing about you.

## CLI: `--segments`

The same as the editor, from the command line. Times are in seconds or
`HH:MM:SS.s`:

```bash
jasna --input input.mp4 --output output.mp4 --segments "10-25,01:10-01:30.5"
```

Jasna restores the listed ranges and copies the rest. A few seconds around
each range are encoded again (but not restored), because the video can only
be cut at some frames.

Limits:

- Works on NVIDIA and AMD.
- One video at a time. No folders, images or streaming.
- Input: H.264, HEVC or AV1 with a constant frame rate. Output: MP4, MOV or
  MKV.
- The output codec is always the input codec.
- On AMD, some H.264 files are encoded again in full. Only the ranges are
  restored.
- With `--restoration-model-name ltx`, every range uses LTX.
- Cannot be used with `--retarget-high-fps` or `--fmp4`.

If the input can't be used, Jasna stops with an error before it starts.

## CLI: `--ltx-auto-split`

The same as **LTX only for big mosaics**, for the command line. Jasna scans
each video first. Then it restores big mosaics with LTX and the rest with
BasicVSR++. It works on folders too:

```bash
jasna --input videos/ --output restored/ --restoration-model-name ltx --ltx-auto-split 12
```

The number is the mosaic size, as % of the video height. The output keeps
the codec of the source video. If that is not `--codec`, `--cq` and
`--encoder-settings` are not used for it. If a video can't be processed in parts, it is
restored whole with LTX. Cannot be used with `--segments` or `--fmp4`.

## Working Directory

While Jasna puts the video together, it keeps temporary files in a hidden
folder next to the output. To use another disk (faster or bigger), set
**Working Directory** in the **Encoding** section, or:

```bash
jasna --input input.mp4 --output output.mp4 --segments "10-25" --working-directory /fast/scratch
```

LTX keeps saved work in `.{output name}.jasna-resume` in the same folder
(default: next to the output; in a Working Directory the name also gets a short code),
including range jobs and auto split. It can take several GB.
It stays after pause, an error, closing Jasna or removing a queue item, and is deleted
after the video finishes successfully. Keep the same Working Directory when continuing.
