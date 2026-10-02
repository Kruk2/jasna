# Restoring Only Parts of a Video

You don't have to restore a whole video. Pick just the scenes that need it —
the rest of the video is copied through untouched, which is much faster and
keeps the original quality everywhere else. The output is still the
full-length video.

## Segment Editor (GUI)

<img width="822" alt="Segment editor" src="https://github.com/user-attachments/assets/c67939a9-37de-46ae-b722-20c6a0933df7" />

Open it with the scissors button on any queued video.

- Preview the video and select frame-accurate ranges. Leave the selection
  empty to restore the whole video.
- **Restore preview** shows the restored video before you commit to
  processing. BasicVSR++ parts update right away. An
  [LTX](models.md#restoration-model) range has to be rendered first: the
  **Render LTX** button shows how long it takes (often several minutes), and
  Jasna asks before long renders.
- Each range can use its own model (**Restore with**). To compare LTX seeds,
  pick another seed and render again; click a tried seed to see it and keep
  the best with **Use this seed**.
- Use the visible **− / +** controls or the mouse wheel to zoom into details,
  then drag the preview to pan. **Reset view** appears only after the view has
  been changed.
- When ranges are selected, the export keeps the source video codec — the
  main **Encoding** codec setting does not apply, because unselected parts
  are copied as-is. The editor tells you upfront if a video can't be used
  with segment processing.

### Mosaic scanning

The editor can find mosaic scenes for you:

- Scan every frame or at 0.25–2 second intervals. Scanning runs on the GPU
  and reaches about **2,000 FPS on an RTX 5090**.
- After scanning, adjust the confidence slider to update the amber detected
  ranges, then add them to your purple restoration selection with one click.
- With **Show masks in preview** on, each mosaic is tagged with its size
  (longest side as % of the video height).
- **LTX only for big mosaics** (shown after a scan when LTX can run): adds ranges
  with a big mosaic as LTX and the rest as BasicVSR++, so the slow LTX only
  runs where it helps most. A mosaic is big when its longest side is at least
  the slider's share of the video height; 12% is a good start. Check the
  ranges and change any you don't like.

The detection model and confidence are remembered per queued video and used
during final processing, so different videos can use different settings.

### Suggesting better masks

When a detection looks wrong, you can help improve future models. Pause on
the frame, click **Suggest better mask**, and outline each mosaic area — the
editor guides you through drawing. If the mosaic fades out with soft or
blurry edges, include that soft region too.

Submitting uploads the frame and your mask **anonymously**. The data is
encrypted on your machine before upload, and the only attached details are
the app version, the detection model name, and the frame resolution — never
file names, timestamps, or anything identifying you.

## CLI: `--segments`

The CLI counterpart of the editor. Times accept seconds or `HH:MM:SS.s`:

```bash
jasna --input input.mp4 --output output.mp4 --segments "10-25,01:10-01:30.5"
```

Jasna restores the listed ranges and usually copies everything else. A short
transition around the nearest safe video cut points is re-encoded but not
restored — this is required to splice the video back together seamlessly.

What works with segment processing:

- NVIDIA and AMD GPUs.
- One video at a time (no folders, images, or streaming).
- H.264, HEVC, or AV1 input with constant frame rate; MP4, MOV, or MKV output.
- On AMD, some H.264 files are fully re-encoded; only the selected ranges are restored.
- The output codec always matches the input codec.
- With `--restoration-model-name ltx`, every range uses LTX and `--ltx-seed`.
- Cannot be combined with `--retarget-high-fps`.
- Cannot be combined with `--fmp4`; the output is assembled after processing finishes.

Incompatible input is rejected with a clear error before processing starts.

## CLI: `--ltx-auto-split`

The CLI version of **LTX only for big mosaics**. Jasna scans each video
first, then restores big mosaics with LTX and smaller ones with BasicVSR++.
It works on folders too:

```bash
jasna --input videos/ --output restored/ --restoration-model-name ltx --ltx-auto-split 12
```

A split video keeps its own codec. A video segment processing can't handle is
restored whole with LTX. Cannot be combined with `--segments`.

## Working directory

While assembling segment output, Jasna keeps temporary files in a hidden
folder next to the output video. If you want them on another (faster or
bigger) drive, set the **Working directory** in the GUI's Encoding section,
or use the CLI flag:

```bash
jasna --input input.mp4 --output output.mp4 --segments "10-25" --working-directory /fast/scratch
```
