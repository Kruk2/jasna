# Using the GUI

A tour of the app — including the small features that are easy to miss.
Every setting has a tooltip: hover the ⓘ icon next to it.

## The queue

- Add videos or images with the **Add Files** button (the folder icon next
  to it adds a whole folder), or just **drag & drop** them onto the queue.
- **Reorder jobs by dragging** a queue item by its handle — processing runs
  top to bottom.
- Each queue item has its own buttons: the **scissors** opens the
  [Segment Editor](segments.md) to restore only parts of that video, and the
  cross removes it from the queue.
- While processing, each item shows live progress, FPS, and time remaining.
- **Stop** ends the run. The item you stopped goes back to **Pending**, so
  the next start processes it again from the beginning; the half-written
  output file is left on disk for you to delete.
- **Clear Done** removes finished jobs; **Clear** empties the queue.

## Output settings

- **Same as input** writes each result next to its original file. Turn it
  off to pick one output folder for everything.
- The **filename pattern** controls output names — `{original}` stands for
  the input name. If a pattern would overwrite something, the affected queue
  items are highlighted immediately.
- **File conflict** decides what happens when the output file already
  exists: **Auto rename** (default, safe), **Overwrite**, or **Skip**.

## Presets

The bar at the top of the settings panel stores complete setting
combinations. Create a preset for each kind of content you process (for
example "1080p fast" and "4K best quality") and switch with one click. Jasna
remembers your last-used preset across restarts.

## Per-video settings

Detection model and confidence chosen in the [Segment Editor](segments.md)
are remembered **per queued video**, so different videos in one queue can use
different detection settings.

## Restored video player

Select **Video Player** in the header to watch a video as Jasna restores it.
Choose the detection model, confidence, and optional RTX Super Resolution or
unet-4x pass, select **Choose video**, then press the normal play button.
Use Space to play or pause and the Left/Right arrow keys to seek by 30 seconds.
Seek, volume, and full-screen controls remain available during playback.
Full screen expands the video to the largest size that preserves its source
aspect ratio. Move the pointer to the bottom edge to reveal the playback panel;
moving outside that panel hides it again.

**Choose video** is the only file action. Selecting it during playback stops
the current restoration before opening the picker. Changing the detection
model, confidence, or secondary restoration also reloads the restoration
pipeline at the current playback position.

The player does not create an output file. If restoration is slower than the
video, audio pauses while more restored frames are prepared. It is available
only while the queue and restoration preview are idle.

## Interactive image restoration

For still images, **Process images interactively** (in the SD 1.5 section)
opens a side-by-side view: step through your images, try different seeds,
compare the original, mask, and restored result, and save only the variants
you like. Much faster than re-running whole jobs when experimenting.

## System check

On the first launch of each version, Jasna runs a system check (GPU, driver,
memory, install path). You can re-run it any time from the header — useful
after a driver update or when something misbehaves.

## Other bits worth knowing

- **Language**: switch the interface language from the header dropdown
  (restart for the full effect).
- **Logs**: the Logs button in the bottom bar shows live logs with level
  filters and an export button — attach an export when reporting a problem.
- **System stats**: GPU, VRAM, RAM, and CPU usage are shown in the bottom
  bar while processing.
- **License**: enter your supporter key from the header; the chip shows
  whether it's active.
