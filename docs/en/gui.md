# Using the GUI

A tour of the app — including the small features that are easy to miss.
Every setting has a tooltip: hover the ⓘ icon next to it.

A loading window shows the installation check before the interface opens.
The first launch after an install or update takes longer. Startup errors stay visible;
use a local NTFS disk on Windows, or an ext4, Btrfs or XFS disk on Linux.
Network disks are unsupported.

## The queue

- Add videos or images with the **Add Files** button (the folder icon next
  to it adds a whole folder), or just **drag & drop** them onto the queue.
- For more than 20 items, use the arrows below the queue to view another page.
  All pages process in order; dragging changes order within the shown page.
- **Reorder jobs by dragging** a queue item by its handle — processing runs
  top to bottom.
- Each video has **scissors** for the [Segment Editor](segments.md) and a
  **play** button that opens it in the restored player, ready but paused. Use
  the **three-dot** menu (or right-click a row) to open its input folder.
  Completed videos reveal their output file when the file manager supports it.
  The menu can also copy the relevant path, open a completed output, or
  requeue a finished, failed, or skipped item at the end of the queue.
  Image rows have the folder menu too; the cross removes any item from the queue.
- While processing, each item shows live progress, FPS, and time remaining.
- Every time you press **Start**, Jasna processes every item still in the
  queue from top to bottom, including finished or failed items. Change your
  settings and press Start again without adding the files again.
- **Stop** ends the run. The interrupted item goes back to **Pending**, and
  the next Start begins the whole queue again. A half-written output file is
  left on disk; the **File conflict** setting below decides what happens to it.
- **Clear Done** removes finished jobs; **Clear** empties the queue.

## Output settings

- **Same as input** writes each result next to its original file. Click its
  button beside Output location to return to this mode after choosing a folder.
- Type a folder path directly, or use the folder button. Either choice turns
  off **Same as input**.
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

In the header player, **Choose video** opens the file picker. Selecting it
during playback stops the current restoration first. Changing the detection
model, confidence, or secondary restoration also reloads the restoration
pipeline at the current playback position. While playing, the status displays
how many seconds of restored video are buffered ahead of the playhead. A buffer
that repeatedly falls to zero means restoration cannot sustain real-time
playback with the current settings.

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

- **Language**: the first start follows your system language. Switch it any
  time from the header dropdown (restart for the full effect).
- **Logs**: the Logs button in the bottom bar shows live logs with level
  filters and an export button (the newest 5000 entries) — attach an export when
  reporting a problem.
- **System stats**: GPU, VRAM, RAM, and CPU usage are shown in the bottom
  bar while processing.
- **Activation**: open **Activate Jasna** in the header to activate online or enter an Activation key manually. The chip shows whether activation is valid on this machine.

## Activate paid models

Open **Activate Jasna** in the header. **Online** is selected by default; both **Online** and **Manual** show a **Machine key** you can copy but cannot edit.

- **Online**: enter the **License code** from your official seller and press **Activate online**.
- **Manual**: copy the Machine key and enter it with your License code on the [activation page](https://jasna-activation.kruk2.workers.dev) from any connected device, or ask your official seller. **Open activation page** opens it in your browser. Paste the returned **Activation key** into Jasna and press **Activate**.

Pressing **Activate online** sends only your License code and Machine key to that service over HTTPS. Jasna checks and saves only the returned Activation key for this machine; it does not save your License code. Later starts and restoration check activation offline, with no periodic requests.

Both choices share **3 registered machines or workspaces per License code**. Retrieving the same Machine key's activation uses no extra slot. Only the owner can release slots and approve replacements; buyers cannot remove registrations. Releasing a slot does not disable keys already issued to older offline versions. A key covers only the model versions you purchased.

For **RunPod or Vast**, select **Cloud workspace** in either choice after your seller enables cloud use for your License code. Keep the Jasna config folder on a persistent volume; on Linux the default is `~/.config/jasna`, or `jasna/` under `XDG_CONFIG_HOME`. Host and GPU changes then reuse the same slot. Copying this folder with its Activation key transfers the workspace identity.

If online activation fails, select **Manual** and use the page on another connected device, or ask your official seller with your License code and Machine key. Access to the service is not guaranteed, including in mainland China; the same 3-slot limit applies.

If an Activation key is rejected, check the Machine key and copy the full key again; for a cloud workspace, restore the config folder from your volume. Earlier activation formats need a replacement from your official seller. License codes are sold only on Buy Me a Coffee and Unifans (kruk2).
