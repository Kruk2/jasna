# Using the GUI

Every setting has a tooltip. Move the mouse over the ⓘ icon next to it.

The first start after an install or update takes longer. A loading window shows
the installation check before the main window opens. If the check fails, the
error stays on screen.

Install Jasna on a local disk (NTFS on Windows; ext4, Btrfs, XFS or ZFS on Linux),
in a folder whose path has no Chinese, Japanese, accented or other
non-English characters. Network disks
are not supported. Use an SSD: on a hard disk (HDD) Jasna starts and loads
models slowly, and it warns you at startup.

## The queue

- Add videos or images with **Add files**, or with the folder icon next to it
  for a whole folder. You can also drag and drop files onto the queue.
- Drag an item by its handle to change the order. Jasna processes the queue
  from top to bottom.
- With more than 20 items, use the arrows below the queue to change the page.
  All pages are processed in order. Dragging works only inside one page.
- Buttons on each video:
  - **Scissors** opens the [Segment Editor](segments.md).
  - **Play** opens the video in the Video Player, paused.
  - **Three dots** (or right-click): **Open containing folder**, **Copy path**,
    **Open restored output**, and **Requeue**, which puts a finished, failed or
    skipped item back at the end of the queue.
  - **Cross** removes the item.
- During processing, each item shows progress, FPS and time left.
- **▶** (start) continues a paused queue from its pending items. Finished items
  stay finished. After the queue finishes, press ▶ to process everything again
  with your current settings.
- **Ⅱ** (pause) appears for jobs using LTX or ranges. It asks first, then keeps saved work
  and puts the current item back to **Pending**. Press **▶** to continue.
  If the input, settings or Jasna version changed, Jasna asks before replacing saved work;
  **No** keeps it and skips that video.
- **■** (stop) is used for other jobs. They start over next time;
  **File Conflict** decides what happens to any unfinished output.
- Closing the window while a video is processing asks first: **Minimize** keeps
  it running, **Stop and quit** ends it.
- **Clear done** removes finished items. **Clear** removes everything.

## Output settings

- **Same as input** saves each result next to its original file.
- Type a folder path or use the folder button to save somewhere else. This
  turns off **Same as input**. Click **Same as input** to turn it back on.
- **Output pattern** sets the output name. `{original}` is the input file
  name. If a name would overwrite another file, the queue items are marked at
  once.
- **File Conflict** (in the **Output** tab) decides what to do when the
  output file already exists: **Auto-Rename** (default), **Overwrite** or
  **Skip**.

## Simple, Advanced and Expert

The switch under the preset bar picks how much you see, in the same tabs.
**Simple** shows the main choices: restoration model, file conflict, codec,
quality and what to do when the queue is done. **Advanced** shows more
settings and adds the Detection and Upscale tabs. **Expert** adds speed and
troubleshooting options most people never need. Settings you can't see still
apply, and Jasna remembers your choice.

## Presets

The bar at the top of the settings saves all settings under one name. Make one
preset for each type of video, for example "1080p fast" and "4K best quality",
and switch with one click. Jasna remembers the last preset after a restart.

## Per-video settings

The detection model and detection threshold you set in the [Segment Editor](segments.md)
are saved for that one video. Other videos in the queue keep their own.

## Video Player

Click **Video Player** in the header to watch a video while Jasna restores it.
No output file is created.

1. Choose the detection model, detection threshold and, if you want, a
   secondary restoration (**RTX Super Res** or **UNet 4x**).
2. Click **Choose video**.
3. Press play.

- **Space**: play or pause. **Left/Right**: jump 30 seconds.
- In full screen, move the mouse to the bottom edge to show the controls.
- If you change the model, detection threshold or secondary restoration, playback
  restarts from the same position.
- The status shows how many seconds are restored ahead. If it drops to zero
  again and again, your GPU is too slow for real time with these settings.
  Audio waits until more frames are ready.
- The player always uses BasicVSR++, not LTX.
- It works only when the queue and the **Restore preview** are not running.

## Images, one by one

For still images, **Process images interactively** (in the **Image Restoration (still images)** section)
opens a side-by-side view. Go through your images, try other seeds, compare
original, mask and result, and save only the results you like. This is much
faster than running the whole job again for each try.

## System check

On the first start of each version, Jasna checks FFprobe, the GPU, the driver
and the install path. You can run it again from the header, for example after a driver
update.

On AMD it also checks the AMD driver and AMD's AMF video runtime, which Jasna
needs to read and write videos. If the driver is older than tested or its
version can't be read, you get a warning and can still continue. If videos then
fail, install the driver and AMF from [Quick start](../../README.md#quick-start).

## Other settings

- **Language**: the first start uses your system language. Change it in the
  header. Restart Jasna to change all texts.
- **Logs**: the Logs button in the bottom bar shows live logs. **Export** saves
  the last 5000 lines. Please attach it when you report a problem.
- **System stats**: the bottom bar shows GPU, VRAM, RAM and CPU use during
  processing.
- **Tray**: click the gear in the header and turn on **Minimize to tray** or
  **Close to tray** to keep Jasna running without a taskbar button. Click the
  tray icon to open it again; on Windows, right-click it to quit. On Linux it
  needs a tray that shows app icons (KDE, Ubuntu); **Close to tray** is
  Windows only.

## Activate supporter models

Supporter models (UNet 4x, SD 1.5, LTX) need activation. Open **Activate Jasna** in
the header.

Enter your **License code** and click **Activate**. Jasna picks how it checks
your license by itself:

- **PC with a TPM 2.0 security chip** (most Windows 11 PCs): Jasna works
  offline from then on. On Windows, a permission prompt may appear once; this
  is a one-time license check for offline use. If you decline it, Jasna checks
  online instead; enter your License code again to switch to offline.
- **Other PCs, rented GPUs and Docker**: Jasna checks your license online while
  it uses supporter models, so keep the PC connected. If the connection stays
  down, the queue stops. Start it again once you are back online.

Good to know:

- The header shows **Active (offline)** or **Active (online)**.
- One License code works on **3 devices at the same time**. Activating the
  same PC again does not use a new slot.
- Reopen **Activate Jasna** to see **Devices in use**: offline PCs plus
  devices running supporter models online right now. Jasna checks this online.
- Offline PCs keep their slot. Reinstalling the system or changing the GPU,
  disk or RAM keeps it; a new motherboard or a security chip reset in the BIOS
  counts as a new PC. To free a slot, ask where you got your License code.
- Online devices count only while Jasna runs, and each open Jasna counts.
  "In use on 3 devices"? Close Jasna on another device and start again. If
  Jasna crashed or was force-closed, wait up to 15 minutes.
- If Jasna works online on a PC with a security chip, the activation message
  says why.
- Linux with a security chip: Jasna asks you to run
  `sudo usermod -aG tss $USER`. Restart the PC and activate again to work
  offline. On NixOS, first add `security.tpm2.enable = true;` and put your
  user in the `tss` group in `configuration.nix`.
- Linux in distrobox or toolbox: the container must use the host's TPM group.
  Inside it, `ls -ln /dev/tpmrm0` shows the group number; if `id` does not list
  that number, create the container again with
  `--additional-flags "--group-add <number>"`, then activate again.
- "No maker certificate" on an AMD PC: update the BIOS. Older AMD security chip
  firmware often has no certificate. If that does not help, Jasna works online.
- Moved from an older Jasna? Your activation carries over on the first job. If
  it says the activation moved, enter your License code again. Jasna 0.11 keeps
  working on the same PC.
- Have an email and key from Jasna 0.10? That key no longer works. Get a new
  License code from where you bought it.

License codes come only from
[Buy Me a Coffee](https://buymeacoffee.com/kruk2),
[Unifans](https://app.unifans.io/c/kruk2) and, for China, the Xianyu seller in the
[Chinese README](../../README.zh.md#在闲鱼购买). Support tiers: see
[Supporting the project](../../README.md#supporting-the-project).
