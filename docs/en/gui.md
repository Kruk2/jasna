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
- **Ⅱ** (pause) appears for jobs using LTX. It asks first, then keeps saved work
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
- **File Conflict** (in **Basic Processing**) decides what to do when the
  output file already exists: **Auto-Rename** (default), **Overwrite** or
  **Skip**.

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

On AMD it also checks AMD's AMF video runtime, which Jasna needs to read and
write videos. On Windows it comes with the AMD driver. On Linux, install it once.
We tested [AMF 26.20](https://repo.radeon.com/amf/26.20/):

```bash
curl -fLO https://repo.radeon.com/amf/26.20/amf_installer_26.20.zip
unzip amf_installer_26.20.zip
sudo bash amf_installer_26.20.sh
```

Accept AMD's license when prompted, then restart Jasna. If using Distrobox,
install AMF inside the same container as Jasna. Compiling FFmpeg with
`--enable-amf` does not install this runtime.

## Other settings

- **Language**: the first start uses your system language. Change it in the
  header. Restart Jasna to change all texts.
- **Logs**: the Logs button in the bottom bar shows live logs. **Export** saves
  the last 5000 lines. Please attach it when you report a problem.
- **System stats**: the bottom bar shows GPU, VRAM, RAM and CPU use during
  processing.

## Activate supporter models

Supporter models (UNet 4x, SD 1.5, LTX) need activation. Open **Activate Jasna** in
the header.

If Jasna is already activated, opening this window checks online and shows
**Devices used**, for example **2 of 3**. If the count is unavailable, reopen
the window to try again. Restoration still works offline.

- **Online** (default): enter your **License code** and click
  **Activate online**.
- **Manual**, for a PC without internet or when online fails:
  1. Copy the **Machine key**.
  2. On any device with internet, open the
     [activation page](https://jasna-activation.kruk2.workers.dev) (or click
     **Open activation page**) and enter the Machine key and your License code.
     You can also send both to where you got your License code.
  3. Paste the **Activation key** you get back into Jasna and click
     **Activate**.

Good to know:

- One License code works on **3 PCs or cloud workspaces**. Activating the same
  PC again does not use a new slot.
- A new motherboard (Windows) or a fresh Linux install counts as a new PC
  and uses a slot. Reinstalling Windows or changing the GPU, disk or RAM
  does not: just activate again. On a few Windows PCs without a real
  motherboard ID, a new system disk also counts as a new PC.
- An Activation key covers only the models you bought.
- **Activate online** sends only your License code and Machine key. After
  that, Jasna works offline.
- Key rejected? Check the Machine key and copy the whole Activation key again.
  Keys made for older Jasna versions need a new key from where you got your License code.
- Have an email and key from Jasna 0.10? That key no longer works. Get a new
  License code from where you bought it.

License codes come only from
[Buy Me a Coffee](https://buymeacoffee.com/kruk2),
[Unifans](https://app.unifans.io/c/kruk2) and, for China, the Xianyu seller in the
[Chinese README](../../README.zh.md#在闲鱼购买). Support tiers: see
[Supporting the project](../../README.md#supporting-the-project).
