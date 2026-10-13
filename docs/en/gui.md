# Using the GUI

Every setting has a tooltip. Move the mouse over the ⓘ icon next to it.

The first start after an install or update takes longer. A loading window shows
the installation check before the main window opens. If the check fails, the
error stays on screen.

Install Jasna on a local disk: NTFS on Windows, ext4, Btrfs or XFS on Linux.
Network disks are not supported.

## The queue

- Add videos or images with **Add Files**, or with the folder icon next to it
  for a whole folder. You can also drag and drop files onto the queue.
- Drag an item by its handle to change the order. Jasna processes the queue
  from top to bottom.
- With more than 20 items, use the arrows below the queue to change the page.
  All pages are processed in order. Dragging works only inside one page.
- Buttons on each video:
  - **Scissors** opens the [Segment Editor](segments.md).
  - **Play** opens the video in the Video Player, paused.
  - **Three dots** (or right-click) opens the input folder, copies a path,
    opens the finished output, or puts a finished, failed or skipped item back
    at the end of the queue.
  - **Cross** removes the item.
- During processing, each item shows progress, FPS and time left.
- **Start** always processes everything in the queue, also items that are
  already finished or failed. So you can change settings and press Start
  again without adding the files again.
- **Stop** ends the run. The current item goes back to **Pending**. The half
  written output file stays on disk; **File conflict** decides what happens
  with it next time.
- **Clear Done** removes finished items. **Clear** removes everything.

## Output settings

- **Same as input** saves each result next to its original file.
- Type a folder path or use the folder button to save somewhere else. This
  turns off **Same as input**. Click **Same as input** to turn it back on.
- **Filename pattern** sets the output name. `{original}` is the input file
  name. If a name would overwrite another file, the queue items are marked at
  once.
- **File conflict** decides what to do when the output file already exists:
  **Auto rename** (default), **Overwrite** or **Skip**.

## Presets

The bar at the top of the settings saves all settings under one name. Make one
preset for each type of video, for example "1080p fast" and "4K best quality",
and switch with one click. Jasna remembers the last preset after a restart.

## Per-video settings

The detection model and confidence you set in the [Segment Editor](segments.md)
are saved for that one video. Other videos in the queue keep their own.

## Video Player

Click **Video Player** in the header to watch a video while Jasna restores it.
No output file is created.

1. Choose the detection model, confidence and, if you want, RTX Super
   Resolution or unet-4x.
2. Click **Choose video**.
3. Press play.

- **Space**: play or pause. **Left/Right**: jump 30 seconds.
- In full screen, move the mouse to the bottom edge to show the controls.
- If you change the model, confidence or secondary restoration, playback
  restarts from the same position.
- The status shows how many seconds are restored ahead. If it drops to zero
  again and again, your GPU is too slow for real time with these settings.
  Audio waits until more frames are ready.
- The player always uses BasicVSR++, not LTX.
- It works only when the queue and the restore preview are not running.

## Images, one by one

For still images, **Process images interactively** (in the **Image Restoration** section)
opens a side-by-side view. Go through your images, try other seeds, compare
original, mask and result, and save only the results you like. This is much
faster than running the whole job again for each try.

## System check

On the first start of each version, Jasna checks the GPU, driver, memory and
install path. You can run it again from the header, for example after a driver
update.

## Other settings

- **Language**: the first start uses your system language. Change it in the
  header. Restart Jasna to change all texts.
- **Logs**: the Logs button in the bottom bar shows live logs. **Export** saves
  the last 5000 lines. Please attach it when you report a problem.
- **System stats**: the bottom bar shows GPU, VRAM, RAM and CPU use during
  processing.

## Activate paid models

Paid models (unet-4x, SD 1.5, LTX) need activation. Open **Activate Jasna** in
the header.

- **Online** (default): enter the **License code** from your seller and click
  **Activate online**.
- **Manual**, for a PC without internet or when online fails:
  1. Copy the **Machine key**.
  2. On any device with internet, open the
     [activation page](https://jasna-activation.kruk2.workers.dev) (or click
     **Open activation page**) and enter the Machine key and your License code.
     You can also send both to your seller.
  3. Paste the **Activation key** you get back into Jasna and click
     **Activate**.

Good to know:

- One License code works on **3 PCs or cloud workspaces**. Activating the same
  PC again does not use a new slot. Only the owner can free a slot; you can't
  remove a PC yourself.
- An Activation key covers only the models you bought.
- **Activate online** sends only your License code and Machine key, over HTTPS.
  Jasna saves only the Activation key, not your License code. After that,
  Jasna checks activation offline and does not contact the server again.
- If the main server can't be reached, Jasna tries a
  [server in Hong Kong](https://47.238.135.110). With Chinese language it
  tries Hong Kong first. Both use the same 3 slots. Access is not guaranteed,
  for example in mainland China.
- Key rejected? Check the Machine key and copy the whole Activation key again.
  Keys made for older Jasna versions need a new key from your seller.
- Rented GPUs (RunPod, Vast) can be activated only from the command line. See
  [Cloud GPUs](cloud.md).

License codes are sold only on
[Buy Me a Coffee](https://buymeacoffee.com/kruk2),
[Unifans](https://app.unifans.io/c/kruk2) and, for China, the seller in the
[Chinese README](../../README.zh.md#在闲鱼购买). Prices: see
[Supporting the project](../../README.md#supporting-the-project).
