# Streaming

Streaming lets you watch a restored video while Jasna is still restoring it.
You don't wait for the whole file. Seeking works.

## Browser player

Start streaming from the command line. Jasna opens a player in your browser.
Choose a video file there and play it.

```bash
jasna --stream
```

On Windows use the same file as the app: `jasna.exe --stream`.

Useful options: `--stream-port` (default `8765`) and `--no-browser` if you want
to open the player yourself.

Streaming always uses BasicVSR++ (not LTX) and H.264. To watch inside the app
instead, use the [Video Player](gui.md#video-player).

## Stash

You can use Jasna inside [Stash](https://github.com/stashapp/stash) with a
custom Stash version. When you play a scene, Stash starts Jasna and the video
is restored while you watch.

Download: **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

Setup:

1. Download this Stash version.
2. Before you start Stash, set these environment variables:
   - `JASNA_CLI_PATH`: full path to `jasna.exe`.
   - `JASNA_WORKING_DIR`: full path to the folder with `jasna.exe`.
3. **Important:** run streaming once on a short video, with the same settings
   you will use in Stash. The first run prepares files for your GPU and takes
   long. If you skip this, Stash gives up waiting.
4. Start Stash and play a scene.

If Stash logs `timeout waiting for jasna-cli to become healthy`, check
`JASNA_CLI_PATH` first. Then do step 3.
