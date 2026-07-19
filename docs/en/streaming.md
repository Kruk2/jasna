# Streaming

Streaming lets you watch restored video on the fly, without processing the
whole file first. Seeking is supported.

## Browser player

Streaming mode is CLI-only for now. It opens a player in your browser — pick
a video file and start watching:

```bash
jasna --stream
```

On Windows, streaming uses the same file as the app: `jasna.exe --stream`.

Useful options: `--stream-port` (default `8765`) and `--no-browser` if you
want to open the player yourself.

## Stash integration

Jasna can be used inside [Stash](https://github.com/stashapp/stash) through a
custom Stash fork. Play a scene and Stash launches Jasna automatically,
processing as you watch.

Custom fork:
**[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

Setup:

1. Download the Stash fork from the link above.
2. Set environment variables before starting Stash:
   - `JASNA_CLI_PATH`: full path to `jasna.exe`, unless you renamed it.
   - `JASNA_WORKING_DIR`: full path to the folder containing that executable.
3. **Important:** Before using Stash, run streaming once on a short video
   with the same settings you plan to use in Stash. This prepares the
   GPU-specific detection cache and avoids the first health-check timeout.
4. Start Stash and play a scene.

If Stash logs `timeout waiting for jasna-cli to become healthy`, check
`JASNA_CLI_PATH` first, then precompile as above.
