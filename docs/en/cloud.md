# Cloud GPUs (RunPod, Vast, Docker)

> Cloud licenses are not sold right now. This will change in the future.

You can run Jasna on a rented GPU. Use the Linux NVIDIA version from the
command line inside a supported container. GitHub source checkouts can't use
supporter models.

## What you need

- Internet on the machine: Jasna checks your license online while it uses
  supporter models.
- A normal container. Vast works. A Docker container started with
  `--pid=host` doesn't.
- LTX doesn't work on RunPod or on gVisor containers (for example Modal) for now;
  BasicVSR++ does. For LTX use Vast.
- An NVIDIA GPU with driver **580 or newer**.
- At least **80 GB** of disk.
- `NVIDIA_DRIVER_CAPABILITIES=all` in your template or instance settings.
  Jasna can't read or save videos without it. On Vast, add
  `-e NVIDIA_DRIVER_CAPABILITIES=all` to the Docker options.
- Log in as `root`. RunPod and Vast do this by default.

## Install

Paste these lines into the terminal of your rented machine. If you use a
newer Jasna, change `0.11.1` to your version.

```bash
apt-get update && apt-get install -y zstd wget libgl1 libglib2.0-0

cd /root
for part in 000 001 002 003; do wget https://github.com/Kruk2/jasna/releases/download/v0.11.1/jasna-linux-nvidia-0.11.1.tar.zst.part$part; done

cat jasna-linux-nvidia-0.11.1.tar.zst.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-0.11.1
```

Unpack in `/root` like above, as the same user that runs Jasna. Keep Jasna on
the machine's own disk (ext4, Btrfs, XFS, ZFS or the container's own
filesystem). Jasna won't start from `/workspace` or a network drive, and LTX
may refuse to load from a network share or network volume.

## Check the video encoder

Some rented machines have a GPU whose video encoder doesn't work. Check it
before you activate; it takes a second:

```bash
./tools/ffmpeg -loglevel error -f lavfi -i testsrc2=duration=1:size=1280x720 -c:v hevc_nvenc -f null - && echo OK
```

No `OK`? Delete the machine and rent another one. Jasna can't save videos
there.

## Activate

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --license-code YOUR-LICENSE-CODE
```

After `installation verified` it can look stuck for up to a minute. Just
wait, don't press Ctrl+C.

RunPod's `/workspace` can ignore private file permissions. Keep activation in
`/root/jasna-config` as above. On other providers, you can use a persistent
folder that preserves private permissions.

Run the `export` line every time before starting Jasna. A rented GPU counts as
one of your 3 devices only while Jasna runs on it. Keep `jasna-config` to skip
entering the code next time; if you lose it, just activate again.

## Restore a video

```bash
export XDG_CONFIG_HOME=/root/jasna-config
./jasna --input /root/video.mp4 --output /root/video-restored.mp4 --restoration-model-name ltx
```

The first time, Jasna asks to download the LTX model (15 GB). Type `y`.

On an RTX 50-series GPU, add `--ltx-fast` to restore about 1.4× faster
(it downloads about 8 GB more).

To send a video up and get the result back, use your provider's file
manager, or run this on your PC:

```bash
scp -P PORT video.mp4 root@HOST:/root/
scp -P PORT root@HOST:/root/video-restored.mp4 .
```

## Problems

| You see | Do this |
| ------- | ------- |
| `unsigned or modified Jasna runtime` | Unpack again in `/root` as `root`. |
| `libGL.so.1: cannot open shared object file` | Run the `apt-get` line from Install. |
| `Failed to open hevc encoder` | This machine's video encoder is broken. Rent another one. |
| `in use on three devices` | Close Jasna on another device, then try again. |
| `could not reach the license service` or `lost the connection` | Check the machine's internet, then start again. |
| `No activation found` or `Activate Jasna with your License code` | Run the `export` line, then start Jasna again, or activate again. |
| `unsafe Jasna launch environment` | Start it with `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` |

`Bit depth 10 ... not supported` is OK. Jasna reads that video a bit
slower, that's all.

Don't share your `jasna-config` folder. Anyone who has it can use your
activation.
