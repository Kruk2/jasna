# Cloud GPUs (RunPod, Vast, Docker)

You can run Jasna on a rented GPU with the Linux NVIDIA version, from the
command line. Your License code works there. In a container or virtual machine
Jasna always works online, so the machine counts as one of your 3 devices only
while Jasna runs, and it needs internet while you use supporter models. GitHub source checkouts can't use supporter models.

## What you need

- An NVIDIA GPU with driver **580 or newer**, and at least **80 GB** of disk.
  Rent an RTX card (for example RTX 4090, RTX 5090, RTX A6000 or L40S).
  Data-center GPUs like A100, H100 and B200 don't work.
- `NVIDIA_DRIVER_CAPABILITIES=all` in your template or instance settings, or
  Jasna can't read or save videos. On Vast, add
  `-e NVIDIA_DRIVER_CAPABILITIES=all` to the Docker options.
- Log in as `root`. RunPod and Vast do this by default.
- For LTX, use Vast. LTX doesn't work on RunPod or Modal for now; BasicVSR++
  does.
- Docker: don't start the container with `--pid=host`.

## Install

Paste these lines into the terminal of your rented machine. They download the
newest Jasna into `/root`:

```bash
apt-get update && apt-get install -y curl zstd wget libgl1 libglib2.0-0

cd /root
curl -s https://api.github.com/repos/Kruk2/jasna/releases/latest | grep -o 'https://[^"]*jasna-linux-nvidia-[^"]*part[0-9]*' | sort -u | xargs -n1 wget
cat jasna-linux-nvidia-*.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-*/
```

Keep Jasna on the machine's own disk. It won't start from `/workspace`, a
network drive or a network volume.

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

Run the `export` line every time before starting Jasna. If `jasna-config` is
gone (for example on a new machine), just activate again. Don't put it in
RunPod's `/workspace`, and don't share it: anyone who has it can use your
activation.

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
| `in use on three devices` | Close Jasna on another device, or wait 15 minutes if you just closed it, then try again. |
| `could not reach the license service` or `lost the connection` | Check the machine's internet, then start again. |
| `No activation found` or `Activate Jasna with your License code` | Run the `export` line, then start Jasna again, or activate again. |
| `unsafe Jasna launch environment` | Start it with `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` |
| `LTX worker could not start` … `could not verify this library or module` | This machine (for example RunPod) can't run LTX. Use Vast. |

`Bit depth 10 ... not supported` is OK. Jasna reads that video a bit
slower, that's all.
