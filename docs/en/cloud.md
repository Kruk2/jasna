# Cloud GPUs (RunPod, Vast, Docker)

You can run Jasna on a rented GPU. Use the Linux NVIDIA version from the
command line (the app window won't open there).

## What you need

- Your seller has to turn on **cloud** for your License code.
- An NVIDIA GPU with driver **580 or newer**.
- At least **80 GB** of disk.
- `NVIDIA_DRIVER_CAPABILITIES=all` in your template or instance settings.
  Jasna can't read or save videos without it. On Vast, add
  `-e NVIDIA_DRIVER_CAPABILITIES=all` to the Docker options.
- Log in as `root`. RunPod and Vast do this by default.

## Install

Paste these lines into the terminal of your rented machine. If you use a
newer Jasna, change `0.11.0` to your version.

```bash
apt-get update && apt-get install -y zstd wget libgl1 libglib2.0-0

cd /root
for part in 000 001 002; do wget https://github.com/Kruk2/jasna/releases/download/v0.11.0/jasna-linux-nvidia-0.11.0.tar.zst.part$part; done

cat jasna-linux-nvidia-0.11.0.tar.zst.part* | zstd -dc | tar -xpf - --same-owner
cd jasna-linux-nvidia-0.11.0
```

Unpack in `/root` like above. Jasna won't start from `/workspace` or a
network drive.

## Activate

```bash
export XDG_CONFIG_HOME=/workspace/jasna-config
./jasna --license-code YOUR-LICENSE-CODE --cloud-workspace
```

After `installation verified` it can look stuck for up to a minute. Just
wait, don't press Ctrl+C.

Your activation is saved in `/workspace/jasna-config`. Run the `export`
line every time before you start Jasna. If you rent a new machine, bring
that folder with you (RunPod volume, or download it to your PC). If you lose
it, you need a free slot to activate again.

## Restore a video

```bash
export XDG_CONFIG_HOME=/workspace/jasna-config
./jasna --input /root/video.mp4 --output /root/video-restored.mp4 --restoration-model-name ltx
```

The first time, Jasna asks to download the LTX model (15 GB). Type `y`.

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
| `This container needs a Cloud workspace key` | Add `--cloud-workspace`. |
| `enable cloud activation` | Ask your seller to turn on cloud. |
| `No activation found` or `Cloud workspace identity is missing` | Run the `export` line, then start Jasna again. |
| `unsafe Jasna launch environment` | Start it with `env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD ./jasna ...` |

`Bit depth 10 ... not supported` is OK. Jasna reads that video a bit
slower, that's all.

Don't share your `jasna-config` folder. Anyone who has it can use your
activation.
