# Tuning VRAM and GPU Usage

When VRAM gets low, Jasna moves waiting frames to system RAM and back. You
don't need to set anything for this. The settings below change how much VRAM
and GPU time a job uses.

These settings are for BasicVSR++. LTX manages its memory by itself, see
[What LTX needs](models.md#what-ltx-needs).

## Max clip size and temporal overlap

Jasna restores mosaics in clips: groups of frames processed together.

- **Max Clip Size**: the longest clip, in frames. Longer clips give a more
  stable result but use more VRAM.
- **Temporal Overlap**: smooths the places where two clips meet. More overlap
  takes more time. Above `20` it usually doesn't help.

Good start:

- Use the biggest **Max Clip Size** that fits in your VRAM. Bigger clips are
  more stable *and* faster.
- Set **Temporal Overlap** between `8` and `20`.
- Keep **Enable Crossfade** on. It costs nothing, because it uses frames that are
  already done.

One 1080p video, overlap 8, RTX 5090. Your times will be different, but the
trend is the same:

| Max clip size | Time | VRAM |
| -------------:| ----:| ----:|
| 30            | 51 s | 3.4 GB |
| 60            | 40 s | 3.7 GB |
| 90            | 34 s | 4.1 GB |
| 180           | 35 s | 5.0 GB |
| 360           | 32 s | 6.3 GB |
| 720           | 33 s | 8.0 GB |

Small clips are **slower**, not faster. The overlap frames are processed again
for every clip, so more clips means more repeated work. Above about `180` the
time doesn't get better, only VRAM goes up. 4K video needs about 1.5x more
VRAM at the same clip size.

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8 --enable-crossfade
```

## Out of VRAM?

1. Lower **Max Clip Size**. The table above shows what each step costs.
2. Turn off secondary restoration, or use a lighter one (see the
   [comparison](models.md#speed-and-vram-comparison)).
3. Close other apps that use the GPU (browsers, games, video players).

Cards with 4–6 GB of VRAM: start with max clip size `30` for 1080p video. 4K
needs more.

Turning off **Compile BasicVSR++** does not help: it saves less than 0.5 GB and
makes processing about 3x slower.

## Model compilation

On NVIDIA, Jasna compiles the detection and restoration models (and UNet 4x,
if used) for your exact GPU (TensorRT). This makes processing much faster. AMD never compiles.

- It happens on the first job and takes about **15–60 minutes**. Close other
  apps and don't use the PC during this time.
- The result is saved in `model_weights` and used again on every later job.
- You can copy `model_weights` from an older Jasna version to a newer one, so
  you don't wait again.

You can turn compilation off for the restoration model, but there is almost
no reason to.

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

Same 1080p video, clip size 180:

|                    | Time | VRAM |
| ------------------ | ----:| ----:|
| Compiled (default) | 33 s | 5.0 GB |
| Not compiled       | 107 s | 4.6 GB |

Changing max clip size does not start the compilation again.
