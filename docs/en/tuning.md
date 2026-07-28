# Tuning VRAM and GPU Usage

Jasna automatically manages VRAM: when it runs low, queued frames are
temporarily moved to system RAM and back. You don't need to configure
anything for that. The settings below control how much VRAM and GPU time
processing takes.

## Max clip size and temporal overlap

Jasna restores mosaics in clips — sequences of frames processed together.
**Max clip size** caps how long a clip can get (longer = better temporal
consistency, more VRAM). **Temporal overlap** smooths the seams where clips
meet; larger overlap costs processing time. Going above `20` overlap usually
does not help much.

Recommended starting point:

- Use the highest **max clip size** your GPU has the memory for. Bigger clips
  are steadier *and* faster.
- Set **temporal overlap** between `8` and `20`.
- Keep crossfade enabled (it's free — it reuses frames that are already
  processed).

One 1080p video, overlap 8, on an RTX 5090. Your times will differ; the shape
of the curve won't:

| Max clip size | Time | VRAM |
| -------------:| ----:| ----:|
| 30            | 51 s | 3.4 GB |
| 60            | 40 s | 3.7 GB |
| 90            | 34 s | 4.1 GB |
| 180           | 35 s | 5.0 GB |
| 360           | 32 s | 6.3 GB |
| 720           | 33 s | 8.0 GB |

Small clips are **slower**, not faster: the overlap frames are processed again
for every clip, so more clips means more repeated work. Past about `180` the
time stops improving and only VRAM grows. 4K video needs roughly half again as
much VRAM at the same clip size.

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8 --enable-crossfade
```

## Out of VRAM?

1. Reduce **max clip size** — see the table above for what each step costs you.
2. Skip secondary restoration, or pick a lighter one
   (see the [comparison table](models.md#speed-and-vram-comparison)).

Turning off model compilation is no longer a useful way to save VRAM: it frees
under half a gigabyte and makes processing about three times slower.

## Restoration model compilation

On NVIDIA, the restoration model is compiled into TensorRT engines. You can opt
out, but there is little reason to (AMD always uses the PyTorch model):

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

Same 1080p video at clip size 180:

|                    | Time | VRAM |
| ------------------ | ----:| ----:|
| Compiled (default) | 33 s | 5.0 GB |
| No compilation     | 107 s | 4.6 GB |

The engines are the same whatever max clip size you pick, so changing that
setting never makes Jasna compile them again.
