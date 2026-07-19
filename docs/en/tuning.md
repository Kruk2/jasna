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

- Use the highest **max clip size** your GPU can handle.
- Set **temporal overlap** between `8` and `20`.
- Keep crossfade enabled (it's free — it reuses frames that are already
  processed).

Limited testing guidance:

| Max clip size | Temporal overlap | Notes |
| -------------:| ----------------:| ----- |
| 60            | 6                | Lower VRAM option. |
| 90            | 8                | Current default-style balance. |
| 180           | 15               | Needs 12 GB+ VRAM with BasicVSR++ compilation enabled; less with compilation disabled. |

4K videos use more VRAM. A lower clip size may produce similar quality and
process faster. Clip sizes below `60` can work on some videos, but `60` is
preferred even if you need to disable model compilation.

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8 --enable-crossfade
```

## Out of VRAM?

1. Reduce **max clip size** first — for example from `180` to `60`.
2. If that's not enough, disable BasicVSR++ compilation (below). Processing
   gets slower but peak VRAM drops.
3. Skip secondary restoration, or pick a lighter one
   (see the [comparison table](models.md#speed-and-vram-comparison)).

## Restoration model compilation

On NVIDIA, the restoration model is compiled into TensorRT engines.
Compilation improves speed but uses more VRAM. You can opt out at the cost of
performance (AMD always uses the PyTorch model):

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

Compiled engine VRAM only, not total processing VRAM:

|                               | Clip 60 | Clip 180 |
| ----------------------------- | -------:| --------:|
| Engine VRAM, compiled         | ~1.9 GB | ~5.4 GB  |
| Engine VRAM, no compilation   | ~1.2 GB | ~1.2 GB  |
