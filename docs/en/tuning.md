# Tuning VRAM and GPU Usage

When VRAM gets low, Jasna moves waiting frames to system RAM and back. You
don't need to set anything for this. The settings below change how much VRAM
and GPU time a job uses.

The clip and compilation settings are for BasicVSR++. LTX manages its memory by itself, see
[What LTX needs](models.md#what-ltx-needs).

## Max clip size and temporal overlap

Jasna restores mosaics in clips: groups of frames processed together.

- **Max Clip Size** (**Model** tab, **Advanced** view): the longest clip, in frames. Longer clips give a more
  stable result but use more VRAM.
- **Temporal Overlap**: smooths the places where two clips meet. More overlap
  takes more time. Above `20` it usually doesn't help.

Good start:

- Use the biggest **Max Clip Size** that fits in your VRAM. Bigger clips are
  more stable *and* faster.
- Set **Temporal Overlap** between `8` and `20`.

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
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8
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

## Environment variables (advanced)

These are for troubleshooting and testing. The defaults are best for most people.

| Variable | Values | What it does |
| --- | --- | --- |
| `JASNA_LTX_VAE_ENCODER` | `auto` (default), `full`, `half`, `max` | Command line only, the app has **LTX Encoder** (see [LTX options](models.md#ltx-options)). Picks the LTX video encoder: `half` and `max` are faster and a little softer. `auto`: `half` for Basic and Standard Quality, `full` for the others. |
| `JASNA_LTX_VAE_DECODER` | `auto` (default), `full`, `half`, `max` | Command line only, the app has **LTX Decoder**. Picks the LTX video decoder: `half` and `max` are faster and a little softer. `auto`: `max` for Basic and Standard Quality, `full` for the others. |
| `JASNA_DECODE_BACKEND` | `auto` (default), `vali`, `pyav-hw`, `pyav-sw` | Command line only, the app has **Video Decoder** (**Model** tab, **Expert** view). Picks how Jasna reads the input video. Try `pyav-sw` (slower) if a video fails to open or shows broken frames. `vali` is NVIDIA only. |
| `JASNA_TRT_CUDAGRAPHS` | `1` (default), `0` | NVIDIA: `0` turns off a speed-up of compiled BasicVSR++. Try it if you get GPU errors. |
| `JASNA_AMD_GRAPHS` | `1` (default), `0` | AMD: `0` turns off a BasicVSR++ speed-up. Try it if you get GPU errors or run out of VRAM. |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` (default), `0` | AMD: `0` turns off a faster attention mode. Try it together with `JASNA_AMD_GRAPHS=0` if you get GPU errors. |
| `TVAI_MODEL_DATA_DIR`, `TVAI_MODEL_DIR` | folder paths | Your Topaz Video model folders, needed for TVAI (see [Secondary restoration](models.md#secondary-restoration)). |

Windows: open Command Prompt in the Jasna folder, set the variable and start
Jasna in the same window (or save both lines as a `.bat` file there):

```bat
set JASNA_TRT_CUDAGRAPHS=0
jasna.exe
```

Linux (on AMD start `./run_jasna_amd.sh` instead of `./jasna`):

```bash
JASNA_TRT_CUDAGRAPHS=0 ./jasna
```

The setting lasts until you close that window.
