# LTX encode pass on NVIDIA: window buffers on the GPU, background latent writes, Triton encoder convs (2026-11-12)

The LTX encode pass (crop every window's frames onto its canvas, VAE-encode them) before and after three
changes. Code: main `268011ea` plus this change. Per-run values are in the `.csv`.

- Window buffers on the GPU: the frame loop crops each window's frames into a buffer on the GPU instead of
  copying every crop to host memory. Each of those copies waited until the GPU had run the crop, which sat
  behind the encoder's kernels, so the loop starved the encoder. Buffers stay on the GPU while they fit in
  the free VRAM beyond a 10 GiB reserve, else in host memory as before. Same latents.
- Background latent writes: each window's latent is written to disk on a thread of its own. Each write
  ends with a flush to disk (~30 ms on a hard disk), which kept the encoder waiting. Same latents.
- Triton convs in the encoder: the fp16 head's convs (all but the first) run on a Triton conv kernel that
  accumulates in fp16 like comfy-kitchen's, reads the padding in place instead of from a padded copy, and
  on RTX 50 also takes the 1024-channel layers kitchen declines, so one more down block runs in fp16.
  RTX 40 and 30 use it for the convs where it beats kitchen and keep that block as before. The latents
  change slightly and stay within 1 % of the previous distance from an fp32 encoder (RTX 50: closer).

## Encode pass, S610

S610, 1080p, 6056 frames, 143 windows at 512, Standard LTX (`ltx`), timed by the pipeline as its
`[timing] LTX encode` line (from the end of the scan to the end of the encode, encoder load included).
One process per run, compile caches warm, before (A) and after (B) alternated.

| run | before (A) | after (B) |
|---|---|---|
| 1 | failed (*) | 47.65 s |
| 2 | 56.16 s | 45.24 s |
| 3 | 55.17 s | 45.44 s |
| 4 (next day) | 55.62 s | 47.29 s |
| 5 | 52.58 s | 43.07 s |
| 6 | 54.80 s | 43.17 s |
| median | 55.17 s | 45.34 s (-18 %) |

(*) The process exited during its scan with an access violation in the C runtime, in code this change
does not touch; it was not repeated. Runs 1-3 and 4-6 are two sessions, each alternating A and B.

With main's encoder (no Triton convs) the changed pass writes exactly main's latents (same digest), so
the first two changes do not change the output; on its own the pass then took 47.6 s. B wrote the same
latents in all 16 runs here and below; main wrote different latents in every window once in 13 runs (run 6), so
main's compiled encoder is not always deterministic across processes; B's has not shown that.

Ten more runs of B back to back: 45.27, 44.89, 44.98, 44.80, 50.03, 50.97, 46.27, 45.03, 44.77, 43.70 s
(median 45.0 s), all with the same latents. Five whole S610 jobs (scan to output) with B finished
without error; their encode passes took 68.3 s (the first, while the transformer file was still read
from a cold disk cache beside it), 44.2, 45.1, 47.4 and 47.1 s.

## Where the time went (before)

Measured from outside the pipeline on one run of main, 52.1 s:

| part | time |
|---|---|
| encoder load | 1.1 s |
| first encoder call (loads the compiled encoder) | ~3 s |
| encoder on the GPU, 143 windows | ~38.5 s (0.27 s per window) |
| encoder idle between windows | 9.2 s |

The frame loop (decode, crop, hand-off) ran the whole pass and set its pace: it spent 23-27 s waiting
for crop copies to reach host memory, 7-8 s decoding and 8-10 s waiting for a free encode slot. Each
latent's disk write added ~30 ms per window on the encode thread.

| step | encode pass | encoder idle |
|---|---|---|
| main | 52.1-53.2 s | 9.2-9.6 s |
| window buffers on the GPU | 46.8-47.7 s | 5.4-5.5 s |
| + background latent writes | 45.3-45.7 s | 3.8-4.0 s |
| + Triton encoder convs | 43.1-47.7 s | |

(Rows 2-3 come from the outside measurement, which repeats the encode thread's code; the last row is
the change itself.) What is left of the idle time is the frame loop waiting for the video decoder,
whose per-batch wait sits behind the encoder's kernels (~31 s of the loop).

Tried without gain: high-priority streams for the frame loop and the decoder, 4 instead of 2 windows
waiting for the encoder, decode batches of 16 frames, handing decoded batches over without the
decoder's host wait, and waiting for each window on an event before copying its latent (~0.2 s).

## Encoder, one window

The encoder as the pass loads it, fed bf16 pixels like the pass, timed in turn with main's encoder (7
rounds at 512 on six real S610 windows from the pass, 5 rounds at 768 on a centre crop). Latent
difference: relative L2 against an fp32 encoder.

| canvas | main | with Triton convs | vs fp32, main | vs fp32, Triton convs |
|---|---|---|---|---|
| 512 | 0.2672 s | 0.2546 s (-4.7 %) | 1.379e-2 | 1.311e-2 |
| 768 | 0.6033 s | 0.5685 s (-5.8 %) | 6.95e-3 | 6.75e-3 |

`scripts/benchmark_ltx_encode.py --variants fp32,shipped` on an S610 centre crop, each tree in its own
process (not alternated; the GPU ran ~8 % slower during the second 768 run, see its fp32 row in the
`.csv`): 512 0.2727 -> 0.2571 s, 768 0.6144 -> 0.6116 s; distance from fp32 7.07e-3 -> 6.84e-3 at
512 and 6.95e-3 -> 6.75e-3 at 768.

Other cards, measured on rented instances with `scripts/benchmark_ltx_encoder_convs.py --verify`
(one window of `assets/test_clip1_1080p.mp4`, bf16 pixels; the kitchen head is the encoder as on
main). One entry per CUDA capability; RTX 40 and 30 keep the 1024-channel block in bf16, since their
fp16 sums there move the latents ~5 % further from fp32 (under 1 % on RTX 50).

| GPU | kitchen head, 512 | Triton convs, 512 | 768 | rel L2 vs fp32 at 512, kitchen -> Triton |
|---|---|---|---|---|
| RTX 5090 | 0.2703 s | 0.2585 s (-4.4 %) | -4.9 % | 6.60e-3 -> 6.54e-3 |
| RTX 5060 | 1.261 s | 1.197 s (-5.1 %) | | 6.71e-3 -> 6.67e-3 |
| RTX 4090 | 0.3572 s | 0.3373 s (-5.6 %) | -5.0 % | 8.15e-3 -> 8.11e-3 |
| RTX 4070 | 0.885 s | 0.852 s (-3.8 %) | -4.3 % | 8.14e-3 -> 8.11e-3 |
| RTX 4060 | 1.863 s | 1.793 s (-3.8 %) | | 8.25e-3 -> 8.23e-3 |
| RTX 3090 | 0.7735 s | 0.7376 s (-4.6 %) | -3.9 % | 8.14e-3 -> 8.18e-3 |
| RTX 3060 | 2.035 s | 1.944 s (-4.5 %) | -4.2 % | 8.14e-3 -> 8.18e-3 |

At 768 every entry's latents are closer to fp32 than the kitchen head's (RTX 3090: 7.97e-3 -> 7.89e-3).

Per conv on the RTX 5090 (512, random data): kitchen with the padded copy it needs, cuDNN bf16, and
the Triton conv with its fastest launch.

| layer | per window | kitchen + copy | cuDNN bf16 | Triton |
|---|---|---|---|---|
| res 128 | 8 | 6.47 ms | 8.67 ms | 5.01 ms (350 TFLOPS) |
| downsample 128 -> 64 | 1 | 5.94 ms | 4.71 ms | 3.00 ms |
| res 256 | 12 | 5.40 ms | 7.90 ms | 4.91 ms (357) |
| time downsample 256 | 1 | 5.43 ms | 7.93 ms | 5.01 ms |
| res 512 | 8 | 10.15 ms | 15.13 ms | 9.84 ms (360) |
| downsample 512 -> 128 | 1 | 3.31 ms | 4.18 ms | 2.53 ms |
| res 1024 | 4 | declined | 7.70 ms | 5.01 ms (359) |

The 1024-channel block's activations stay below ~870 on six real windows (fp16 reaches 65504); the
block after it reaches ~2e5 and stays bf16.

## Setup

- RTX 5090 (32 GB), Windows 11, Core i9-13900K. Python 3.13, torch 2.12.1+cu130, Triton 3.7.1,
  comfy-kitchen 0.2.38. Source tree, LTX in-process.
- The desktop was in use: other processes kept 2-5 CPU cores busy during every run (a virus scanner,
  a backup client, telemetry); no other GPU job ran during them. Before runs are slower in the final
  table than in the earlier measurements (52-53 s) under that load.
- `scripts/benchmark_ltx_encoder_convs.py --verify` measures the conv kernel on another GPU; the rented cards
  ran Linux.

## Limits

One GPU and one clip for the pass. The other cards' numbers are for one window of one clip. The window
buffers and background writes also run on AMD, where they are not measured. The time saved per pass
depends on the disk (latent writes) and on how many windows are open at once (GPU buffers).
