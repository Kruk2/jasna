# LTX VAE encoder on an RTX 5090: fp16-accumulate head on comfy-kitchen (2026-11-12)

The LTX VAE encoder (full encoder, `vae.safetensors`) before and after running `conv_in` and the first six
down blocks in fp16 on comfy-kitchen's `fp16_conv3d`, which accumulates in fp16
(`video_encoder.NVIDIA_KITCHEN_DOWN_BLOCKS`, SM 12.0 only). Before, every conv ran as a compiled bf16
cuDNN conv. Code: main `a453bc84` plus this change. Per-run values are in the `.csv`.

## Encode pass, S610

S610, 1080p, 6056 frames, 143 windows at 512, Standard LTX (`ltx`). The pass is timed by the pipeline, the
same way as the `[timing] LTX encode` line, from the end of the scan to the end of the encode (encoder load
included). Runs alternated before/after, one process per run, compile caches warm.

| run | before | after |
|---|---|---|
| 1 | 67.9 s | 54.5 s |
| 2 | 66.7 s | 53.2 s |
| 3 | 67.5 s | 56.1 s |
| median | 67.5 s (0.39-0.40 s/window) | 54.5 s (0.30 s/window, -19 %) |

Peak VRAM of the pass: 3382 MiB before, 2620 MiB after.

## Encoder, one window

One 121-frame S610 window, center-cropped, encoders loaded side by side and timed in turn (7 rounds at 512,
5 at 768). Latent difference: relative L2 against the bf16 encoder and against an fp32 encoder. The
fp16-head rows take fp16 pixels; the encode pass hands the encoder bf16 pixels (last section).

| 512 | s/window | activations | vs bf16 | vs fp32 |
|---|---|---|---|---|
| bf16 (before) | 0.3604 | 2.55 GiB | | 7.1e-3 |
| kitchen6 (after) | 0.2747 (-24 %) | 1.63 GiB | 7.7e-3 | 5.5e-3 |
| kitchen1 | 0.3429 | 1.63 GiB | 7.3e-3 | 5.4e-3 |
| kitchen3 | 0.3160 | 1.63 GiB | 7.4e-3 | 5.3e-3 |
| fp16 head on cuDNN (fp32 accumulate) | 0.3637 | 2.55 GiB | 6.7e-3 | 4.0e-3 |

| 768 | s/window | activations | vs bf16 | vs fp32 |
|---|---|---|---|---|
| bf16 (before) | 0.802 | 5.74 GiB | | 7.0e-3 |
| kitchen6 (after) | 0.608 (-24 %) | 3.65 GiB | 7.6e-3 | 5.5e-3 |
| kitchen5 | 0.607 | 3.65 GiB | 7.5e-3 | 5.4e-3 |

- Fed fp16 pixels, the fp16 head lands closer to an fp32 encoder than bf16 does: fp16 keeps three more
  mantissa bits than bf16, which outweighs the fp16 accumulation (cuDNN's fp16 convs, which accumulate in
  fp32, are closer still but no faster). The encode pass builds its pixels in bf16, so there the rounding
  of the input dominates and the latents are as close to fp32 as before (next section).
- Kitchen declines `down_blocks[6]` (1024 channels), so kitchen6 and kitchen7 are the same; the blocks after
  it leave fp16's range.
- On a 20 s 1080p clip (600 frames, 6 windows) the restored frames after the change differ from before by
  44.3 dB PSNR in the restored areas (mean 1.0 of 255). An fp32 encoder differs from the bf16 one by the
  same 44.2 dB, and from the fp16 head by 44.2 dB too: the denoiser turns any small latent change into a
  difference of this size, and the fp16 head is no further from the fp32 result than bf16 was. Runs of the
  same code are bit-identical.

## The encoder as the encode pass loads it

`scripts/benchmark_ltx_encode.py --peaks --repeats 7`, 512, same S610 window, one variant after the other
(not alternated; another process used the GPU briefly during this run). `eager+cl3d+compile` is the bf16
encoder; `shipped` is the encoder as loaded after the change, fed bf16 pixels like the encode pass.

| variant | s/window | activations | vs fp32 |
|---|---|---|---|
| fp32 | 0.871 | 6.06 GiB | |
| eager+cl3d+compile (bf16) | 0.359 | 2.55 GiB | 7.08e-3 |
| shipped (fp16 kitchen head, bf16 pixels) | 0.268 | 1.80 GiB | 7.07e-3 |
| kitchen6+cl3d+compile (fp16 pixels) | 0.267 | 1.63 GiB | 5.54e-3 |

GEMM rates in the same run: bf16 233.4 TFLOPS, fp16 with fp32 accumulation 228.7, with fp16
accumulation 355.2.

## Where the time goes (512)

Before the change, one window spent ~323 ms in cuDNN convs (91 %) and ~30 ms in compiled glue
(PixelNorm, SiLU, causal padding, space-to-depth skips). The convs total 76 TFLOP per window, so they
ran at ~235 TFLOPS, the card's bf16 GEMM rate; cuDNN autotuning gained nothing and channels-first layouts
were up to 40 % slower.

GEMM rates, 8192 cubed: bf16 230.6 TFLOPS, fp16 with fp32 accumulation 226.1, fp16 with fp16 accumulation
364.1, fp8 449.2, int8 686.9 TOPS.

Per conv shape (3x3x3, channels-last, random data; count per window):

| layer | count | GFLOP | cuDNN bf16 | kitchen fp16 |
|---|---|---|---|---|
| conv_in 48->128 | 1 | 658 | 4.41 ms (149 TFLOPS) | 2.33 ms (282) |
| res 128 | 8 | 1754 | 8.02 ms (219) | 4.92 ms (356) |
| downsample 128->64 | 1 | 877 | 4.08 ms (215) | 4.30 ms (204) |
| res 256 | 12 | 1754 | 7.79 ms (225) | 5.07 ms (346) |
| time downsample 256 | 1 | 1769 | 9.52 ms (186) | 4.43 ms (399) |
| res 512 | 8 | 3537 | 16.6 ms (213) | 9.54 ms (371) |
| downsample 512->128 | 1 | 899 | 3.85 ms (233) | 2.42 ms (372) |
| res 1024 (32 px) | 4 | 1797 | 7.65 ms (235) | declined |
| downsample 1024->128 | 1 | 232 | 1.11 ms (208) | |
| res 1024 (16 px) | 4 | 232 | 1.24 ms (187) | |

A Triton implicit-GEMM conv that promotes its fp16 sums to fp32 every 64 channels reached 308-321 TFLOPS
with 5-15x less error per conv than kitchen; it was dropped, since kitchen is 10-15 % faster on the large
shapes and its error does not show in the output.

## Setup

- RTX 5090 (32 GB), driver 610.74, Windows 11 (26200), Core i9-13900K. Python 3.13, torch 2.12.1+cu130,
  cuDNN 9.20, Triton 3.7.1, comfy-kitchen 0.2.38. Source tree, LTX in-process.
- The desktop was in use, and other processes kept 4-6 CPU cores busy during the encode-pass and 512
  runs; no other GPU job ran during them. In the run of the previous section about 3 cores were busy and
  another process used the GPU for a short time. The 768 table, the conv table and the first GEMM rates
  have no load record.
- `scripts/benchmark_ltx_encode.py --peaks --variants fp32,eager+cl3d+compile,kitchen3+cl3d+compile,kitchen6+cl3d+compile`
  measures the same on another GPU.

## Limits

One GPU. RTX 30 and 40 cards are not measured and keep the bf16 encoder until they are. Workstation cards
that report SM 12.0 take the fp16 head without having been measured. One clip for the encode pass, one for
the output check.
