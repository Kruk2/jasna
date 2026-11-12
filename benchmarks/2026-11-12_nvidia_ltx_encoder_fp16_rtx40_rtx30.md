# LTX VAE encoder on an RTX 4090 and an RTX 3090: fp16-accumulate head (2026-11-12)

The measurement that adds SM 8.9 and SM 8.6 to `video_encoder.NVIDIA_KITCHEN_DOWN_BLOCKS`: the LTX VAE
encoder (full encoder, `vae.safetensors`) with `conv_in` and the first N down blocks in fp16 on
comfy-kitchen's `fp16_conv3d`, against the compiled bf16 cuDNN encoder. The RTX 5090 numbers are in
`2026-11-12_nvidia_ltx_encoder_fp16.md`. Per-run values are in the `.csv`.

`scripts/benchmark_ltx_encode.py --peaks --repeats 7`, one 121-frame window of `assets/test_clip1_1080p.mp4`,
center-cropped. Latent differences are relative L2 against the fp32 encoder. Each GPU ran the 512 set three
times (the first compiles, one with the variant order reversed); the three agree within 1 %, and the tables
show the second run.

## 512

| variant | RTX 4090 s/window | vs fp32 | RTX 3090 s/window | vs fp32 | activations |
|---|---|---|---|---|---|
| fp32 | 1.220 | | 2.482 | | 6.06 GiB |
| bf16 cuDNN (before) | 0.495 | 7.82e-3 | 1.142 | 7.81e-3 | 2.55 GiB |
| kitchen1 | 0.477 | 6.90e-3 | 1.101 | 6.88e-3 | 1.63 / 2.08 GiB |
| kitchen3 | 0.429 | 6.88e-3 | 0.990 | 6.89e-3 | 1.63 / 1.86 GiB |
| kitchen6 | 0.354 (-28 %) | 7.50e-3 | 0.798 (-30 %) | 7.48e-3 | 1.63 GiB |
| fp16 head on cuDNN (fp32 accumulate) | 0.495 | 3.55e-3 | 1.156 | 3.56e-3 | 2.55 GiB |

Where two activation figures are given, the first is the RTX 4090's.

## 768

| variant | RTX 4090 s/window | vs fp32 | RTX 3090 s/window | vs fp32 | activations |
|---|---|---|---|---|---|
| fp32 | 2.740 | | 5.832 | | 13.64 GiB |
| bf16 cuDNN (before) | 1.112 | 7.54e-3 | 2.615 | 7.53e-3 | 5.74 GiB |
| kitchen3 | 0.964 | 6.59e-3 | 2.272 | 6.58e-3 | 3.65 / 4.19 GiB |
| kitchen6 | 0.801 (-28 %) | 7.19e-3 | 1.833 (-30 %) | 7.17e-3 | 3.65 GiB |

## The encoder as the encode pass loads it

With the two table entries in place, `shipped` is the encoder `load_video_encoder` returns on these GPUs, fed
bf16 pixels as the encode pass feeds it (the `kitchenN` rows above take fp16 pixels).

| | RTX 4090 bf16 | RTX 4090 shipped | RTX 3090 bf16 | RTX 3090 shipped |
|---|---|---|---|---|
| 512 s/window | 0.495 | 0.352 (-29 %) | 1.136 | 0.790 (-30 %) |
| 512 vs fp32 | 7.82e-3 | 8.15e-3 | 7.82e-3 | 8.14e-3 |
| 512 activations | 2.55 GiB | 1.80 GiB | 2.55 GiB | 1.80 GiB |
| 768 s/window | 1.109 | 0.795 (-28 %) | 2.612 | 1.824 (-30 %) |
| 768 vs fp32 | 7.54e-3 | 7.99e-3 | 7.52e-3 | 7.97e-3 |
| 768 activations | 5.74 GiB | 4.05 GiB | 5.74 GiB | 4.05 GiB |

- The whole kitchen6 speedup reaches the loaded encoder on both GPUs.
- Fed bf16 pixels, its latents are 4 % (512) and 6 % (768) further from the fp32 encoder's than the bf16
  encoder's are; fed fp16 pixels they are 4-5 % closer. On the RTX 5090 with another clip the two were equal
  (7.07e-3 against 7.08e-3), and the restored output there differed from an fp32 encoder's by the same
  44.2 dB before and after.
- Six blocks is the entry for both: fewer blocks are slower, an fp16 head on cuDNN is no faster than bf16
  (the gain is the fp16 accumulation), and kitchen declines the seventh block's shape.

## GEMM rates

8192 cubed, TFLOPS, three runs each.

| | bf16 | fp16, fp32 accumulation | fp16, fp16 accumulation |
|---|---|---|---|
| RTX 4090 | 166.7-169.1 | 163.8-166.1 | 262.9-268.4 (1.59x bf16) |
| RTX 3090 | 69.5-70.2 | 66.1-66.5 | 113.6-115.4 (1.65x bf16) |

## Setup

- Two rented single-GPU Linux instances (Ubuntu 22.04 containers), nothing else on the GPU. RTX 4090 (24 GB):
  driver 615.71.09. RTX 3090 (24 GB): driver 595.91.07. Python 3.13, torch 2.12.1+cu130, cuDNN 9.20,
  Triton 3.7.1, comfy-kitchen 0.2.38.
- Code: main `8b8deeb6`; the `shipped` rows with the SM 8.9 / SM 8.6 table entries added.
- TensorRT and the hardware decode wheel were not installed; neither is used here.

## Limits

One card per generation and one clip. The encode pass itself was not timed on these GPUs, only the encoder;
on the RTX 5090 the whole per-window gain reached the pass. Other RTX 40 and 30 models share the
capability and take the fp16 head without having been measured.
