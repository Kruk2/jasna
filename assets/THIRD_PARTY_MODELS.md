# Model licenses and provenance

Hashes below identify the v0.11.0 candidate artifacts. Release preparation
must fail or update this file when a bundled model changes.

## Lada BasicVSR++ mosaic restoration

- Jasna model name: `basicvsrpp`
- File: `lada_mosaic_restoration_model_generic_v1.2.pth`
- Upstream: <https://huggingface.co/ladaapp/lada/blob/3bfd69ffc21518bde80ba6b61696d51efd0a398b/lada_mosaic_restoration_model_generic_v1.2.pth>
- Upstream revision: `3bfd69ffc21518bde80ba6b61696d51efd0a398b`
- SHA-256: `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee`
- License: AGPL-3.0
- Copyright: ladaapp and contributors

This checkpoint is redistributed unmodified.

## Lada YOLO v4 fast mosaic detection

- Jasna model name: `lada-yolo-v4`
- File: `lada_mosaic_detection_model_v4_fast.pt`
- Upstream: <https://huggingface.co/ladaapp/lada/blob/404620fe2f6b72657b92f76e62af914c8b3ee686/lada_mosaic_detection_model_v4_fast.pt>
- Upstream revision: `404620fe2f6b72657b92f76e62af914c8b3ee686`
- SHA-256: `9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4`
- License: AGPL-3.0
- Copyright: ladaapp and contributors

This checkpoint is redistributed unmodified.

## Jasna RF-DETR v6

- Jasna model name: `rfdetr-v6`
- NVIDIA file: `rfdetr-v6.onnx`
- NVIDIA SHA-256: `b6555cfce325d1d8bc413422cd46f3a453511a246c6f29ce652382998049d825`
- AMD file: `rfdetr-v6.pt`
- AMD SHA-256: `f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55`
- Architecture/source: [RF-DETR 1.8.3](https://github.com/roboflow/rf-detr/tree/3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86)
- License: Apache-2.0
- Copyright: 2026 Kruk2

This project-trained RF-DETR Seg Medium checkpoint uses classes
`Background` and `mosaic`. The ONNX and PyTorch files are deployment
formats of the same trained model.

## Jasna RF-DETR VR v1

- Jasna model name: `rfdetr-vr-v1`
- NVIDIA file: `rfdetr-vr-v1.onnx`
- NVIDIA SHA-256: `6e2ed2043851dccb97f21deda38dc20ea2b8e265e682359752e815c600030a40`
- AMD file: `rfdetr-vr-v1.pt`
- AMD SHA-256: `55543c83911921ef79cd8cae8540bd25e34c7daf488e77f79d233d6926973a2e`
- Architecture/source: [RF-DETR 1.8.3](https://github.com/roboflow/rf-detr/tree/3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86)
- License: Apache-2.0
- Copyright: 2026 Kruk2

This project-trained RF-DETR Seg Large checkpoint is trained for side-by-side
VR material. The ONNX and PyTorch files are deployment formats of the same
trained model.

## ZeLeFans VR Mosaic Detection v2 accurate

- Jasna model name: `zelefans-vr-yolo-v2`
- Upstream: <https://huggingface.co/zelefans/vrmr>
- Upstream project: <https://codeberg.org/zelefans/vr_remove_mosaic>
- Pinned revision: `0f65a21133335f9a4ec6fc5d7da8d3385bfdb8b1`
- Upstream file: `lada_vr_mosaic_detection_model_v2_accurate.pt`
- SHA-256: `91fe7a48b0e9edf51361918c8a30f752c64511005e643343a7382d951f3fe0f8`
- License: Apache-2.0

This optional checkpoint is not bundled in the standard v0.11.0 package.

## Supporter models

`unet-4x.onnx.enc` and the encrypted SD 1.5 Jasna checkpoint are
project-trained, proprietary supporter models. Copyright 2026 Kruk2. They are
provided only for use with Jasna by a holder of a valid supporter key.
Redistribution, extraction, modification, and use outside Jasna are not
granted.

The base v0.11.0 release may include `unet-4x.onnx.enc`; SD 1.5 is downloaded
separately when requested. The protection implementation and supporter-model
terms are separate from Jasna's AGPL-covered public application source.
