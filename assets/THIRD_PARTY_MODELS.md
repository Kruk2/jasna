# Model licenses and provenance

Hashes below identify the v0.11.0 candidate artifacts. Release preparation
must fail or update this file when a bundled model changes.

## Lada BasicVSR++ mosaic restoration

- Jasna model name: `basicvsrpp`
- File: `lada_mosaic_restoration_model_generic_v1.2.pth`
- Upstream: `ladaapp/lada` model repository
- Upstream revision: `3bfd69ffc21518bde80ba6b61696d51efd0a398b`
- SHA-256: `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee`
- License: AGPL-3.0
- Copyright: ladaapp and contributors

This checkpoint is redistributed unmodified.

## Lada YOLO v4 fast mosaic detection

- Jasna model name: `lada-yolo-v4`
- File: `lada_mosaic_detection_model_v4_fast.pt`
- Upstream: `ladaapp/lada` model repository
- Upstream revision: `404620fe2f6b72657b92f76e62af914c8b3ee686`
- SHA-256: `9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4`
- License: AGPL-3.0
- Copyright: ladaapp and contributors

This checkpoint is redistributed unmodified.

## LTX restoration (LTX-2.5 video VAE and fine-tuned transformer)

- Jasna model name: `ltx`
- Folder: `model_weights/ltx-restore/`
- `vae.safetensors`: Lightricks `ltx-2.5-video-vae-bf16.safetensors`, redistributed unmodified
- `vae.safetensors` SHA-256: `847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3`
- `vae-decoder.safetensors`: Jasna fine-tune of the LTX-2.5 video VAE decoder
- `transformer.safetensors`: Jasna fine-tune of the LTX-2.5 22B transformer (video path only, INT8)
- Upstream: [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
- License: LTX-2.x Community License (`assets/licenses/LTX-2.x-Community-License.txt`)
- Copyright: Lightricks Ltd.; fine-tunes 2026 Kruk2

These files are not part of release packages yet; hashes for the fine-tuned
files are added when they are.

## NomosUni SPAN 2x upscaler

- Jasna model name: `span` (whole-video upscaling)
- File: `span-nomosuni-2x.safetensors`
- Upstream: `2x-NomosUni_span_multijpg_ldl` by Philip Hofmann (Helaman), listed on
  [OpenModelDB](https://openmodeldb.info/models/2x-NomosUni-span-multijpg-ldl)
- Upstream SHA-256: `bc5fa00b9ee304301a616163275e34421bb507586176db4387239fa3a87e6bb6`
- SHA-256: `7eb740e61a69ea6b4ad45d88d79c712c8df7787da4f51839efe364bf2ba6e890`
- Architecture: SPAN (Wan et al., [hongyuanyu/SPAN](https://github.com/hongyuanyu/SPAN), Apache-2.0)
- License: CC-BY-4.0 (`assets/licenses/CC-BY-4.0.txt`)
- Copyright: Philip Hofmann

Modified: `scripts/fuse_span_weights.py` folds each training-time convolution
chain of the upstream checkpoint into the single convolution it computes at
inference. The result computes the same picture.

## Jasna RF-DETR v6

- Jasna model name: `rfdetr-v6`
- NVIDIA file: `rfdetr-v6.onnx`
- NVIDIA SHA-256: `59e5b2f5b7b7a5713d0f102fbe409cca3f4243377bd17b30e3d6169b2a366af6`
- AMD file: `rfdetr-v6.pt`
- AMD SHA-256: `a92ccba2f4fe4eb7e7bcdb567c1bacbe01c630ea0278a25e5138988725b98243`
- Architecture/source: [RF-DETR 1.8.3](https://github.com/roboflow/rf-detr/tree/3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86)
- License: Apache-2.0
- Copyright: 2026 Kruk2

This project-trained RF-DETR Seg Medium checkpoint uses classes
`Background` and `mosaic`. The ONNX and PyTorch files are deployment
formats of the same trained model.

## Jasna RF-DETR VR v1

- Jasna model name: `rfdetr-vr-v1`
- NVIDIA file: `rfdetr-vr-v1.onnx`
- NVIDIA SHA-256: `9a4cfb97a953e9f76eb0e653b8623d93c276c34db28f6efd8bbc88daa32ec24c`
- AMD file: `rfdetr-vr-v1.pt`
- AMD SHA-256: `b380e3b30d65210fd2f61fef654996c3592f5c2e4ec8df3d4873ee9514bf3b3f`
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
