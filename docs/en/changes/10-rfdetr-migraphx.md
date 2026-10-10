# Validated RF-DETR MIGraphX selection

English (default) | [中文](../../zh/changes/10-rfdetr-migraphx.md)

Feature: `10-rfdetr-migraphx`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Select a validated AMD RF-DETR artifact at the shared detection registry boundary. Validate source weights, runtime, architecture, file hashes, and tensor ABI; use the same selection for scanning and processing.

## Usage and default behavior

Linux AMD gfx1100 rfdetr-v6 FP16 with an installed matching sidecar admits the direct MIGraphX route. No sidecar means the normal product Torch path remains selected; an installed but invalid sidecar fails visibly.

## Direct prerequisites

- [Opt-in Windows HIP resize normalization](05-windows-hip-resize.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

This does not change the detection threshold, tracker, restoration model, product batch, or NVIDIA backend. Artifact acceptance is not transferable across runtime upgrades without identity validation.

## Validation and reproduction

```bash
python -m pytest -q tests/test_detection_registry.py tests/test_migraphx_artifact.py tests/test_model_weights_dir.py tests/test_rfdetr_migraphx_product.py tests/test_rfdetr_postprocess.py tests/test_windows_hip_resize_integration.py tests/test_yolo_call.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/RFDETR_MIGRAPHX_CORE_CN.md](../../../docs/RFDETR_MIGRAPHX_CORE_CN.md)
- [docs/RFDETR_MIGRAPHX_PRODUCT_CN.md](../../../docs/RFDETR_MIGRAPHX_PRODUCT_CN.md)
- [jasna/gui/engine_preflight.py](../../../jasna/gui/engine_preflight.py)
- [jasna/migraphx_artifact.py](../../../jasna/migraphx_artifact.py)
- [jasna/mosaic/detection_registry.py](../../../jasna/mosaic/detection_registry.py)
- [jasna/mosaic/rfdetr.py](../../../jasna/mosaic/rfdetr.py)
- [jasna/mosaic/rfdetr_migraphx_runner.py](../../../jasna/mosaic/rfdetr_migraphx_runner.py)
- [jasna/mosaic/yolo.py](../../../jasna/mosaic/yolo.py)
- [tests/test_detection_registry.py](../../../tests/test_detection_registry.py)
- [tests/test_migraphx_artifact.py](../../../tests/test_migraphx_artifact.py)
- [tests/test_model_weights_dir.py](../../../tests/test_model_weights_dir.py)
- [tests/test_rfdetr_migraphx_product.py](../../../tests/test_rfdetr_migraphx_product.py)
- [tests/test_rfdetr_postprocess.py](../../../tests/test_rfdetr_postprocess.py)
- [tests/test_windows_hip_resize_integration.py](../../../tests/test_windows_hip_resize_integration.py)
- [tests/test_yolo_call.py](../../../tests/test_yolo_call.py)
