from __future__ import annotations

from pathlib import Path

import torch

RFDETR_MODEL_NAMES: frozenset[str] = frozenset({"rfdetr-v2", "rfdetr-v3", "rfdetr-v4", "rfdetr-v5"})
YOLO_MODEL_NAMES: frozenset[str] = frozenset({"lada-yolo-v2", "lada-yolo-v4"})

DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v5"

YOLO_MODEL_FILES: dict[str, str] = {
    "lada-yolo-v2": "lada_mosaic_detection_model_v2.pt",
    "lada-yolo-v4": "lada_mosaic_detection_model_v4_fast.pt",
}


def coerce_detection_model_name(name: str) -> str:
    name = str(name).strip().lower()
    if name in RFDETR_MODEL_NAMES or name in YOLO_MODEL_NAMES:
        return name
    return DEFAULT_DETECTION_MODEL_NAME


def detection_model_weights_path(name: str) -> Path:
    name = coerce_detection_model_name(name)
    if name in RFDETR_MODEL_NAMES:
        return Path("model_weights") / f"{name}.onnx"
    if name in YOLO_MODEL_NAMES:
        return Path("model_weights") / YOLO_MODEL_FILES[name]
    return Path("model_weights") / f"{DEFAULT_DETECTION_MODEL_NAME}.onnx"


def precompile_detection_engine(
    detection_model_name: str,
    detection_model_path: Path,
    batch_size: int,
    device: torch.device,
    fp16: bool,
) -> None:
    if device.type != "cuda":
        return
    det_name = coerce_detection_model_name(detection_model_name)
    if det_name in RFDETR_MODEL_NAMES:
        from jasna.mosaic.rfdetr import compile_rfdetr_engine

        compile_rfdetr_engine(detection_model_path, device, batch_size=int(batch_size), fp16=bool(fp16))
    elif det_name in YOLO_MODEL_NAMES:
        from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine
        from jasna.mosaic.yolo import YoloMosaicDetectionModel

        compile_yolo_to_tensorrt_engine(
            detection_model_path,
            batch=int(batch_size),
            fp16=bool(fp16) and (device.type == "cuda"),
            imgsz=YoloMosaicDetectionModel.DEFAULT_IMGSZ,
            device=device,
        )

