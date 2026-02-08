from __future__ import annotations

from pathlib import Path

RFDETR_MODEL_NAMES: frozenset[str] = frozenset({"rfdetr-v2", "rfdetr-v3"})
YOLO_MODEL_NAMES: frozenset[str] = frozenset({"lada-yolo-v2", "lada-yolo-v4"})

DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v3"

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

