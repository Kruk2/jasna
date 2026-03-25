from __future__ import annotations

import re
from pathlib import Path

RFDETR_MODEL_NAMES: frozenset[str] = frozenset({"rfdetr-v2", "rfdetr-v3", "rfdetr-v4", "rfdetr-v5"})
YOLO_MODEL_NAMES: frozenset[str] = frozenset({"lada-yolo-v2", "lada-yolo-v4"})

DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v5"

YOLO_MODEL_FILES: dict[str, str] = {
    "lada-yolo-v2": "lada_mosaic_detection_model_v2.pt",
    "lada-yolo-v4": "lada_mosaic_detection_model_v4_fast.pt",
}

_RFDETR_PATTERN = re.compile(r"^rfdetr-.+$")


def is_rfdetr_model(name: str) -> bool:
    return bool(_RFDETR_PATTERN.match(name))


def is_yolo_model(name: str) -> bool:
    return name in YOLO_MODEL_NAMES


def discover_available_detection_models(model_weights_dir: Path = Path("model_weights")) -> list[str]:
    rfdetr_names: list[str] = []
    yolo_names: list[str] = []
    if model_weights_dir.is_dir():
        for f in model_weights_dir.iterdir():
            if f.suffix == ".onnx" and is_rfdetr_model(f.stem):
                rfdetr_names.append(f.stem)
        yolo_files_reverse = {v: k for k, v in YOLO_MODEL_FILES.items()}
        for f in model_weights_dir.iterdir():
            if f.name in yolo_files_reverse:
                yolo_names.append(yolo_files_reverse[f.name])
    rfdetr_names.sort(reverse=True)
    yolo_names.sort(reverse=True)
    return rfdetr_names + yolo_names


def coerce_detection_model_name(name: str) -> str:
    name = str(name).strip().lower()
    if is_rfdetr_model(name) or is_yolo_model(name):
        return name
    return DEFAULT_DETECTION_MODEL_NAME


def detection_model_weights_path(name: str) -> Path:
    name = coerce_detection_model_name(name)
    if is_rfdetr_model(name):
        return Path("model_weights") / f"{name}.onnx"
    if is_yolo_model(name):
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
    if is_rfdetr_model(det_name):
        from jasna.mosaic.rfdetr import compile_rfdetr_engine

        compile_rfdetr_engine(detection_model_path, device, batch_size=int(batch_size), fp16=bool(fp16))
    elif is_yolo_model(det_name):
        from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine
        from jasna.mosaic.yolo import YoloMosaicDetectionModel

        compile_yolo_to_tensorrt_engine(
            detection_model_path,
            batch=int(batch_size),
            fp16=bool(fp16) and (device.type == "cuda"),
            imgsz=YoloMosaicDetectionModel.DEFAULT_IMGSZ,
            device=device,
        )

