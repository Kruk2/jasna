from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

from jasna.accelerator import is_amd_device, is_nvidia_device
from jasna.engine_paths import model_weights_dir

@dataclass(frozen=True)
class DetectionModelSpec:
    name: str
    backend: str
    filename: str


DETECTION_MODEL_SPECS: dict[str, DetectionModelSpec] = {
    "lada-yolo-v2": DetectionModelSpec(
        "lada-yolo-v2",
        "yolo",
        "lada_mosaic_detection_model_v2.pt",
    ),
    "lada-yolo-v4": DetectionModelSpec(
        "lada-yolo-v4",
        "yolo",
        "lada_mosaic_detection_model_v4_fast.pt",
    ),
    "zelefans-vr-yolo-v2": DetectionModelSpec(
        "zelefans-vr-yolo-v2",
        "yolo",
        "lada_vr_mosaic_detection_model_v2_accurate.pt",
    ),
}

RFDETR_MODEL_NAMES: frozenset[str] = frozenset({"rfdetr-v2", "rfdetr-v3", "rfdetr-v4", "rfdetr-v5"})
YOLO_MODEL_NAMES: frozenset[str] = frozenset(
    name
    for name, spec in DETECTION_MODEL_SPECS.items()
    if spec.backend == "yolo"
)

DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v5"

YOLO_MODEL_FILES: dict[str, str] = {
    name: spec.filename
    for name, spec in DETECTION_MODEL_SPECS.items()
    if spec.backend == "yolo"
}

_RFDETR_PATTERN = re.compile(r"^rfdetr-.+$")


def is_rfdetr_model(name: str) -> bool:
    return bool(_RFDETR_PATTERN.match(name))


def is_yolo_model(name: str) -> bool:
    spec = DETECTION_MODEL_SPECS.get(str(name))
    return spec is not None and spec.backend == "yolo"


def detection_model_spec(name: str) -> DetectionModelSpec:
    normalized = str(name).strip().lower()
    spec = DETECTION_MODEL_SPECS.get(normalized)
    if spec is not None:
        return spec
    if is_rfdetr_model(normalized):
        return DetectionModelSpec(
            normalized,
            "rfdetr",
            f"{normalized}.onnx",
        )
    valid = sorted(RFDETR_MODEL_NAMES | set(DETECTION_MODEL_SPECS))
    raise ValueError(
        f"Unknown detection model '{normalized}'. Valid names: {', '.join(valid)}"
    )


def discover_available_detection_models(weights_dir: Path | None = None) -> list[str]:
    weights_dir = weights_dir if weights_dir is not None else model_weights_dir()
    rfdetr_names: list[str] = []
    yolo_names: list[str] = []
    if weights_dir.is_dir():
        for f in weights_dir.iterdir():
            if f.suffix == ".onnx" and is_rfdetr_model(f.stem):
                rfdetr_names.append(f.stem)
        yolo_files_reverse = {v: k for k, v in YOLO_MODEL_FILES.items()}
        for f in weights_dir.iterdir():
            if f.name in yolo_files_reverse:
                yolo_names.append(yolo_files_reverse[f.name])
    rfdetr_names.sort(reverse=True)
    yolo_names.sort(reverse=True)
    return rfdetr_names + yolo_names


def detection_model_choices(weights_dir: Path | None = None) -> list[str]:
    choices = discover_available_detection_models(weights_dir)
    if not choices:
        choices.append(DEFAULT_DETECTION_MODEL_NAME)
    return choices


def coerce_detection_model_name(name: str) -> str:
    name = str(name).strip().lower()
    return detection_model_spec(name).name


def detection_model_weights_path(name: str) -> Path:
    name = coerce_detection_model_name(name)
    base = model_weights_dir()
    return base / detection_model_spec(name).filename


def require_detection_model_weights(name: str) -> Path:
    path = detection_model_weights_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Detection model weights not found: {path}")
    return path


def build_detection_model(
    detection_model_name: str,
    detection_model_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    score_threshold: float,
    fp16: bool,
):
    det_name = coerce_detection_model_name(detection_model_name)
    if is_rfdetr_model(det_name):
        from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

        return RfDetrMosaicDetectionModel(
            onnx_path=detection_model_path,
            batch_size=int(batch_size),
            device=device,
            score_threshold=float(score_threshold),
            fp16=bool(fp16),
        )
    from jasna.mosaic.yolo import YoloMosaicDetectionModel

    return YoloMosaicDetectionModel(
        model_path=detection_model_path,
        batch_size=int(batch_size),
        device=device,
        score_threshold=float(score_threshold),
        fp16=bool(fp16),
    )


def precompile_detection_engine(
    detection_model_name: str,
    detection_model_path: Path,
    batch_size: int,
    device: torch.device,
    fp16: bool,
) -> None:
    if not (is_nvidia_device(device) or is_amd_device(device)):
        return
    det_name = coerce_detection_model_name(detection_model_name)
    if is_rfdetr_model(det_name):
        from jasna.mosaic.rfdetr import compile_rfdetr_engine

        compile_rfdetr_engine(detection_model_path, device, batch_size=int(batch_size), fp16=bool(fp16))
    elif is_yolo_model(det_name) and is_nvidia_device(device):
        from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine
        from jasna.mosaic.yolo import YoloMosaicDetectionModel

        compile_yolo_to_tensorrt_engine(
            detection_model_path,
            batch=int(batch_size),
            fp16=bool(fp16) and (device.type == "cuda"),
            imgsz=YoloMosaicDetectionModel.DEFAULT_IMGSZ,
            device=device,
        )
