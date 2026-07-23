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


@dataclass(frozen=True)
class RfDetrModelConfig:
    resolution: int
    score_threshold: float
    fixed_batch_size: int | None
    torch_variant: str | None

    @property
    def dynamic_batch(self) -> bool:
        return self.fixed_batch_size is None

    def engine_batch_size(self, requested_batch_size: int) -> int:
        requested_batch_size = int(requested_batch_size)
        if requested_batch_size <= 0:
            raise ValueError(
                f"requested_batch_size must be > 0, got {requested_batch_size}"
            )
        if self.fixed_batch_size is not None:
            return self.fixed_batch_size
        return requested_batch_size


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

RFDETR_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "rfdetr-v2",
        "rfdetr-v3",
        "rfdetr-v4",
        "rfdetr-v5",
        "rfdetr-v6",
        "rfdetr-v6-large",
        "rfdetr-vr-v1",
    }
)
YOLO_MODEL_NAMES: frozenset[str] = frozenset(
    name
    for name, spec in DETECTION_MODEL_SPECS.items()
    if spec.backend == "yolo"
)

DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v6"

RFDETR_MODEL_CONFIGS: dict[str, RfDetrModelConfig] = {
    "rfdetr-v5": RfDetrModelConfig(768, 0.25, 4, None),
    "rfdetr-v6": RfDetrModelConfig(576, 0.35, None, "medium"),
    "rfdetr-v6-large": RfDetrModelConfig(768, 0.40, None, "large"),
    "rfdetr-vr-v1": RfDetrModelConfig(768, 0.40, None, "large"),
}
_RFDETR_FALLBACK_CONFIG = RfDetrModelConfig(768, 0.25, None, None)

YOLO_MODEL_FILES: dict[str, str] = {
    name: spec.filename
    for name, spec in DETECTION_MODEL_SPECS.items()
    if spec.backend == "yolo"
}

_RFDETR_PATTERN = re.compile(r"^rfdetr-.+$")


def rfdetr_weights_suffix() -> str:
    """AMD runs RF-DETR from the trained torch checkpoint (``.pt``); NVIDIA
    compiles the ONNX export to TensorRT (``.onnx``). Resolved from the active
    accelerator vendor — release binaries are vendor-specific."""
    return ".pt" if is_amd_device() else ".onnx"


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
            f"{normalized}{rfdetr_weights_suffix()}",
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
        rfdetr_suffix = rfdetr_weights_suffix()
        for f in weights_dir.iterdir():
            if f.suffix == rfdetr_suffix and is_rfdetr_model(f.stem):
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


def rfdetr_model_config(name: str) -> RfDetrModelConfig:
    return RFDETR_MODEL_CONFIGS.get(coerce_detection_model_name(name), _RFDETR_FALLBACK_CONFIG)


def recommended_score_threshold(name: str) -> float:
    coerced = coerce_detection_model_name(name)
    if is_rfdetr_model(coerced):
        return rfdetr_model_config(coerced).score_threshold
    return 0.25


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

        config = rfdetr_model_config(det_name)
        return RfDetrMosaicDetectionModel(
            weights_path=detection_model_path,
            batch_size=config.engine_batch_size(batch_size),
            device=device,
            resolution=config.resolution,
            dynamic_batch=config.dynamic_batch,
            score_threshold=float(score_threshold),
            fp16=bool(fp16),
            torch_variant=config.torch_variant,
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

        config = rfdetr_model_config(det_name)
        compile_rfdetr_engine(
            detection_model_path,
            device,
            batch_size=config.engine_batch_size(batch_size),
            resolution=config.resolution,
            dynamic_batch=config.dynamic_batch,
            fp16=bool(fp16),
        )
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
