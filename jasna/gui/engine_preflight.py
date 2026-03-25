from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jasna.gui.models import AppSettings


@dataclass(frozen=True)
class EngineRequirement:
    key: str
    label: str
    paths: tuple[Path, ...]
    exists: bool
    missing_paths: tuple[Path, ...]


@dataclass(frozen=True)
class EnginePreflightResult:
    requirements: tuple[EngineRequirement, ...]
    should_warn_first_run_slow: bool

    @property
    def missing(self) -> tuple[EngineRequirement, ...]:
        return tuple(r for r in self.requirements if not r.exists)


def _detection_weights_path(settings: AppSettings) -> Path:
    from jasna.mosaic.detection_registry import coerce_detection_model_name, detection_model_weights_path

    return detection_model_weights_path(coerce_detection_model_name(str(settings.detection_model)))


def run_engine_preflight(settings: AppSettings) -> EnginePreflightResult:
    from jasna.engine_paths import (
        get_basicvsrpp_sub_engine_paths,
        get_onnx_tensorrt_engine_path,
        get_unet4x_engine_path,
        get_yolo_tensorrt_engine_path,
    )
    from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model, coerce_detection_model_name

    reqs: list[EngineRequirement] = []

    det_name = coerce_detection_model_name(str(settings.detection_model))
    det_weights = _detection_weights_path(settings)
    if is_rfdetr_model(det_name):
        det_engine = get_onnx_tensorrt_engine_path(
            det_weights,
            batch_size=int(settings.batch_size),
            fp16=bool(settings.fp16_mode),
        )
        det_exists = det_engine.is_file()
        reqs.append(
            EngineRequirement(
                key="rfdetr",
                label=f"RF-DETR ({det_weights.name})",
                paths=(det_engine,),
                exists=det_exists,
                missing_paths=() if det_exists else (det_engine,),
            )
        )
    elif is_yolo_model(det_name):
        det_engine = get_yolo_tensorrt_engine_path(det_weights, fp16=bool(settings.fp16_mode))
        det_exists = det_engine.is_file()
        reqs.append(
            EngineRequirement(
                key="yolo",
                label=f"YOLO ({det_weights.name})",
                paths=(det_engine,),
                exists=det_exists,
                missing_paths=() if det_exists else (det_engine,),
            )
        )

    restoration_model_path = Path("model_weights") / "lada_mosaic_restoration_model_generic_v1.2.pth"
    if bool(settings.compile_basicvsrpp):
        sub_paths = get_basicvsrpp_sub_engine_paths(str(restoration_model_path), bool(settings.fp16_mode), int(settings.max_clip_size))
        all_engine_paths = tuple(Path(p) for p in sub_paths.values())
        missing_paths = tuple(p for p in all_engine_paths if not p.is_file())
        reqs.append(
            EngineRequirement(
                key="basicvsrpp",
                label="BasicVSR++ (restoration sub-engines)",
                paths=all_engine_paths,
                exists=len(missing_paths) == 0,
                missing_paths=missing_paths,
            )
        )

    if settings.secondary_restoration == "unet-4x":
        unet_engine = get_unet4x_engine_path(fp16=bool(settings.fp16_mode))
        unet_exists = unet_engine.is_file()
        reqs.append(
            EngineRequirement(
                key="unet_4x",
                label="UNet 4x (secondary restoration)",
                paths=(unet_engine,),
                exists=unet_exists,
                missing_paths=() if unet_exists else (unet_engine,),
            )
        )

    should_warn = any(not r.exists for r in reqs)

    return EnginePreflightResult(
        requirements=tuple(reqs),
        should_warn_first_run_slow=bool(should_warn),
    )

