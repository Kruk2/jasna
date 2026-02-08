from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from jasna.gui.models import AppSettings


@dataclass(frozen=True)
class EngineRequirement:
    key: str
    label: str
    paths: tuple[Path, ...]
    exists: bool
    missing_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BasicvsrppRiskCheck:
    is_risky: bool
    vram_gb: float
    approx_safe_max_clip_length: int
    requested_clip_length: int


@dataclass(frozen=True)
class EnginePreflightResult:
    requirements: tuple[EngineRequirement, ...]
    should_warn_first_run_slow: bool
    basicvsrpp_risk: BasicvsrppRiskCheck

    @property
    def missing(self) -> tuple[EngineRequirement, ...]:
        return tuple(r for r in self.requirements if not r.exists)


def _detection_weights_path(settings: AppSettings) -> Path:
    from jasna.mosaic.detection_registry import coerce_detection_model_name, detection_model_weights_path

    return detection_model_weights_path(coerce_detection_model_name(str(settings.detection_model)))


def run_engine_preflight(settings: AppSettings) -> EnginePreflightResult:
    from jasna.trt import get_onnx_tensorrt_engine_path
    from jasna.mosaic.detection_registry import RFDETR_MODEL_NAMES, YOLO_MODEL_NAMES, coerce_detection_model_name
    from jasna.mosaic.yolo_tensorrt_compilation import get_yolo_tensorrt_engine_path
    from jasna.restorer.basicvrspp_tenorrt_compilation import (
        SMALL_TRT_CLIP_LENGTH,
        SMALL_TRT_CLIP_LENGTH_TRIGGER,
        _get_approx_max_tensorrt_clip_length,
        get_compiled_mosaic_restoration_model_path_for_clip,
    )
    from jasna.restorer.swin2sr_tensorrt_compilation import get_compiled_swin2sr_engine_path

    device = torch.device("cuda:0")

    reqs: list[EngineRequirement] = []

    det_name = coerce_detection_model_name(str(settings.detection_model))
    det_weights = _detection_weights_path(settings)
    if det_name in RFDETR_MODEL_NAMES:
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
    elif det_name in YOLO_MODEL_NAMES:
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
    basic_missing = False
    basic_paths: list[Path] = []
    basic_main_missing = False
    if bool(settings.compile_basicvsrpp):
        main_engine = Path(
            get_compiled_mosaic_restoration_model_path_for_clip(
                checkpoint_path=str(restoration_model_path),
                clip_length=int(settings.max_clip_size),
                fp16=bool(settings.fp16_mode),
            )
        )
        basic_paths.append(main_engine)
        basic_main_missing = not main_engine.is_file()
        basic_missing = basic_missing or basic_main_missing

        if int(settings.max_clip_size) > int(SMALL_TRT_CLIP_LENGTH_TRIGGER):
            small_engine = Path(
                get_compiled_mosaic_restoration_model_path_for_clip(
                    checkpoint_path=str(restoration_model_path),
                    clip_length=int(SMALL_TRT_CLIP_LENGTH),
                    fp16=bool(settings.fp16_mode),
                )
            )
            basic_paths.append(small_engine)
            basic_missing = basic_missing or (not small_engine.is_file())

        missing_paths = tuple(p for p in basic_paths if not p.is_file())
        reqs.append(
            EngineRequirement(
                key="basicvsrpp",
                label="BasicVSR++ (restoration)",
                paths=tuple(basic_paths),
                exists=not basic_missing,
                missing_paths=missing_paths,
            )
        )

    swin_missing = False
    if str(settings.secondary_restoration).lower() == "swin2sr" and bool(settings.swin2sr_tensorrt) and bool(settings.fp16_mode):
        swin_engine = Path(
            get_compiled_swin2sr_engine_path(
                engine_dir=str(Path("model_weights")),
                batch_size=int(settings.swin2sr_batch_size),
                fp16=True,
            )
        )
        swin_missing = not swin_engine.is_file()
        reqs.append(
            EngineRequirement(
                key="swin2sr",
                label="Swin2SR (secondary)",
                paths=(swin_engine,),
                exists=not swin_missing,
                missing_paths=() if not swin_missing else (swin_engine,),
            )
        )

    should_warn = any(not r.exists for r in reqs)

    vram_gb, approx_max_clip_length = _get_approx_max_tensorrt_clip_length(device)
    requested_clip = int(settings.max_clip_size)
    risky = (
        bool(settings.compile_basicvsrpp)
        and bool(basic_main_missing)
        and bool(settings.fp16_mode)
        and int(approx_max_clip_length) > 0
        and requested_clip > int(approx_max_clip_length)
    )

    return EnginePreflightResult(
        requirements=tuple(reqs),
        should_warn_first_run_slow=bool(should_warn),
        basicvsrpp_risk=BasicvsrppRiskCheck(
            is_risky=bool(risky),
            vram_gb=float(vram_gb),
            approx_safe_max_clip_length=int(approx_max_clip_length),
            requested_clip_length=requested_clip,
        ),
    )

