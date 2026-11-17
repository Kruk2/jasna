from __future__ import annotations

from collections.abc import Sequence
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


def first_run_warning(missing_keys: Sequence[str]) -> str:
    """The note shown before a run that first has to build the engines named by ``missing_keys``."""
    from jasna.gui.locales import t

    missing_lines = "\n".join(f"- {t(f'engine_name_{key}')}" for key in missing_keys)
    return f"{t('engine_first_run_body')}\n\n{t('engine_first_run_missing')}\n{missing_lines}"


def _detection_weights_path(settings: AppSettings) -> Path:
    from jasna.mosaic.detection_registry import coerce_detection_model_name, detection_model_weights_path

    return detection_model_weights_path(coerce_detection_model_name(str(settings.detection_model)))


def run_engine_preflight(settings: AppSettings) -> EnginePreflightResult:
    import torch

    from jasna.accelerator import is_amd_device
    from jasna.engine_paths import (
        default_restoration_model_path,
        expected_unet4x_engine_path,
        get_basicvsrpp_sub_engine_paths,
    )
    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        default_detection_batch_size,
        detection_engine_path,
        is_rfdetr_model,
        is_yolo_model,
    )

    reqs: list[EngineRequirement] = []
    device = torch.device("cuda:0")
    amd = is_amd_device(device)

    det_name = coerce_detection_model_name(str(settings.detection_model))
    det_weights = _detection_weights_path(settings)
    if is_rfdetr_model(det_name):
        if amd:
            # AMD runs RF-DETR through the rfdetr torch model; no precompiled
            # engine/cache artifact to check or build.
            det_engine = None
            det_exists = True
        else:
            det_engine = detection_engine_path(
                det_name, det_weights, default_detection_batch_size(device), fp16=bool(settings.fp16_mode)
            )
            det_exists = det_engine.is_file()
        if det_engine is not None:
            reqs.append(
                EngineRequirement(
                    key="rfdetr",
                    label=f"RF-DETR ({det_weights.name})",
                    paths=(det_engine,),
                    exists=det_exists,
                    missing_paths=() if det_exists else (det_engine,),
                )
            )
    elif is_yolo_model(det_name) and not amd:
        det_engine = detection_engine_path(
            det_name, det_weights, default_detection_batch_size(device), fp16=bool(settings.fp16_mode)
        )
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

    standard_model = settings.restoration_model == "basicvsrpp"
    restoration_model_path = default_restoration_model_path("basicvsrpp")
    if standard_model and bool(settings.compile_basicvsrpp) and not amd:
        sub_paths = get_basicvsrpp_sub_engine_paths(str(restoration_model_path), bool(settings.fp16_mode))
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

    if standard_model and settings.secondary_restoration == "unet-4x" and not amd:
        unet_engine = expected_unet4x_engine_path(fp16=bool(settings.fp16_mode))
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
