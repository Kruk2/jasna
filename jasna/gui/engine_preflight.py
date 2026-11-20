from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jasna.gui.models import AppSettings


@dataclass(frozen=True)
class EngineRequirement:
    key: str
    paths: tuple[Path, ...]
    exists: bool


@dataclass(frozen=True)
class EnginePreflightResult:
    requirements: tuple[EngineRequirement, ...]

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


def _engine_on_disk(path: Path, gpu: str) -> bool:
    """Whether the engine at ``path`` exists, or its legacy file the run renames to it."""
    from jasna.engine_paths import legacy_engine_path

    return path.is_file() or legacy_engine_path(path, gpu).is_file()


def run_engine_preflight(settings: AppSettings) -> EnginePreflightResult:
    import torch

    from jasna.accelerator import cuda_arch_name, is_nvidia_device
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

    device = torch.device("cuda:0")
    if not is_nvidia_device(device) or not torch.cuda.is_available():
        # Only NVIDIA GPUs run TensorRT engines; AMD runs every model through PyTorch.
        return EnginePreflightResult(requirements=())
    gpu = cuda_arch_name(device)
    fp16 = bool(settings.fp16_mode)
    reqs: list[EngineRequirement] = []

    det_name = coerce_detection_model_name(str(settings.detection_model))
    for key, matches in (("rfdetr", is_rfdetr_model), ("yolo", is_yolo_model)):
        if matches(det_name):
            det_engine = detection_engine_path(
                det_name, _detection_weights_path(settings), default_detection_batch_size(device), fp16=fp16, gpu=gpu
            )
            reqs.append(EngineRequirement(key=key, paths=(det_engine,), exists=_engine_on_disk(det_engine, gpu)))

    standard_model = settings.restoration_model == "basicvsrpp"
    if standard_model and bool(settings.compile_basicvsrpp):
        sub_paths = get_basicvsrpp_sub_engine_paths(str(default_restoration_model_path("basicvsrpp")), fp16, gpu)
        all_engine_paths = tuple(Path(p) for p in sub_paths.values())
        reqs.append(
            EngineRequirement(
                key="basicvsrpp",
                paths=all_engine_paths,
                exists=all(_engine_on_disk(p, gpu) for p in all_engine_paths),
            )
        )

    if standard_model and settings.secondary_restoration == "unet-4x":
        unet_engine = expected_unet4x_engine_path(fp16=fp16, gpu=gpu)
        reqs.append(EngineRequirement(key="unet_4x", paths=(unet_engine,), exists=_engine_on_disk(unet_engine, gpu)))

    return EnginePreflightResult(requirements=tuple(reqs))
