from __future__ import annotations

import gc
import logging
import os

import torch

from jasna.restorer.basicvsrpp_sub_engines import (
    all_sub_engines_exist,
    compile_basicvsrpp_sub_engines,
    get_sub_engine_paths,
)

logger = logging.getLogger(__name__)


def get_gpu_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    props = torch.cuda.get_device_properties(idx)
    return float(props.total_memory) / (1024**3)


def compile_mosaic_restoration_model(
    mosaic_restoration_model_path: str,
    device: str | torch.device,
    fp16: bool,
    mosaic_restoration_config_path: str | None = None,
    max_clip_size: int = 60,
    optimization_level: int = 5,
) -> bool:
    """Compile BasicVSR++ sub-engines (backbone × 4 + upsample × 1).

    Returns True if all sub-engines exist after this call.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if all_sub_engines_exist(mosaic_restoration_model_path, fp16, max_clip_size):
        return True

    if device.type != "cuda":
        return False

    vram_gb = get_gpu_vram_gb(device)
    if vram_gb < 4:
        msg = "Skipping TRT compilation: GPU VRAM < 4 GB."
        logger.info("%s", msg)
        return False

    if not fp16:
        msg = (
            "Skipping TRT compilation: FP32 is not recommended for TensorRT. "
            "Consider using FP16 instead."
        )
        logger.info("%s", msg)
        return False

    from jasna.models.basicvsrpp.inference import load_model

    model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16)
    compile_basicvsrpp_sub_engines(
        model=model,
        device=device,
        fp16=fp16,
        model_weights_path=mosaic_restoration_model_path,
        max_clip_size=max_clip_size,
        optimization_level=optimization_level,
    )
    del model

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_sub_engines_exist(mosaic_restoration_model_path, fp16, max_clip_size)


def basicvsrpp_startup_policy(
    *,
    restoration_model_path: str,
    device: torch.device,
    fp16: bool,
    compile_basicvsrpp: bool,
    max_clip_size: int = 60,
    optimization_level: int = 5,
) -> bool:
    """Returns whether runtime should attempt TensorRT execution.

    Policy:
    - If ``compile_basicvsrpp`` is True: use TRT if sub-engines exist,
      otherwise try to compile them.
    - If ``compile_basicvsrpp`` is False: never use TRT.
    """
    restoration_model_path = str(restoration_model_path)
    fp16 = bool(fp16)

    if not bool(compile_basicvsrpp):
        return False

    if all_sub_engines_exist(restoration_model_path, fp16, max_clip_size):
        return True

    return compile_mosaic_restoration_model(
        mosaic_restoration_model_path=restoration_model_path,
        device=device,
        fp16=fp16,
        max_clip_size=max_clip_size,
        optimization_level=optimization_level,
    )

