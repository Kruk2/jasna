"""Compile TensorRT engines in a subprocess to guarantee full VRAM release."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30 * 60


@dataclass
class EngineCompilationRequest:
    device: str
    fp16: bool

    basicvsrpp: bool = False
    basicvsrpp_model_path: str = ""
    basicvsrpp_max_clip_size: int = 60

    detection: bool = False
    detection_model_name: str = ""
    detection_model_path: str = ""
    detection_batch_size: int = 4

    unet4x: bool = False

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> EngineCompilationRequest:
        return EngineCompilationRequest(**json.loads(s))


@dataclass
class EngineCompilationResult:
    use_basicvsrpp_tensorrt: bool = False


def _basicvsrpp_engines_exist(model_path: str, fp16: bool, max_clip_size: int) -> bool:
    from jasna.engine_paths import all_basicvsrpp_sub_engines_exist
    return all_basicvsrpp_sub_engines_exist(model_path, fp16, max_clip_size)


def _detection_engine_exists(detection_model_name: str, detection_model_path: str, batch_size: int, fp16: bool) -> bool:
    from jasna.engine_paths import get_onnx_tensorrt_engine_path, get_yolo_tensorrt_engine_path

    if detection_model_name.startswith("rfdetr"):
        return get_onnx_tensorrt_engine_path(detection_model_path, batch_size=batch_size, fp16=fp16).exists()
    if detection_model_name.startswith("lada-yolo"):
        return get_yolo_tensorrt_engine_path(detection_model_path, fp16=fp16).exists()
    return True


def _unet4x_engine_exists(fp16: bool) -> bool:
    from jasna.engine_paths import get_unet4x_engine_path
    return get_unet4x_engine_path(fp16=fp16).exists()  # uses default UNET4X_ONNX_PATH


def ensure_engines_compiled(
    req: EngineCompilationRequest,
    log_callback: typing.Callable[[str], None] | None = None,
) -> EngineCompilationResult:
    result = EngineCompilationResult()

    need_basicvsrpp = req.basicvsrpp and req.fp16 and not _basicvsrpp_engines_exist(
        req.basicvsrpp_model_path, req.fp16, req.basicvsrpp_max_clip_size
    )
    need_detection = req.detection and not _detection_engine_exists(
        req.detection_model_name, req.detection_model_path, req.detection_batch_size, req.fp16
    )
    need_unet4x = req.unet4x and not _unet4x_engine_exists(req.fp16)

    if req.basicvsrpp:
        if not req.fp16:
            result.use_basicvsrpp_tensorrt = False
        elif not need_basicvsrpp:
            result.use_basicvsrpp_tensorrt = True

    if not (need_basicvsrpp or need_detection or need_unet4x):
        return result

    logger.info("Spawning engine compilation subprocess...")
    if log_callback:
        log_callback("Compiling TensorRT engines (this may take several minutes)...")

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--compile-engines", req.to_json()]
    else:
        cmd = [sys.executable, "-m", "jasna.engine_compiler", req.to_json()]

    kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **kwargs)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        if line:
            if log_callback:
                log_callback(line)
                logger.debug("[compiler] %s", line)
            else:
                logger.info("[compiler] %s", line)
    returncode = proc.wait(timeout=_TIMEOUT_SECONDS)

    if returncode != 0:
        raise RuntimeError(f"Engine compilation subprocess failed (exit code {returncode})")

    if req.basicvsrpp:
        result.use_basicvsrpp_tensorrt = _basicvsrpp_engines_exist(
            req.basicvsrpp_model_path, req.fp16, req.basicvsrpp_max_clip_size
        )

    return result


def _subprocess_compile(req: EngineCompilationRequest) -> None:
    import logging as _logging
    import warnings
    warnings.filterwarnings("ignore")
    _logging.disable(_logging.WARNING)

    import torch

    device = torch.device(req.device)

    if req.basicvsrpp and req.fp16 and not _basicvsrpp_engines_exist(
        req.basicvsrpp_model_path, req.fp16, req.basicvsrpp_max_clip_size
    ):
        from jasna.restorer.basicvrspp_tenorrt_compilation import compile_mosaic_restoration_model
        print(f"Compiling BasicVSR++ sub-engines (max_clip_size={req.basicvsrpp_max_clip_size})...")
        compile_mosaic_restoration_model(
            mosaic_restoration_model_path=req.basicvsrpp_model_path,
            device=device,
            fp16=req.fp16,
            max_clip_size=req.basicvsrpp_max_clip_size,
        )
        print("BasicVSR++ sub-engines compiled.")

    if req.detection and not _detection_engine_exists(
        req.detection_model_name, req.detection_model_path, req.detection_batch_size, req.fp16
    ):
        from jasna.mosaic.detection_registry import precompile_detection_engine
        print(f"Compiling detection engine ({req.detection_model_name})...")
        precompile_detection_engine(
            detection_model_name=req.detection_model_name,
            detection_model_path=Path(req.detection_model_path),
            batch_size=req.detection_batch_size,
            device=device,
            fp16=req.fp16,
        )
        print("Detection engine compiled.")

    if req.unet4x and not _unet4x_engine_exists(req.fp16):
        from jasna.engine_paths import UNET4X_ONNX_PATH
        from jasna.restorer.unet4x_secondary_restorer import compile_unet4x_engine
        print("Compiling Unet4x engine...")
        compile_unet4x_engine(UNET4X_ONNX_PATH, device, fp16=req.fp16)
        print("Unet4x engine compiled.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m jasna.engine_compiler <json_request>", file=sys.stderr)
        sys.exit(1)
    req = EngineCompilationRequest.from_json(sys.argv[1])
    _subprocess_compile(req)
