"""Lightweight engine-path helpers — no torch / tensorrt imports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from jasna._frozen import is_frozen


def model_weights_dir() -> Path:
    """Resolve the ``model_weights`` directory.

    When running as a frozen executable, models are bundled next to the executable.
    In a dev / source checkout we fall back to a CWD-relative path so existing dev
    workflows keep working.
    """
    if is_frozen():
        return Path(sys.executable).parent / "model_weights"
    return Path("model_weights")


def engine_system_suffix() -> str:
    return ".win" if os.name == "nt" else ".linux"


def engine_precision_name(*, fp16: bool) -> str:
    return "fp16" if bool(fp16) else "fp32"


def get_onnx_tensorrt_engine_path(
    onnx_path: str | Path,
    *,
    batch_size: int | None = None,
    fp16: bool = True,
) -> Path:
    onnx_path = Path(onnx_path)
    suffix = ""
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        suffix += f".bs{batch_size}"
    suffix += ".fp16" if bool(fp16) else ""
    suffix += engine_system_suffix()
    suffix += ".engine"
    return onnx_path.with_suffix(suffix)


def get_yolo_tensorrt_engine_path(model_path: str | Path, *, fp16: bool) -> Path:
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt YOLO checkpoint, got: {model_path}")
    onnx_path = model_path.with_suffix(".onnx")
    return get_onnx_tensorrt_engine_path(onnx_path, batch_size=None, fp16=bool(fp16))


UNET4X_ONNX_PATH = model_weights_dir() / "unet-4x.onnx"
UNET4X_ONNX_ENC_PATH = model_weights_dir() / "unet-4x.onnx.enc"

SD15_HF_REPO = "Kruk2/sd-15-jav"
SD15_DIR = model_weights_dir() / "sd-15-jav"
SD15_CKPT_PATH = SD15_DIR / "sd15-200000.ckpt"
SD15_CKPT_ENC_PATH = SD15_DIR / "sd15-200000.ckpt.enc"


def get_unet4x_engine_path(onnx_path: str | Path | None = None, fp16: bool = True) -> Path:
    if onnx_path is None:
        onnx_path = UNET4X_ONNX_PATH
    return get_onnx_tensorrt_engine_path(onnx_path, batch_size=None, fp16=fp16)


def get_unet4x_encrypted_engine_path(fp16: bool = True) -> Path:
    plain = get_unet4x_engine_path(UNET4X_ONNX_PATH, fp16=fp16)
    return plain.parent / (plain.name + ".enc")


def unet4x_plaintext_available() -> bool:
    return UNET4X_ONNX_PATH.exists() and not is_frozen()


BASICVSRPP_DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")


def _basicvsrpp_sub_engine_dir(model_weights_path: str) -> str:
    stem = os.path.splitext(os.path.basename(model_weights_path))[0]
    return os.path.join(os.path.dirname(model_weights_path), f"{stem}_sub_engines")


def get_basicvsrpp_sub_engine_paths(
    model_weights_path: str, fp16: bool, max_clip_size: int = 60,
) -> dict[str, str]:
    engine_dir = _basicvsrpp_sub_engine_dir(model_weights_path)
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    paths: dict[str, str] = {}
    for d in BASICVSRPP_DIRECTIONS:
        paths[f"loop_body_{d}"] = os.path.join(engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine")
    paths["preprocess"] = os.path.join(engine_dir, f"preprocess_b{max_clip_size}.trt_{prec}{suf}.engine")
    paths["upsample"] = os.path.join(engine_dir, f"upsample_dyn_b{max_clip_size}.trt_{prec}{suf}.engine")
    return paths


def all_basicvsrpp_sub_engines_exist(
    model_weights_path: str, fp16: bool, max_clip_size: int = 60,
) -> bool:
    return all(os.path.isfile(p) for p in get_basicvsrpp_sub_engine_paths(model_weights_path, fp16, max_clip_size).values())
