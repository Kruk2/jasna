from __future__ import annotations

import gc
import logging
import os
import sys

import torch

from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan

logger = logging.getLogger(__name__)

SMALL_TRT_CLIP_LENGTH = 10
SMALL_TRT_CLIP_LENGTH_TRIGGER = 30


def _approx_max_tensorrt_clip_length(vram_gb: float) -> int:
    if vram_gb < 4:
        return 0
    if vram_gb < 6:
        return 30
    if vram_gb < 8:
        return 60
    if vram_gb < 12:
        return 90
    if vram_gb < 16:
        return 120
    if vram_gb < 24:
        return 180
    if vram_gb < 32:
        return 240
    return 300


def get_gpu_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    props = torch.cuda.get_device_properties(idx)
    return float(props.total_memory) / (1024**3)


def _get_approx_max_tensorrt_clip_length(device: torch.device) -> tuple[float, int]:
    vram_gb = get_gpu_vram_gb(device)
    return vram_gb, _approx_max_tensorrt_clip_length(vram_gb)


def _compile_basicvsrpp_model(
    model: BasicVSRPlusPlusGan,
    device: torch.device,
    dtype: torch.dtype,
    output_path: str,
    max_clip_size: int,
) -> str:
    import psutil
    import torch_tensorrt  # type: ignore[import-not-found]

    workspace_size = int(psutil.virtual_memory().available * 0.8)
    inp = torch.randn(1, max_clip_size, 3, 256, 256, dtype=dtype, device=device)

    with torch_tensorrt.logging.info():
        print(
            f"Compiling BasicVSR++ model (TensorRT workspace_size={workspace_size / (1024 ** 3):.2f} GB). "
            "For large clip length > 100 this can take even few hours."
        )
        trt_gm = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[inp],
            min_block_size=1,
            workspace_size=workspace_size,
            enabled_precisions={dtype},
            use_fp32_acc=False,
            use_explicit_typing=False,
            sparse_weights=False,
            optimization_level=3,
            hardware_compatible=False,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
            truncate_double=True,
        )

    torch_tensorrt.save(trt_gm, output_path, inputs=[inp])
    del trt_gm
    del inp
    return output_path


def _get_compiled_mosaic_restoration_model_path(
    mosaic_restoration_model_path: str,
    clip_length: int,
    fp16: bool,
) -> str:
    precision = "fp16" if fp16 else "fp32"
    output_dir = os.path.dirname(mosaic_restoration_model_path)
    stem = os.path.splitext(os.path.basename(mosaic_restoration_model_path))[0]
    return os.path.join(output_dir, f"{stem}_clip{clip_length}.trt_{precision}.engine")


def get_compiled_mosaic_restoration_model_path_for_clip(
    checkpoint_path: str,
    clip_length: int,
    fp16: bool,
) -> str:
    if checkpoint_path.endswith(".engine"):
        raise ValueError("checkpoint_path must be a .pth/.pt path, not a .engine path")
    return _get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=checkpoint_path,
        clip_length=int(clip_length),
        fp16=bool(fp16),
    )


def load_engine(checkpoint_path: str, device: torch.device) -> BasicVSRPlusPlusGan:
    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    import torch_tensorrt  # type: ignore[import-not-found]  # noqa: F401

    logger.info("Loading TensorRT export from %s", checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        trt_module = torch.export.load(f).module()
        return trt_module.to(device)


def compile_mosaic_restoration_model(
    mosaic_restoration_model_path: str,
    clip_length: int,
    device: str | torch.device,
    fp16: bool,
    mosaic_restoration_config_path: str | None = None,
    interactive: bool = True,
) -> str:
    if isinstance(device, str):
        device = torch.device(device)

    output_path = _get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        clip_length=clip_length,
        fp16=fp16,
    )
    output_path_small = _get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        clip_length=SMALL_TRT_CLIP_LENGTH,
        fp16=fp16,
    )
    requested_exists = os.path.isfile(output_path)
    small_exists = os.path.isfile(output_path_small)
    should_use_small_engine = int(clip_length) > SMALL_TRT_CLIP_LENGTH_TRIGGER
    if requested_exists and (small_exists or not should_use_small_engine):
        return output_path

    if device.type != "cuda":
        return output_path if requested_exists else mosaic_restoration_model_path

    vram_gb, approx_max_clip_length = _get_approx_max_tensorrt_clip_length(device)
    if approx_max_clip_length == 0:
        print("Skipping compilation due to low VRAM (< 4 GB). Pass --no-compile-basicvsrpp to suppress this message.")
        return output_path if requested_exists else mosaic_restoration_model_path

    if not fp16:
        print(
            "Skipping compilation due to FP32 compilation is not recommended for TensorRT. "
            "Consider using FP16 instead to save on VRAM and have faster execution times."
        )
        return output_path if requested_exists else mosaic_restoration_model_path

    should_compile_requested = not requested_exists
    if int(clip_length) > approx_max_clip_length and should_compile_requested:
        if interactive and sys.stdin.isatty():
            print(
                "\n".join(
                    [
                        f"Requested TensorRT clip length {int(clip_length)}, but GPU VRAM is ~{vram_gb:.1f} GB.",
                        f"Approx safe max is {approx_max_clip_length} frames (rule of thumb: ~2.5 GB per +30 frames).",
                        "",
                        "Large clip lengths can:",
                        "- require significantly more VRAM (compilation may OOM)",
                        "- take much longer to compile",
                        "- on videos with poor mosaic detection the performance may be degraded",
                        "",
                        "Continue compilation anyway? [y/N] ",
                    ]
                ),
                end="",
                flush=True,
            )
            if input().strip().lower() not in {"y", "yes"}:
                should_compile_requested = False
        else:
            print(
                f"Skipping compilation due to low VRAM for requested clip length {int(clip_length)} "
                f"(VRAM ~{vram_gb:.1f} GB, approx safe max {approx_max_clip_length}). "
                "Large clip lengths can require significantly more VRAM, take much longer to compile, and may degrade performance on videos with poor mosaic detection."
            )
            should_compile_requested = False

    from jasna.models.basicvsrpp.inference import load_model

    dtype = torch.float16 if fp16 else torch.float32
    should_compile_small = (
        should_use_small_engine and (not small_exists) and SMALL_TRT_CLIP_LENGTH <= approx_max_clip_length
    )
    if should_compile_small or should_compile_requested:
        model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16)
        if should_compile_small:
            _compile_basicvsrpp_model(model, device, dtype, output_path_small, SMALL_TRT_CLIP_LENGTH)
        if should_compile_requested and output_path != output_path_small:
            _compile_basicvsrpp_model(model, device, dtype, output_path, int(clip_length))
        del model

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path if os.path.isfile(output_path) else mosaic_restoration_model_path


def basicvsrpp_startup_policy(
    *,
    restoration_model_path: str,
    max_clip_size: int,
    device: torch.device,
    fp16: bool,
    compile_basicvsrpp: bool,
) -> bool:
    """
    Returns:
        use_tensorrt: whether runtime should attempt TensorRT execution.

    Policy:
    - If `compile_basicvsrpp` is True: use TRT if engines exist, otherwise try to compile them.
    - If `compile_basicvsrpp` is False: never use TRT (even if engines exist on disk).
    """
    restoration_model_path = str(restoration_model_path)
    max_clip_size = int(max_clip_size)
    fp16 = bool(fp16)

    if restoration_model_path.endswith(".engine"):
        if not bool(compile_basicvsrpp):
            raise ValueError("Engine path requires --compile-basicvsrpp (cannot fall back from .engine to .pth)")
        return True

    if bool(compile_basicvsrpp):
        compile_mosaic_restoration_model(
            mosaic_restoration_model_path=restoration_model_path,
            clip_length=max_clip_size,
            device=device,
            fp16=fp16,
            mosaic_restoration_config_path=None,
            interactive=True,
        )
        return True

    return False

