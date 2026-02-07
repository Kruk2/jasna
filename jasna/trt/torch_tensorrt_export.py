from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def engine_system_suffix() -> str:
    return ".win" if os.name == "nt" else ".linux"


def engine_precision_name(*, fp16: bool) -> str:
    return "fp16" if bool(fp16) else "fp32"


def get_workspace_size_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().available * 0.8)


def load_torchtrt_export(*, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    import torch_tensorrt  # noqa: F401

    logger.info("Loading TensorRT export from %s", checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        trt_module = torch.export.load(f).module()
        return trt_module.to(device)


def compile_and_save_torchtrt_dynamo(
    *,
    module: torch.nn.Module,
    inputs: list[torch.Tensor],
    output_path: str,
    dtype: torch.dtype,
    workspace_size_bytes: int,
    message: str,
) -> str:
    import torch_tensorrt  # type: ignore[import-not-found]

    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    with torch_tensorrt.logging.errors():
        print(message)
        logger.info("%s", message)
        trt_gm = torch_tensorrt.compile(
            module,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            workspace_size=int(workspace_size_bytes),
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

    torch_tensorrt.save(trt_gm, output_path, inputs=inputs)
    del trt_gm
    return output_path

