from __future__ import annotations

import gc
import logging
import os

import torch

from jasna.trt.torch_tensorrt_export import (
    compile_and_save_torchtrt_dynamo,
    engine_precision_name,
    engine_system_suffix,
    get_workspace_size_bytes,
    load_torchtrt_export,
)

logger = logging.getLogger(__name__)

SWIN2SR_REPO_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
SWIN2SR_INPUT_SIZE = 256


def _get_compiled_swin2sr_engine_path(*, engine_dir: str, batch_size: int, fp16: bool) -> str:
    precision = engine_precision_name(fp16=fp16)
    system_name = engine_system_suffix()
    return os.path.join(engine_dir, f"swin2sr_bs{int(batch_size)}.trt_{precision}{system_name}.engine")


def get_compiled_swin2sr_engine_path(*, engine_dir: str, batch_size: int, fp16: bool) -> str:
    return _get_compiled_swin2sr_engine_path(engine_dir=str(engine_dir), batch_size=int(batch_size), fp16=bool(fp16))


def load_engine(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    return load_torchtrt_export(checkpoint_path=str(checkpoint_path), device=torch.device(device))


class _Swin2srTrtModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        return out.reconstruction


def _compile_swin2sr_model(
    *,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    output_path: str,
    batch_size: int,
) -> str:
    workspace_size = get_workspace_size_bytes()
    inp = torch.randn(int(batch_size), 3, SWIN2SR_INPUT_SIZE, SWIN2SR_INPUT_SIZE, dtype=dtype, device=device)

    message = (
        f"Compiling Swin2SR TensorRT engine (workspace_size={workspace_size / (1024 ** 3):.2f} GB). "
        "This can take a few minutes."
    )
    compile_and_save_torchtrt_dynamo(
        module=_Swin2srTrtModule(model),
        inputs=[inp],
        output_path=output_path,
        dtype=dtype,
        workspace_size_bytes=workspace_size,
        message=message,
    )
    del inp
    return output_path


def compile_swin2sr_engine(
    *,
    engine_dir: str,
    batch_size: int,
    device: str | torch.device,
    fp16: bool,
) -> str:
    if isinstance(device, str):
        device = torch.device(device)

    output_path = _get_compiled_swin2sr_engine_path(engine_dir=str(engine_dir), batch_size=int(batch_size), fp16=bool(fp16))
    if os.path.isfile(output_path):
        return output_path

    if device.type != "cuda":
        return output_path

    if not bool(fp16):
        return output_path

    os.makedirs(str(engine_dir), exist_ok=True)

    from transformers import Swin2SRForImageSuperResolution

    dtype = torch.float16
    model = Swin2SRForImageSuperResolution.from_pretrained(
        SWIN2SR_REPO_ID,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype)
    model.eval()

    _compile_swin2sr_model(
        model=model,
        device=device,
        dtype=dtype,
        output_path=output_path,
        batch_size=int(batch_size),
    )
    del model

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path

