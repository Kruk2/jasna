from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn import functional as F

from jasna.engine_paths import UNET4X_ONNX_PATH, get_unet4x_engine_path
from jasna.trt.trt_runner import TrtRunner

logger = logging.getLogger(__name__)

UNET4X_INPUT_SIZE = 256
UNET4X_OUTPUT_SIZE = 1024
UNET4X_HR_PREV_SIZE = 1152
UNET4X_COLOR_BLEND = 0.80


def rgb_color_transfer(
    ai_rgb: torch.Tensor,
    ref_rgb: torch.Tensor,
    blend: float = UNET4X_COLOR_BLEND,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    ai_mean = ai_rgb.mean(dim=(-3, -2), keepdim=True)
    ai_std = ai_rgb.std(dim=(-3, -2), keepdim=True).clamp_(min=1e-8)
    ref_mean = ref_rgb.mean(dim=(-3, -2), keepdim=True)
    ref_std = ref_rgb.std(dim=(-3, -2), keepdim=True)
    transferred = (ai_rgb - ai_mean) / ai_std * ref_std + ref_mean
    result = torch.lerp(ai_rgb, transferred, blend)
    result.clamp_(0.0, 1.0)
    if out is not None:
        out.copy_(result)
        return out
    return result


def compile_unet4x_engine(
    onnx_path: Path,
    device: torch.device,
    fp16: bool = True,
) -> Path:
    from jasna.trt import compile_onnx_to_tensorrt_engine
    return compile_onnx_to_tensorrt_engine(
        onnx_path,
        device,
        batch_size=None,
        fp16=bool(fp16),
        workspace_gb=20,
    )


class Unet4xSecondaryRestorer:
    name = "unet-4x"
    num_workers = 1
    preferred_queue_size = 2
    prefers_cpu_input = False

    def __init__(self, *, device: torch.device, fp16: bool = True) -> None:
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self._dtype = torch.float16 if self.fp16 else torch.float32

        self.engine_path = get_unet4x_engine_path(UNET4X_ONNX_PATH, fp16=self.fp16)
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"Unet4x engine not found: {self.engine_path}. "
                "Run engine compilation first via ensure_engines_compiled()."
            )
        self.runner = TrtRunner(
            self.engine_path,
            input_shapes={
                "lr_prev": (1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3),
                "lr_curr": (1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3),
                "hr_prev": (1, UNET4X_HR_PREV_SIZE, UNET4X_HR_PREV_SIZE, 3),
            },
            device=self.device,
        )
        logger.info(
            "Unet4xSecondaryRestorer loaded: %s (%dx%d -> %dx%d)",
            self.engine_path,
            UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE,
            UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE,
        )

        self._stream = torch.cuda.current_stream(self.device)

        self._g_lr_prev = torch.zeros(1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3, dtype=self._dtype, device=self.device)
        self._g_lr_curr = torch.zeros(1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3, dtype=self._dtype, device=self.device)
        self._g_hr_prev = torch.zeros(1, UNET4X_HR_PREV_SIZE, UNET4X_HR_PREV_SIZE, 3, dtype=self._dtype, device=self.device)
        self._g_ref_up = torch.empty(1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=torch.float32, device=self.device)
        self._g_result = torch.empty(1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=torch.float32, device=self.device)

        self.runner.context.set_tensor_address("lr_prev", self._g_lr_prev.data_ptr())
        self.runner.context.set_tensor_address("lr_curr", self._g_lr_curr.data_ptr())
        self.runner.context.set_tensor_address("hr_prev", self._g_hr_prev.data_ptr())

    def _to_nhwc(self, frames_nchw: torch.Tensor) -> torch.Tensor:
        return frames_nchw.permute(0, 2, 3, 1).contiguous()

    def _infer(self) -> None:
        self.runner.context.execute_async_v3(self._stream.cuda_stream)

    def _advance_temporal(self) -> None:
        self._infer()
        self._g_hr_prev.copy_(self.runner.outputs["hr_prev_out"])
        self._g_lr_prev.copy_(self._g_lr_curr)

    def _color_transfer_into_result(self) -> None:
        ai_rgb = self.runner.outputs["hr_display"].float().clamp_(0.0, 1.0)
        ref_up_nchw = F.interpolate(
            self._g_lr_curr.float().permute(0, 3, 1, 2),
            size=(UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE),
            mode="bicubic",
            align_corners=False,
        )
        self._g_ref_up.copy_(ref_up_nchw.permute(0, 2, 3, 1)).clamp_(0.0, 1.0)
        rgb_color_transfer(ai_rgb, self._g_ref_up, out=self._g_result)

    def _bootstrap(self, first_frame_nhwc: torch.Tensor) -> None:
        self._g_lr_prev.copy_(first_frame_nhwc.unsqueeze(0))
        self._g_lr_curr.copy_(first_frame_nhwc.unsqueeze(0))
        hr_init_nchw = F.interpolate(
            first_frame_nhwc.unsqueeze(0).float().permute(0, 3, 1, 2),
            size=(UNET4X_HR_PREV_SIZE, UNET4X_HR_PREV_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        self._g_hr_prev.copy_(hr_init_nchw.permute(0, 2, 3, 1).to(dtype=self._dtype))
        self._advance_temporal()

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        T = int(frames_256.shape[0])
        if T == 0:
            return []

        ks = max(0, int(keep_start))
        ke = min(T, int(keep_end))
        if ks >= ke:
            return []

        frames = frames_256.to(device=self.device, dtype=self._dtype)
        frames_nhwc = self._to_nhwc(frames)

        if ks > 0:
            ctx = ks - 1
            self._bootstrap(frames_nhwc[ctx])
            self._g_lr_curr.copy_(frames_nhwc[ctx : ctx + 1])
            self._advance_temporal()
        else:
            self._bootstrap(frames_nhwc[0])

        kept_count = ke - ks
        result = torch.empty(kept_count, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=torch.float32, device=self.device)

        for i in range(ks, ke):
            self._g_lr_curr.copy_(frames_nhwc[i : i + 1])
            if i == 0:
                self._g_lr_prev.copy_(frames_nhwc[0:1])

            self._advance_temporal()
            self._color_transfer_into_result()
            result[i - ks].copy_(self._g_result[0])

        kept_nchw = result.permute(0, 3, 1, 2).clamp_(0, 1).mul_(255.0).to(dtype=torch.uint8)
        return list(kept_nchw.unbind(0))

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None
