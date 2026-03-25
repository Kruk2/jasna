from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn import functional as F

from jasna.engine_paths import UNET4X_BATCH_SIZE, UNET4X_ONNX_PATH, get_unet4x_engine_path  # noqa: F401
from jasna.trt.trt_runner import TrtRunner

logger = logging.getLogger(__name__)

UNET4X_INPUT_SIZE = 256
UNET4X_OUTPUT_SIZE = 1024


def compile_unet4x_engine(
    onnx_path: Path,
    device: torch.device,
    fp16: bool = True,
) -> Path:
    from jasna.trt import compile_onnx_to_tensorrt_engine
    return compile_onnx_to_tensorrt_engine(
        onnx_path,
        device,
        batch_size=UNET4X_BATCH_SIZE,
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
                "frames_stack": (UNET4X_BATCH_SIZE, 1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3),
                "hr_init": (1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3),
                "lr_init": (1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3),
            },
            device=self.device,
        )
        logger.info(
            "Unet4xSecondaryRestorer loaded: %s (batch=%d, %dx%d -> %dx%d)",
            self.engine_path, UNET4X_BATCH_SIZE,
            UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE,
            UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE,
        )

    def _to_nhwc(self, frames_nchw: torch.Tensor) -> torch.Tensor:
        return frames_nchw.permute(0, 2, 3, 1).contiguous()

    def _init_temporal_state(self, first_frame_nhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame = first_frame_nhwc.unsqueeze(0)  # (1, H, W, 3)
        lr_init = frame
        hr_init = F.interpolate(
            frame.permute(0, 3, 1, 2),  # (1, 3, 256, 256)
            size=(UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()  # (1, 1024, 1024, 3)
        return hr_init, lr_init

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        T = int(frames_256.shape[0])
        if T == 0:
            return []

        ks = max(0, int(keep_start))
        ke = min(T, int(keep_end))
        if ks >= ke:
            return []

        frames = frames_256.to(device=self.device, dtype=self._dtype)
        frames_nhwc = self._to_nhwc(frames)  # (T, 256, 256, 3)

        pad_count = (UNET4X_BATCH_SIZE - T % UNET4X_BATCH_SIZE) % UNET4X_BATCH_SIZE
        if pad_count > 0:
            padding = torch.zeros(
                pad_count, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3,
                dtype=self._dtype, device=self.device,
            )
            frames_nhwc = torch.cat([frames_nhwc, padding], dim=0)

        total = frames_nhwc.shape[0]
        num_batches = total // UNET4X_BATCH_SIZE
        result_nhwc = torch.empty(
            total, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3,
            dtype=self._dtype, device=self.device,
        )

        hr_prev, lr_prev = self._init_temporal_state(frames_nhwc[0])

        for i in range(num_batches):
            start = i * UNET4X_BATCH_SIZE
            batch = frames_nhwc[start : start + UNET4X_BATCH_SIZE]  # (4, 256, 256, 3)
            batch = batch.unsqueeze(1).contiguous()  # (4, 1, 256, 256, 3)

            outs = self.runner.infer({
                "frames_stack": batch,
                "hr_init": hr_prev,
                "lr_init": lr_prev,
            })

            result_nhwc[start : start + UNET4X_BATCH_SIZE] = outs["all_color_outputs"].squeeze(1)
            hr_prev = outs["hr_final"].clone()
            lr_prev = batch[-1:, 0].contiguous()  # (1, 256, 256, 3)

        kept = result_nhwc[ks:ke]  # (T', 1024, 1024, 3)

        kept_nchw = kept.permute(0, 3, 1, 2).float().clamp_(0, 1).mul_(255.0).round_().to(dtype=torch.uint8)
        return list(kept_nchw.unbind(0))

    def close(self) -> None:
        self.runner = None
