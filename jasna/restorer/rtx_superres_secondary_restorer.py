from __future__ import annotations

import logging

import torch

from jasna.session_config import RtxLevelName, RtxQualityName
from jasna.upscale.nvvfx_effects import SuperResChain

logger = logging.getLogger(__name__)

RTX_SUPERRES_INPUT_SIZE = 256


class RtxSuperresSecondaryRestorer:
    name = "rtx-super-res"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        device: torch.device,
        scale: int,
        quality: RtxQualityName,
        denoise: RtxLevelName,
        deblur: RtxLevelName,
    ) -> None:
        self.device = torch.device(device)
        self.input_size = RTX_SUPERRES_INPUT_SIZE
        self.output_size = self.input_size * scale
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
        self._chain = SuperResChain(
            self.device,
            output_size=(self.output_size, self.output_size),
            quality=quality,
            denoise=denoise,
            deblur=deblur,
        )
        logger.info("RtxSuperresSecondaryRestorer: scale=%dx quality=%s denoise=%s deblur=%s (%dx%d -> %dx%d)",
                     scale, quality, denoise, deblur,
                     self.input_size, self.input_size,
                     self.output_size, self.output_size)

    def restore(self, frames: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t = int(frames.shape[0])
        if t == 0:
            return []
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (
            3,
            self.input_size,
            self.input_size,
        ):
            raise ValueError(
                f"expected frames shaped (T, 3, {self.input_size}, {self.input_size}), "
                f"got {tuple(frames.shape)}"
            )

        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []
        return [
            self._chain(frame.to(device=self.device, dtype=torch.float32).contiguous(), self._stream_ptr)
            for frame in frames[ks:ke]
        ]

    def close(self) -> None:
        self._chain.close()
