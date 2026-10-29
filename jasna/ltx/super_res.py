"""RTX Super Res for LTX windows whose crop is larger than the model canvas.

Compose enlarges such a window back to its crop. With super resolution the window is
first upscaled 2x, so compose shrinks it to the crop, or enlarges it by the remainder
past 2x.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import Window
from jasna.session_config import RtxLevelName, RtxQualityName

SUPER_RES_SCALE = 2


def enlarged_windows(windows: Sequence[Window]) -> set[int]:
    """Indices of the windows that compose enlarges on some frame."""
    return {window.index for window in windows if max(crop.side for crop in window.crops) > window.canvas}


def upscaled_crop(crop: CropFrame) -> CropFrame:
    """``crop`` as it lies on the canvas after a 2x upscale."""
    height, width = crop.resize_hw
    return CropFrame(
        crop_box=crop.crop_box,
        resize_hw=(height * SUPER_RES_SCALE, width * SUPER_RES_SCALE),
        pad=tuple(side * SUPER_RES_SCALE for side in crop.pad),
    )


class LtxSuperRes:
    """One RTX Super Res 2x upscaler per canvas size, made on first use."""

    def __init__(
        self, *, device: torch.device, quality: RtxQualityName, denoise: RtxLevelName, deblur: RtxLevelName
    ) -> None:
        self._device = device
        self._quality = quality
        self._denoise = None if denoise == "none" else denoise
        self._deblur = None if deblur == "none" else deblur
        self._upscalers = {}

    def upscale(self, canvas: torch.Tensor) -> torch.Tensor:
        """uint8 ``[3, S, S]`` on the GPU to uint8 ``[3, 2S, 2S]``, ordered after the caller's stream."""
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer

        current = torch.cuda.current_stream(self._device)
        # nvvfx effects fail on any stream but the default one they load with.
        default = torch.cuda.default_stream(self._device)
        default.wait_stream(current)
        with torch.cuda.stream(default):
            size = int(canvas.shape[-1])
            upscaler = self._upscalers.get(size)
            if upscaler is None:
                upscaler = RtxSuperresSecondaryRestorer(
                    device=self._device,
                    scale=SUPER_RES_SCALE,
                    quality=self._quality,
                    denoise=self._denoise,
                    deblur=self._deblur,
                    input_size=size,
                )
                self._upscalers[size] = upscaler
            upscaled = upscaler.restore(canvas[None].float().div_(255.0), keep_start=0, keep_end=1)[0]
        current.wait_stream(default)
        upscaled.record_stream(current)
        return upscaled

    def close(self) -> None:
        for upscaler in self._upscalers.values():
            upscaler.close()
        self._upscalers.clear()
