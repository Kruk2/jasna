"""Whole-frame upscaling with NVIDIA RTX Video Super Resolution (nvvfx), NVIDIA only."""

from __future__ import annotations

import torch

from jasna.session_config import RtxLevelName, RtxQualityName
from jasna.upscale import Size, finish_frame
from jasna.upscale.nvvfx_effects import cleanup_level_names, on_default_stream, video_super_res


class RtxFrameUpscaler:
    """Denoise and deblur (when set) clean the frame at its source size, then super resolution
    scales it straight to the output size."""

    def __init__(
        self,
        device: torch.device,
        *,
        source_size: Size,
        output_size: Size,
        quality: RtxQualityName,
        denoise: RtxLevelName,
        deblur: RtxLevelName,
        strength: float,
    ) -> None:
        self._device = device
        self._output_size = output_size
        self._strength = strength
        self._effects = []
        with torch.cuda.stream(torch.cuda.default_stream(device)):
            try:
                for level in cleanup_level_names(denoise, deblur):
                    self._effects.append(video_super_res(level, source_size, device))
                self._effects.append(video_super_res(quality, output_size, device))
            except BaseException:
                self.close()
                raise

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        with on_default_stream(self._device) as default:
            frame.record_stream(default)
            image = frame.float().div_(255.0)
            for effect in self._effects:
                # Each effect reuses its output buffer; it is read before that effect runs again.
                image = torch.from_dlpack(effect.run(image, stream_ptr=default.cuda_stream).image)
            upscaled = finish_frame(image[None], frame, self._output_size, self._strength)
        upscaled.record_stream(torch.cuda.current_stream(self._device))
        return upscaled

    def close(self) -> None:
        for effect in self._effects:
            effect.close()
        self._effects.clear()
