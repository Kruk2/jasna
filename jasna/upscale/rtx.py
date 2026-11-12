"""Whole-frame upscaling with NVIDIA RTX Video Super Resolution (nvvfx), NVIDIA only."""

from __future__ import annotations

import torch

from jasna.session_config import RtxLevelName, RtxQualityName
from jasna.upscale import Size, finish_frame


def _load(level, size: Size, device: torch.device):
    from nvvfx import VideoSuperRes

    effect = VideoSuperRes(device=device.index or 0, quality=level)
    effect.output_width, effect.output_height = size
    effect.load()
    return effect


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
        import jasna.restorer.rtx_superres_secondary_restorer  # noqa: F401  pins pip TensorRT before nvvfx loads
        from nvvfx import VideoSuperRes

        levels = VideoSuperRes.QualityLevel
        cleanups = [
            getattr(levels, f"{kind}_{level.upper()}")
            for kind, level in (("DENOISE", denoise), ("DEBLUR", deblur))
            if level != "none"
        ]
        self._device = device
        self._output_size = output_size
        self._strength = strength
        self._effects = []
        # nvvfx effects fail on any stream but the default one they load with.
        with torch.cuda.stream(torch.cuda.default_stream(device)):
            try:
                for level in cleanups:
                    self._effects.append(_load(level, source_size, device))
                self._effects.append(_load(getattr(levels, quality.upper()), output_size, device))
            except BaseException:
                self.close()
                raise

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        current = torch.cuda.current_stream(self._device)
        default = torch.cuda.default_stream(self._device)
        default.wait_stream(current)
        frame.record_stream(default)
        with torch.cuda.stream(default):
            image = frame.float().div_(255.0)
            for effect in self._effects:
                # Each effect reuses its output buffer; it is read before that effect runs again.
                image = torch.from_dlpack(effect.run(image, stream_ptr=default.cuda_stream).image)
            upscaled = finish_frame(image[None], frame, self._output_size, self._strength)
        current.wait_stream(default)
        upscaled.record_stream(current)
        return upscaled

    def close(self) -> None:
        for effect in self._effects:
            effect.close()
        self._effects.clear()
