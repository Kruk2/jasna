from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

RTX_SUPERRES_INPUT_SIZE = 256
SCALE_CHOICES = [2, 4]
QUALITY_CHOICES = ["low", "medium", "high", "ultra"]
DENOISE_CHOICES = ["none", "low", "medium", "high", "ultra"]
DEBLUR_CHOICES = ["none", "low", "medium", "high", "ultra"]


def _resolve_quality(name: str):
    from nvvfx import VideoSuperRes
    return {
        "low": VideoSuperRes.QualityLevel.LOW,
        "medium": VideoSuperRes.QualityLevel.MEDIUM,
        "high": VideoSuperRes.QualityLevel.HIGH,
        "ultra": VideoSuperRes.QualityLevel.ULTRA,
    }[name.lower()]


def _resolve_denoise(name: str):
    from nvvfx import VideoSuperRes
    return {
        "low": VideoSuperRes.QualityLevel.DENOISE_LOW,
        "medium": VideoSuperRes.QualityLevel.DENOISE_MEDIUM,
        "high": VideoSuperRes.QualityLevel.DENOISE_HIGH,
        "ultra": VideoSuperRes.QualityLevel.DENOISE_ULTRA,
    }[name.lower()]


def _resolve_deblur(name: str):
    from nvvfx import VideoSuperRes
    return {
        "low": VideoSuperRes.QualityLevel.DEBLUR_LOW,
        "medium": VideoSuperRes.QualityLevel.DEBLUR_MEDIUM,
        "high": VideoSuperRes.QualityLevel.DEBLUR_HIGH,
        "ultra": VideoSuperRes.QualityLevel.DEBLUR_ULTRA,
    }[name.lower()]


class RtxSuperresSecondaryRestorer:
    name = "rtx-super-res"
    num_workers = 1

    def __init__(self, *, device: torch.device, scale: int = 4, quality: str = "high",
                 denoise: Optional[str] = "medium", deblur: Optional[str] = None) -> None:
        from nvvfx import VideoSuperRes

        output_size = RTX_SUPERRES_INPUT_SIZE * scale

        self.device = torch.device(device)
        gpu = self.device.index or 0
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream

        self._sr = VideoSuperRes(device=gpu, quality=_resolve_quality(quality))
        self._sr.output_width = output_size
        self._sr.output_height = output_size
        self._sr.load()

        self._denoise = None
        if denoise is not None and denoise.lower() != "none":
            self._denoise = VideoSuperRes(device=gpu, quality=_resolve_denoise(denoise))
            self._denoise.output_width = output_size
            self._denoise.output_height = output_size
            self._denoise.load()

        self._deblur = None
        if deblur is not None and deblur.lower() != "none":
            self._deblur = VideoSuperRes(device=gpu, quality=_resolve_deblur(deblur))
            self._deblur.output_width = output_size
            self._deblur.output_height = output_size
            self._deblur.load()

        logger.info("RtxSuperresSecondaryRestorer: scale=%dx quality=%s denoise=%s deblur=%s (%dx%d -> %dx%d)",
                     scale, quality, denoise, deblur,
                     RTX_SUPERRES_INPUT_SIZE, RTX_SUPERRES_INPUT_SIZE,
                     output_size, output_size)

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t = int(frames_256.shape[0])
        if t == 0:
            return []

        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []
        frames_256 = frames_256[ks:ke]
        t = int(frames_256.shape[0])

        out: list[torch.Tensor] = []
        for i in range(t):
            frame = frames_256[i].to(device=self.device, dtype=torch.float32).contiguous()

            result = torch.from_dlpack(self._sr.run(frame, stream_ptr=self._stream_ptr).image).clone()
            if self._denoise is not None:
                result = torch.from_dlpack(self._denoise.run(result, stream_ptr=self._stream_ptr).image).clone()
            if self._deblur is not None:
                result = torch.from_dlpack(self._deblur.run(result, stream_ptr=self._stream_ptr).image).clone()

            out_u8 = result.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
            out.append(out_u8)

        return out

    def close(self) -> None:
        if self._sr is not None:
            self._sr.close()
            self._sr = None
        if self._denoise is not None:
            self._denoise.close()
            self._denoise = None
        if self._deblur is not None:
            self._deblur.close()
            self._deblur = None
