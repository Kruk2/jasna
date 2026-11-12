"""Whole-frame upscaling of offline exports: every finished frame is upscaled to the target
height right before the encoder (and before frame generation), so the output video is
larger than the source."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter
    from jasna.media.probe import VideoMetadata

Size = tuple[int, int]
"""``(width, height)`` in pixels."""


def upscaled_size(width: int, height: int, target_height: int) -> Size | None:
    """The output size for ``target_height``, or ``None`` when the source is already at least
    that tall. The width keeps the aspect ratio, rounded to even for 4:2:0 chroma."""
    if height >= target_height:
        return None
    return round(width * target_height / height / 2) * 2, target_height


def upscaled_metadata(metadata: VideoMetadata, size: Size) -> VideoMetadata:
    """``metadata`` as the encoder sees the upscaled output: its size, and the source bitrate
    grown by the pixel ratio, so the source-tied bitrate ceiling keeps each pixel's share."""
    width, height = size
    pixel_ratio = width * height / (metadata.video_width * metadata.video_height)
    return replace(
        metadata, video_width=width, video_height=height, video_bitrate=round(metadata.video_bitrate * pixel_ratio)
    )


def finish_frame(upscaled: torch.Tensor, source: torch.Tensor, size: Size, strength: float) -> torch.Tensor:
    """A model's ``[1, 3, h, w]`` output in ``[0, 1]`` as the uint8 ``[3, H, W]`` output frame:
    resized to exactly ``size`` (bicubic, antialiased when shrinking) and, below full
    ``strength``, mixed with a plain bicubic upscale of the uint8 ``source`` frame."""
    width, height = size
    if tuple(upscaled.shape[-2:]) != (height, width):
        upscaled = F.interpolate(upscaled, size=(height, width), mode="bicubic", antialias=True)
    if strength < 1.0:
        plain = F.interpolate(source[None].to(upscaled.dtype).div_(255.0), size=(height, width), mode="bicubic")
        upscaled = torch.lerp(plain, upscaled, strength)
    return upscaled[0].clamp(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)


class FrameUpscaler(Protocol):
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """uint8 ``[3, H, W]`` GPU frame to the uint8 output frame, ordered after the caller's
        current stream and safe to use on it."""

    def close(self) -> None: ...


class FrameUpscaleWriter:
    """Upscales each frame before ``writer``; owns ``upscaler`` and closes it with ``writer``."""

    def __init__(self, writer: FrameWriter, upscaler: FrameUpscaler) -> None:
        self._writer = writer
        self._upscaler = upscaler

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self._writer.write(self._upscaler(frame), pts)

    def after_write(self, frames_written: int, stop_event) -> None:
        self._writer.after_write(frames_written, stop_event)

    def finish(self) -> None:
        self._writer.finish()

    def close(self) -> None:
        try:
            self._writer.close()
        finally:
            self._upscaler.close()
