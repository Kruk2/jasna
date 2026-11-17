"""Restored GPU frames as PIL images for the GUI previews and the video player."""

from __future__ import annotations

from PIL import Image


def downscaled_size(size: tuple[int, int], max_size: tuple[int, int]) -> tuple[int, int]:
    """``size`` (width, height) shrunk to fit ``max_size`` with its aspect kept; never enlarged."""
    width, height = size
    scale = min(1.0, max_size[0] / width, max_size[1] / height)
    if scale == 1.0:
        return size
    return max(2, round(width * scale)), max(2, round(height * scale))


def preview_image(frame, max_size: tuple[int, int], *, lut_applier, stereo: str) -> Image.Image:
    """The left eye of a restored ``frame`` (all of it for flat video) with the LUT applied,
    shrunk to fit ``max_size``."""
    from jasna.vr180 import left_eye

    frame = left_eye(frame, stereo)
    if lut_applier is not None:
        frame = lut_applier.apply(frame)
    return gpu_frame_image(frame, downscaled_size((int(frame.shape[-1]), int(frame.shape[-2])), max_size))


def gpu_frame_image(frame, size: tuple[int, int]) -> Image.Image:
    """``frame`` (uint8 [3, H, W] on any device) as an RGB image of exactly ``size`` (width, height),
    resized on the frame's device so only the shown pixels are copied to the CPU."""
    import torch
    import torch.nn.functional as functional

    height, width = int(frame.shape[-2]), int(frame.shape[-1])
    target = (max(2, int(size[1])), max(2, int(size[0])))
    if target != (height, width):
        resize_dtype = torch.float16 if frame.is_cuda else torch.float32
        frame = functional.interpolate(
            frame.unsqueeze(0).to(resize_dtype),
            size=target,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).round_().clamp_(0, 255).to(torch.uint8)
    array = frame.to("cpu").permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array).copy()
