"""NVIDIA RTX Video Effects (nvvfx): importing it after the pip TensorRT runtime, and its video
super resolution effects."""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import torch

from jasna.session_config import RtxLevelName, RtxQualityName
from jasna.upscale import Size


@functools.cache
def _preload_tensorrt_runtime() -> None:
    """Pin the pip ``tensorrt`` runtime before nvvfx loads its bundled copy.

    nvvfx (NVIDIA Maxine) ships its own TensorRT and loads ``libnvinfer.so.10``
    with ``RTLD_GLOBAL``. Our BasicVSR++ engines are serialized against the pip
    ``tensorrt`` (a different version), and TRT engines are version-locked. If
    nvvfx loads first, its TensorRT symbols win global symbol resolution,
    torch_tensorrt binds to them, and BasicVSR++ engine deserialization fails
    with a serialization-version mismatch. Loading the pip libs into the global
    scope first makes torch_tensorrt bind to the matching runtime; nvvfx then
    happily reuses it.
    """
    import tensorrt_libs

    libs_dir = os.path.dirname(tensorrt_libs.__file__)
    if sys.platform == "win32":
        for name in ("nvinfer_10.dll", "nvinfer_plugin_10.dll"):
            ctypes.WinDLL(os.path.join(libs_dir, name))
    else:
        for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10"):
            ctypes.CDLL(os.path.join(libs_dir, name), mode=ctypes.RTLD_GLOBAL)


def import_nvvfx() -> ModuleType:
    """``nvvfx``, imported only once the pip TensorRT runtime holds the global symbol scope."""
    _preload_tensorrt_runtime()
    import nvvfx

    return nvvfx


def cleanup_level_names(denoise: RtxLevelName, deblur: RtxLevelName) -> list[str]:
    """The denoise and deblur ``VideoSuperRes`` levels to run, in that order (``"none"`` skips one)."""
    return [f"{kind}_{level}" for kind, level in (("denoise", denoise), ("deblur", deblur)) if level != "none"]


def video_super_res(level_name: str, output_size: Size, device: torch.device):
    """A loaded ``VideoSuperRes`` effect at ``level_name`` (``"high"``, ``"denoise_low"``, ...) with
    a ``(width, height)`` output."""
    VideoSuperRes = import_nvvfx().VideoSuperRes
    effect = VideoSuperRes(device=device.index or 0, quality=getattr(VideoSuperRes.QualityLevel, level_name.upper()))
    effect.output_width, effect.output_height = output_size
    effect.load()
    return effect


@contextmanager
def on_default_stream(device: torch.device) -> Iterator[torch.cuda.Stream]:
    """Runs the block on the default stream, ordered after the caller's stream and before its later
    work: nvvfx effects fail on any stream but the default one they load with."""
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    default.wait_stream(current)
    with torch.cuda.stream(default):
        yield default
    current.wait_stream(default)


class SuperResChain:
    """Super resolution to ``output_size``, then denoise and deblur at that size."""

    def __init__(
        self,
        device: torch.device,
        *,
        output_size: Size,
        quality: RtxQualityName,
        denoise: RtxLevelName,
        deblur: RtxLevelName,
    ) -> None:
        self._effects = [
            video_super_res(level, output_size, device) for level in (quality, *cleanup_level_names(denoise, deblur))
        ]

    def __call__(self, image: torch.Tensor, stream_ptr: int) -> torch.Tensor:
        """Float ``[3, h, w]`` in ``[0, 1]`` to the uint8 ``[3, H, W]`` output, run on ``stream_ptr``."""
        for effect in self._effects:
            image = torch.from_dlpack(effect.run(image, stream_ptr=stream_ptr).image).clone()
        return image.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)

    def close(self) -> None:
        for effect in self._effects:
            effect.close()
        self._effects.clear()
