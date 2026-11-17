"""Verify that the hot path avoids torch CPU tensor dispatch operations.

When frames are offloaded to CPU, any torch op dispatched through the CPU
backend triggers a PyTorch regression. We verify that:
Frame slicing uses numpy (no torch __getitem__ on CPU tensors).
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from jasna.crop_buffer import extract_crop, prepare_crops_for_restoration


class _CpuSliceTracer:
    """Tracks __getitem__ and contiguous() calls on CPU tensors — the two
    operations we specifically avoid by using numpy slicing + np.ascontiguousarray."""

    def __init__(self):
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self._originals: dict[str, object] = {}

    def install(self):
        for name in ("__getitem__", "contiguous"):
            orig = getattr(torch.Tensor, name)
            self._originals[name] = orig

            def make_wrapper(method_name, orig_fn):
                def wrapper(self_tensor, *args, **kwargs):
                    if self_tensor.device.type == "cpu" and self_tensor.numel() > 0:
                        with self._lock:
                            self.calls.append(method_name)
                    return orig_fn(self_tensor, *args, **kwargs)
                return wrapper

            setattr(torch.Tensor, name, make_wrapper(name, orig))

    def uninstall(self):
        for name, orig in self._originals.items():
            setattr(torch.Tensor, name, orig)
        self._originals.clear()


@pytest.mark.usefixtures("exact_crops")
def test_extract_crop_and_prepare_uses_no_cpu_dispatch() -> None:
    """extract_crop + prepare_crops_for_restoration must not trigger torch CPU
    __getitem__/contiguous on CPU tensors."""
    frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    bbox = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)

    tracer = _CpuSliceTracer()
    tracer.install()
    try:
        raw_crop = extract_crop(frame, bbox, 64, 64)
        prepare_crops_for_restoration([raw_crop], device=torch.device("cpu"), dtype=torch.float32)
    finally:
        tracer.uninstall()

    assert tracer.calls == [], (
        f"CPU tensor __getitem__/contiguous detected in extract_crop/prepare: {tracer.calls}"
    )
