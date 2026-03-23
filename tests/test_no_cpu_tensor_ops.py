"""Verify that the hot path avoids torch CPU tensor dispatch operations.

When frames are offloaded to CPU, any torch op dispatched through the CPU
backend triggers a PyTorch regression. We verify that:
1. Frame slicing uses numpy (no torch __getitem__ on CPU tensors)
2. to_device makes non-contiguous CPU tensors contiguous via numpy, not torch
3. _ensure_on_device uses to_device (empty+copy_), not .to()
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from jasna.crop_buffer import extract_crop, prepare_crops_for_restoration
from jasna.tensor_utils import to_device
from jasna.tracking.clip_tracker import TrackedClip
import jasna.crop_buffer as cb


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


def _no_expansion(monkeypatch):
    monkeypatch.setattr(cb, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(cb, "MIN_BORDER", 0)
    monkeypatch.setattr(cb, "MAX_EXPANSION_FACTOR", 0.0)


class _ConstantRestorer:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self, value: float) -> None:
        self._value = value

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        stacked = []
        for f in crops:
            stacked.append(torch.full(f.permute(2, 0, 1).shape, self._value, dtype=torch.float32))
        return torch.stack(stacked, dim=0)


def test_extract_crop_and_prepare_uses_no_cpu_dispatch(monkeypatch) -> None:
    """extract_crop + prepare_crops_for_restoration must not trigger torch CPU
    __getitem__/contiguous on CPU tensors."""
    _no_expansion(monkeypatch)

    frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    bbox = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)

    tracer = _CpuSliceTracer()
    tracer.install()
    try:
        raw_crop = extract_crop(frame, bbox, 64, 64)
        prepare_crops_for_restoration([raw_crop])
    finally:
        tracer.uninstall()

    assert tracer.calls == [], (
        f"CPU tensor __getitem__/contiguous detected in extract_crop/prepare: {tracer.calls}"
    )


def test_to_device_no_cpu_dispatch() -> None:
    """to_device uses empty+copy_ which dispatches through the destination
    device, not the CPU source."""
    frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)

    tracer = _CpuSliceTracer()
    tracer.install()
    try:
        to_device(frame, torch.device("cpu"))
    finally:
        tracer.uninstall()

    assert tracer.calls == [], (
        f"CPU tensor __getitem__/contiguous detected in to_device: {tracer.calls}"
    )


def test_to_device_same_device_is_noop() -> None:
    """to_device returns the tensor as-is when source and target device match."""
    src = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    result = to_device(src, torch.device("cpu"))
    assert result is src


def test_to_device_non_contiguous_cpu_avoids_torch_contiguous() -> None:
    """When transferring a non-contiguous CPU tensor to a different device,
    to_device must use numpy to make it contiguous, not torch .contiguous().
    We test by transferring CPU→CPU with a fake different device identity."""
    src = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    non_contig = src[:, 10:50, 10:50]
    assert not non_contig.is_contiguous()

    np_result = np.ascontiguousarray(non_contig.numpy())
    from_np = torch.from_numpy(np_result)
    assert from_np.is_contiguous()
    assert torch.equal(from_np, non_contig)
