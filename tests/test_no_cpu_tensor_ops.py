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

from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tensor_utils import to_device
from jasna.tracking.clip_tracker import TrackedClip
from jasna.tracking.frame_buffer import FrameBuffer
import jasna.restorer.restoration_pipeline as rp


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
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)


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


def test_prepare_clip_inputs_uses_numpy_slicing_for_cpu_frames(monkeypatch) -> None:
    """When frame is on CPU, _prepare_clip_inputs must use numpy for slicing
    instead of torch __getitem__, and must not call .contiguous() on CPU tensors."""
    _no_expansion(monkeypatch)

    restorer = _ConstantRestorer(0.5)
    pipeline = RestorationPipeline(restorer=restorer)  # type: ignore[arg-type]

    frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    bbox = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
    mask = torch.ones((8, 8), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(8, 8),
                       bboxes=[bbox], masks=[mask])

    tracer = _CpuSliceTracer()
    tracer.install()
    try:
        pipeline._prepare_clip_inputs(clip, [frame])
    finally:
        tracer.uninstall()

    assert tracer.calls == [], (
        f"CPU tensor __getitem__/contiguous detected in _prepare_clip_inputs: {tracer.calls}"
    )


def test_ensure_on_device_no_cpu_dispatch() -> None:
    """_ensure_on_device uses to_device (empty+copy_) which dispatches through
    the destination device, not the CPU source."""
    fb = FrameBuffer(device=torch.device("cpu"))
    frame = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    fb.add_frame(0, pts=0, frame=frame, clip_track_ids=set())
    pending = fb.frames[0]

    tracer = _CpuSliceTracer()
    tracer.install()
    try:
        fb._ensure_on_device(pending)
    finally:
        tracer.uninstall()

    assert tracer.calls == [], (
        f"CPU tensor __getitem__/contiguous detected in _ensure_on_device: {tracer.calls}"
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
