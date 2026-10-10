"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
from importlib.util import find_spec

import pytest
import torch

_HAS_TENSORRT = find_spec("tensorrt") is not None

#: The AMD/ROCm build ships without the NVIDIA-only pieces (TensorRT sub-engines,
#: nvidia-smi driver checks, the CUDA fused preprocess kernel, the unet-4x /
#: RTX Super Resolution / TVAI secondary restorers) and takes different branches
#: in vendor-aware code (RF-DETR weights are `.pt` instead of `.onnx`, the driver
#: string is a ROCm version). Tests for those paths are NVIDIA-only by nature.
IS_AMD_BUILD = getattr(torch.version, "hip", None) is not None

requires_nvidia = pytest.mark.skipif(
    IS_AMD_BUILD,
    reason="NVIDIA-only path: this build has no TensorRT / nvidia-smi / CUDA kernels",
)

if _HAS_TENSORRT and find_spec("tensorrt_libs") is not None:
    import tensorrt_libs

collect_ignore = [] if _HAS_TENSORRT else [
    "test_basicvsrpp_sub_engines.py",
    # Imports jasna.restorer.basicvsrpp_sub_engines at module scope, so without
    # this the import error aborts collection of the WHOLE suite on the AMD build.
    "test_basicvsrpp_engine_compilation.py",
    "test_rtx_superres_restorer.py",
    "test_torch_tensorrt_export.py",
    "test_trt_runner.py",
    "test_trt_utils.py",
    "test_unet4x_secondary_restorer.py",
]


@pytest.fixture
def hidpi(request):
    """Reproduce Windows display scaling on a platform whose DPI factor is always 1.

    CustomTkinter multiplies its detected per-monitor DPI factor by these process-global
    factors, so setting them makes geometry()/minsize() and CTk widget sizes behave exactly
    as on a scaled Windows monitor while winfo_* keeps reporting physical pixels - the
    asymmetry behind issue #241. They are class attributes on ScalingTracker and leak into
    every later test unless reset.
    """
    import customtkinter as ctk

    factor = request.param
    ctk.set_widget_scaling(factor)
    ctk.set_window_scaling(factor)
    try:
        yield factor
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)


@pytest.fixture
def no_gpu_cleanup(monkeypatch):
    """Skip the per-job torch cleanup: it initializes CUDA, which can outlast thread joins in a busy run."""
    monkeypatch.setattr("jasna.gui.processor._cleanup_torch", lambda torch_mod: None)
