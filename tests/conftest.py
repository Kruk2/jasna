"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
from importlib.util import find_spec
import sys

import pytest

_HAS_TENSORRT = find_spec("tensorrt") is not None

if _HAS_TENSORRT and find_spec("tensorrt_libs") is not None:
    import tensorrt_libs

collect_ignore = [] if _HAS_TENSORRT else [
    "test_basicvsrpp_sub_engines.py",
    "test_rtx_superres_restorer.py",
    "test_torch_tensorrt_export.py",
    "test_trt_runner.py",
    "test_trt_utils.py",
    "test_unet4x_secondary_restorer.py",
]


@pytest.fixture(scope="session", autouse=True)
def synchronize_windows_rocm_before_interpreter_teardown():
    """Drain queued HIP work before CPython unloads the Windows runtime."""

    yield
    if sys.platform != "win32":
        return

    import torch

    if getattr(torch.version, "hip", None) and torch.cuda.is_available():
        torch.cuda.synchronize()


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
