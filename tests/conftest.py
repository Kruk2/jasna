"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
import gc
import re
import sys
import time
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

# LTX tests import triton/tensorrt transitively through jasna.ltx; skip them on
# machines without those modules (the AMD CI box), the same policy as the TRT list.
for _mod, _tests in (
    ("triton", [
        "test_ltx_transformer.py",
        "test_ltx_vae.py",
        "test_ltx_inference.py",
        "test_ltx_restore.py",
        "test_ltx_sampler_gpu.py",
        "test_ltx_storage_cleanup.py",
        "test_ltx_trial.py",
        "test_ltx_vae_gpu.py",
        "test_ltx_denoise.py",
        "test_benchmark_ltx_decode.py",
        "test_benchmark_ltx_denoise.py",
        "test_pipeline_init.py",
        "test_gui_ltx_models.py",
        "test_ltx_seed_preview.py",
        "test_segment_editor.py",
        "test_basicvsrpp_engine_compilation.py",
    ]),
    ("tensorrt", [
        "test_benchmark_ltx_decode.py",
        "test_benchmark_ltx_denoise.py",
        "test_pipeline_init.py",
    ]),
    # torchtrt cache routing needs the newer protection submodule, which this
    # checkout does not vendor; skip instead of failing collection.
    ("jasna.protection.export_cache", [
        "test_torchtrt_cache_routing.py",
    ]),
):
    if find_spec(_mod) is None:
        collect_ignore.extend(_tests)


def _has_module(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


#: ``jasna/protection`` is not vendored in this checkout: the licensing code only ships
#: in the official release. Tests that drive it cannot run here and must skip instead of
#: failing inside mock with an AttributeError.
PROTECTION_MISSING = not _has_module("jasna.protection.license_store")
_PROTECTION_REF = re.compile(r"jasna\.protection\.(\w+)")

if PROTECTION_MISSING:
    # The whole module is the activation dialog; every test in it needs the package.
    collect_ignore.extend(["test_offline_activation_ui_cli.py"])

# Same package, different entry point: the opaque LTX boundary tests cannot even be
# collected without it (module-scope fixture), so they error rather than fail.
if not _has_module("jasna.protection.opaque_ltx"):
    collect_ignore.extend(["test_opaque_ltx_native_boundary.py"])

# Node ids that need the package without naming it in their own body.
_UNVENDORED_PROTECTION_TESTS = {
    "tests/test_ltx_model_files.py::test_cli_offers_the_missing_model_only_when_ltx_runs",
}


def pytest_collection_modifyitems(config, items):
    if not PROTECTION_MISSING:
        return
    import inspect

    reason = "needs the private jasna.protection package, which this checkout does not vendor"
    for item in items:
        if item.nodeid in _UNVENDORED_PROTECTION_TESTS:
            item.add_marker(pytest.mark.skip(reason=reason))
            continue
        function = getattr(item, "function", None)
        if function is None:
            continue
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            continue
        referenced = {name for name in _PROTECTION_REF.findall(source)
                      if not _has_module("jasna.protection." + name)}
        if referenced:
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_configure(config):
    """Turn off CustomTkinter's per-monitor DPI detection on Windows, as ``run_gui()`` does.

    Otherwise every CTk window multiplies its sizes by the test machine's display scaling,
    so GUI tests would pass or fail depending on the monitor. With it off, the only scaling
    is the factor a test sets (see ``hidpi``), exactly as on Linux.
    """
    if sys.platform == "win32":
        import customtkinter as ctk

        ctk.deactivate_automatic_dpi_awareness()


@pytest.fixture
def show_window():
    """Process Tk events until a window is viewable, so its widgets have real sizes.

    On Windows CustomTkinter withdraws every new window to recolor its title bar and maps
    it again from an ``after`` callback; until then Tk reports 1x1 geometry, and a single
    ``update()`` may run before that callback.
    """

    def show(window) -> None:
        deadline = time.monotonic() + 2.0
        window.update()
        while not window.winfo_viewable() and time.monotonic() < deadline:
            window.update()
        assert window.winfo_viewable()

    return show


@pytest.fixture
def tk_garbage_collected():
    """Finalize Tk objects left by earlier GUI tests on the main thread.

    Code that runs ``gc.collect()`` on a worker thread (the GUI processor does after every
    job) would otherwise finalize them there, and each Tk call off the main thread waits a
    second for a main loop that tests never run, outlasting the tests' join timeouts.
    """
    gc.collect()


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
