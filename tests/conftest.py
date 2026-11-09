"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
import gc
import hashlib
import sys
import time
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

_HAS_TENSORRT = find_spec("tensorrt") is not None

if _HAS_TENSORRT and find_spec("tensorrt_libs") is not None:
    import tensorrt_libs

collect_ignore = [] if _HAS_TENSORRT else [
    "test_basicvsrpp_sub_engines.py",
    "test_rtx_superres_restorer.py",
    "test_torch_tensorrt_export.py",
    "test_trt_utils.py",
]


def pytest_configure(config):
    """Turn off CustomTkinter's per-monitor DPI detection on Windows, as ``run_gui()`` does.

    Otherwise every CTk window multiplies its sizes by the test machine's display scaling,
    so GUI tests would pass or fail depending on the monitor. With it off, the only scaling
    is the factor a test sets (see ``hidpi``), exactly as on Linux.
    """
    if sys.platform == "win32":
        import customtkinter as ctk

        ctk.deactivate_automatic_dpi_awareness()


@pytest.fixture(scope="session", autouse=True)
def rocm_cuda_initialized():
    """Initialize ROCm torch before a test fakes ``torch.version.cuda``: torch's lazy CUDA
    capability check would then parse gfx architecture names as CUDA ones and fail."""
    import torch

    if torch.version.hip and torch.cuda.is_available():
        torch.cuda.init()


@pytest.fixture
def torch_config():
    """Set torch config values for one test through the config's own ``patch``: pytest 9.2's
    ``monkeypatch.setattr`` deletes them on undo, which hides them for the rest of the run."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        yield lambda config, **values: stack.enter_context(config.patch(**values))


@pytest.fixture
def rocm_probe(monkeypatch, tmp_path):
    """Set up a frozen AMD startup with a native GPU probe result."""
    import subprocess

    from jasna import os_utils, rocm_bootstrap

    monkeypatch.setattr(rocm_bootstrap, "is_frozen", lambda: True)
    monkeypatch.setattr(rocm_bootstrap.sys, "argv", ["jasna"])
    monkeypatch.setattr(rocm_bootstrap.sys, "executable", str(tmp_path / "jasna"))
    monkeypatch.setattr(os_utils, "get_application_executable", lambda: str(tmp_path / "jasna"))
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        # setenv records the original state, so the selection the probe writes is undone too.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)

    def configure(output, platform, kernel_packs=("gfx1030", "gfx1036", "gfx1100", "gfx1151", "gfx1200", "gfx1201")):
        monkeypatch.setattr(rocm_bootstrap.sys, "platform", platform)
        packs = tmp_path / "torch" / ".kpack"
        packs.mkdir(parents=True, exist_ok=True)
        for architecture in kernel_packs:
            (packs / f"torch_{architecture}.kpack").touch()
        relative = "lib/llvm/bin/offload-arch.exe" if platform == "win32" else "bin/rocminfo"
        tool = tmp_path / "_rocm_sdk_core" / relative
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.touch()

        def probe(*args, **kwargs):
            if isinstance(output, Exception):
                raise output
            stdout, returncode = output if isinstance(output, tuple) else (output, 0)
            result = subprocess.CompletedProcess(args[0], returncode, stdout)
            if kwargs.get("check"):
                result.check_returncode()
            return result

        monkeypatch.setattr(subprocess, "run", probe)

    return configure


@pytest.fixture
def tk_root():
    """A CTk root window, destroyed after the test; skips when there is no display."""
    from tkinter import TclError

    import customtkinter as ctk

    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


@pytest.fixture
def user_config_dir(monkeypatch, tmp_path):
    """Jasna's per-user config folder (settings.json) under ``tmp_path / "config"``, on every
    platform: Windows keeps it in %APPDATA%, the others in $XDG_CONFIG_HOME."""
    home = tmp_path / "config"
    monkeypatch.setenv("APPDATA", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home / "jasna"


@pytest.fixture
def jasna_app(monkeypatch, tmp_path, user_config_dir):
    """The real main window after its startup work, on a machine reporting a Blackwell NVIDIA
    GPU and an SSD install, with settings under ``tmp_path / "config"`` and restoration models
    (LTX included) under ``tmp_path / "models"``."""
    import threading
    from tkinter import TclError

    from jasna import engine_paths, os_utils
    from jasna.accelerator import AcceleratorVendor
    from jasna.gui import app as app_module

    monkeypatch.setattr(app_module, "_warm_up_cuda", lambda: (AcceleratorVendor.NVIDIA, True, True))
    monkeypatch.setattr(os_utils, "install_on_hard_disk", lambda: False)
    monkeypatch.setattr(os_utils, "windows_module_loaded", lambda _name: False)
    monkeypatch.setattr(engine_paths, "default_restoration_model_path", lambda name: tmp_path / "models" / name)
    try:
        app = app_module.JasnaApp(skip_wizard=True)
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    app.update_idletasks()
    for thread in threading.enumerate():
        if thread.name == "cuda-warmup":
            thread.join()
    app._main_thread._run_pending()
    yield app
    if app.children:  # empty once the test has closed the window
        app._stop_system_stats_poller()
        app._main_thread.close()
        app.destroy()


@pytest.fixture
def gui_language(user_config_dir):
    """Switch the GUI language for one test; the choice is saved under ``tmp_path``."""
    from jasna.gui.locales import get_locale

    locale = get_locale()
    original = locale.current_language
    yield locale.set_language
    locale.set_language(original)


@pytest.fixture
def settle_activation(monkeypatch):
    """Return ``settle(window)``: finish every activation worker thread, then run what the
    workers posted to ``window``'s main-thread queue, until no worker is left."""
    from jasna.gui import activation_worker

    threads = []
    start = activation_worker._start
    monkeypatch.setattr(activation_worker, "_start", lambda *args: threads.append(start(*args)) or threads[-1])

    def settle(window) -> None:
        while threads:
            threads.pop(0).join()
            window._main_thread._run_pending()

    return settle


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
def nvidia_vendor(monkeypatch):
    """Report a CUDA build of torch, so tests of NVIDIA behavior run the same on a ROCm host."""
    import torch

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")


@pytest.fixture
def no_gpu_cleanup(monkeypatch):
    """Skip the per-job torch cleanup: it initializes CUDA, which can outlast thread joins in a busy run."""
    monkeypatch.setattr("jasna.gui.processor._cleanup_torch", lambda torch_mod: None)


@pytest.fixture
def amd_gpu(monkeypatch):
    """Report a ROCm build of torch whose GPU is the given gfx architecture."""
    import torch

    def use(architecture: str = "gfx1201") -> None:
        monkeypatch.setattr(torch.version, "hip", "7.15.26333")
        monkeypatch.setattr(
            torch.cuda, "get_device_properties",
            lambda _device: SimpleNamespace(gcnArchName=f"{architecture}:sramecc-:xnack-"),
        )

    return use


@pytest.fixture
def amd_driver_environment(monkeypatch, amd_gpu):
    import torch

    amd_gpu()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def sparse_ltx_downloads(monkeypatch):
    """A sparse file of a published LTX size counts as that current download."""
    from jasna.ltx import model_files

    zeros = hashlib.sha256(bytes(1 << 20)).hexdigest()
    monkeypatch.setattr(model_files, "LTX_DOWNLOAD_HEADS", dict.fromkeys(model_files.LTX_DOWNLOAD_HEADS, zeros))
