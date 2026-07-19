import logging
import multiprocessing
import os
import sys
from pathlib import Path, PureWindowsPath

logger = logging.getLogger(__name__)

from jasna import startup_timing  # noqa: F401  captures PROCESS_START near process start

if sys.platform == "win32":
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

# Wire CUDA/TensorRT/native-lib DLL search before any torch/native import. The Nuitka
# build has no PyInstaller runtime hook, so this is the only place it runs; without it
# torch's CUDA init depends on ambient PATH and randomly fails the system check.
from jasna.packaging.windows_dll_paths import configure_windows_dll_search_paths

configure_windows_dll_search_paths()

if len(sys.argv) >= 3 and sys.argv[1] == "--compile-engines":
    from jasna.engine_compiler import EngineCompilationRequest, _subprocess_compile
    _subprocess_compile(EngineCompilationRequest.from_json(sys.argv[2]))
    sys.exit(0)

_JASNA_MAIN_PID = os.environ.get("JASNA_MAIN_PID")
if _JASNA_MAIN_PID and str(os.getpid()) != _JASNA_MAIN_PID:
    if len(sys.argv) < 2 or sys.argv[1] != "--multiprocessing-fork":
        sys.exit(0)
if multiprocessing.parent_process() is not None:
    sys.exit(0)
os.environ["JASNA_MAIN_PID"] = str(os.getpid())

from jasna._frozen import is_frozen
from jasna.bootstrap import sanitize_sys_path_for_local_dev
from jasna.os_utils import drop_console_window

if not is_frozen():
    sanitize_sys_path_for_local_dev(Path(__file__).resolve().parent)


def _preload_native_libs():
    """Import native media libraries before tkinter on Linux.

    On Linux, loading Tcl/Tk (via customtkinter) first can introduce shared
    library conflicts that prevent PyAV's bundled libav from initializing.
    Importing it before tkinter avoids this.
    """
    if sys.platform != "linux":
        return
    for mod in ("av",):
        try:
            __import__(mod)
        except Exception:
            logger.warning("Native preload of %s failed", mod, exc_info=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()

if multiprocessing.parent_process() is None:
    argv0_path = (
        PureWindowsPath(sys.argv[0])
        if sys.platform == "win32"
        else Path(sys.argv[0])
    )
    argv0_stem = argv0_path.stem.lower()

    if sys.platform == "win32":
        if argv0_stem == "jasna-cli":
            if len(sys.argv) == 1:
                from jasna.main import build_parser

                build_parser().print_help()
                raise SystemExit(0)

            from jasna.main import main

            main()
        elif argv0_stem == "jasna-gui":
            drop_console_window()
            _preload_native_libs()
            from jasna.gui import run_gui

            run_gui()
        else:
            if len(sys.argv) > 1:
                from jasna.main import main

                main()
            else:
                drop_console_window()
                _preload_native_libs()
                from jasna.gui import run_gui

                run_gui()
    else:
        if len(sys.argv) > 1:
            from jasna.main import main

            main()
        else:
            _preload_native_libs()
            from jasna.gui import run_gui

            run_gui()
