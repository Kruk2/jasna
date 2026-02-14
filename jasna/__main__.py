import multiprocessing
import os
import sys
from pathlib import Path

_JASNA_MAIN_PID = os.environ.get("JASNA_MAIN_PID")
if _JASNA_MAIN_PID and str(os.getpid()) != _JASNA_MAIN_PID:
    if len(sys.argv) < 2 or sys.argv[1] != "--multiprocessing-fork":
        sys.exit(0)
if multiprocessing.parent_process() is not None:
    sys.exit(0)
os.environ["JASNA_MAIN_PID"] = str(os.getpid())

from jasna.bootstrap import sanitize_sys_path_for_local_dev

if not getattr(sys, "frozen", False):
    sanitize_sys_path_for_local_dev(Path(__file__).resolve().parent)


def _preload_native_libs():
    """Import native GPU libraries before tkinter on Linux.

    On Linux, loading Tcl/Tk (via customtkinter) first introduces shared
    library conflicts that prevent _python_vali and PyNvVideoCodec native
    extensions from initializing. Importing them before tkinter avoids this.
    """
    if sys.platform != "linux":
        return
    for mod in ("python_vali", "PyNvVideoCodec"):
        try:
            __import__(mod)
        except Exception:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()

if multiprocessing.parent_process() is None:
    argv0_stem = Path(sys.argv[0]).stem.lower()

    if sys.platform == "win32":
        if argv0_stem == "jasna-cli":
            if len(sys.argv) == 1:
                from jasna.main import build_parser

                build_parser().print_help()
                raise SystemExit(0)

            from jasna.main import main

            main()
        elif argv0_stem == "jasna-gui":
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
    else:
        if len(sys.argv) > 1:
            from jasna.main import main

            main()
        else:
            _preload_native_libs()
            from jasna.gui import run_gui

            run_gui()
