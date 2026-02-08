import sys
from pathlib import Path

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
