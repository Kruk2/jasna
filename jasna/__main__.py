import sys
from pathlib import Path

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
        from jasna.gui import run_gui

        run_gui()
    else:
        if len(sys.argv) > 1:
            from jasna.main import main

            main()
        else:
            from jasna.gui import run_gui

            run_gui()
else:
    if len(sys.argv) > 1:
        from jasna.main import main

        main()
    else:
        from jasna.gui import run_gui

        run_gui()
