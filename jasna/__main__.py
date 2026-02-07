import sys

if len(sys.argv) > 1:
    # CLI mode when arguments provided
    from jasna.main import main
    main()
else:
    # GUI mode when no arguments
    from jasna.gui import run_gui
    run_gui()
