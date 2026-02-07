import os
import sys

if len(sys.argv) > 1:
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        ATTACH_PARENT_PROCESS = -1
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12

        if kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
            for handle_id, stream_name in ((STD_OUTPUT_HANDLE, "stdout"), (STD_ERROR_HANDLE, "stderr")):
                handle = kernel32.GetStdHandle(handle_id)
                fd = msvcrt.open_osfhandle(handle, 1)
                setattr(sys, stream_name, os.fdopen(fd, "w"))
        else:
            kernel32.AllocConsole()
            sys.stdout = open("CONOUT$", "w")
            sys.stderr = open("CONOUT$", "w")
    from jasna.main import main
    main()
else:
    from jasna.gui import run_gui
    run_gui()
