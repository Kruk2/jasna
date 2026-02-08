import sys

from jasna.packaging.windows_dll_paths import configure_windows_dll_search_paths

if sys.platform == "win32" and getattr(sys, "frozen", False):
    configure_windows_dll_search_paths()

