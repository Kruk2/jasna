import sys

from jasna.packaging.linux_lib_paths import configure_linux_dll_search_paths

if sys.platform == "linux" and getattr(sys, "frozen", False):
    configure_linux_dll_search_paths()

