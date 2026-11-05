import os
from unittest.mock import patch

from jasna.engine_paths import engine_system_suffix


class TestEngineSystemSuffix:
    def test_windows(self):
        with patch.object(os, "name", "nt"):
            assert engine_system_suffix() == ".win"

    def test_linux(self):
        with patch.object(os, "name", "posix"):
            assert engine_system_suffix() == ".linux"

