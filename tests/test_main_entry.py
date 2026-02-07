import sys
from unittest.mock import patch

import pytest


def test_no_args_dispatches_to_gui() -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    with patch.object(sys, "argv", ["jasna"]):
        with patch("jasna.gui.run_gui") as run_gui:
            with patch("jasna.main.main"):
                import jasna.__main__  # noqa: F401
                run_gui.assert_called_once()


def test_with_args_dispatches_to_main() -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    with patch.object(sys, "argv", ["jasna", "--version"]):
        with patch("jasna.main.main") as main:
            with patch("jasna.gui.run_gui"):
                import jasna.__main__  # noqa: F401
                main.assert_called_once()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console attach only on Windows")
def test_windows_cli_attach_console_does_not_crash(monkeypatch) -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    monkeypatch.setattr(sys, "argv", ["jasna", "--version"])
    kernel32 = type("K", (), {})()
    kernel32.AttachConsole = lambda _: True
    kernel32.GetStdHandle = lambda _: 1
    monkeypatch.setattr("jasna.__main__.ctypes.windll.kernel32", kernel32)
    monkeypatch.setattr("jasna.__main__.msvcrt.open_osfhandle", lambda h, m: 1)
    monkeypatch.setattr("jasna.__main__.os.fdopen", lambda fd, mode: sys.__stdout__)

    with patch("jasna.main.main") as main:
        with patch("jasna.gui.run_gui"):
            import jasna.__main__  # noqa: F401
            main.assert_called_once()
