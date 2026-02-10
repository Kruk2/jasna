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


def test_linux_with_args_dispatches_to_main(monkeypatch) -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    with patch.object(sys, "argv", ["jasna", "--version"]):
        with patch("jasna.main.main") as main:
            with patch("jasna.gui.run_gui"):
                import jasna.__main__  # noqa: F401
                main.assert_called_once()


def test_windows_jasna_cli_no_args_prints_help_and_exits_0(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "jasna.__main__" or name.startswith("jasna.gui"):
            del sys.modules[name]

    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.setattr(sys, "argv", [r"C:\Program Files\Jasna\jasna-cli.exe"])

    with patch("jasna.main.build_parser") as build_parser:
        parser = build_parser.return_value
        with pytest.raises(SystemExit) as e:
            import jasna.__main__  # noqa: F401

        assert e.value.code == 0
        parser.print_help.assert_called_once()

    assert "jasna.gui" not in sys.modules


def test_windows_jasna_cli_with_args_dispatches_to_main(monkeypatch) -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.setattr(sys, "argv", [r"C:\Program Files\Jasna\jasna-cli.exe", "--version"])

    with patch("jasna.main.main") as main:
        with patch("jasna.gui.run_gui"):
            import jasna.__main__  # noqa: F401
            main.assert_called_once()


def test_windows_jasna_gui_with_args_still_dispatches_to_gui(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "jasna.__main__" or name.startswith("jasna.gui"):
            del sys.modules[name]
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.setattr(sys, "argv", [r"C:\Program Files\Jasna\jasna-gui.exe", "--version"])

    with patch("jasna.gui.run_gui") as run_gui:
        with patch("jasna.main.main") as main:
            import jasna.__main__  # noqa: F401
            run_gui.assert_called_once()
            main.assert_not_called()


def test_spawn_child_does_not_run_dispatch(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "jasna.__main__" or name.startswith("jasna.gui"):
            del sys.modules[name]
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "multiprocessing.parent_process",
        lambda: MagicMock(),
        raising=False,
    )
    with patch("jasna.gui.run_gui") as run_gui:
        with patch("jasna.main.main") as main:
            import jasna.__main__  # noqa: F401
            run_gui.assert_not_called()
            main.assert_not_called()
