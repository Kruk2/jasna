import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def restore_jasna_modules():
    saved = {name: module for name, module in sys.modules.items() if name.startswith("jasna")}
    yield
    for name in [name for name in sys.modules if name.startswith("jasna") and name not in saved]:
        del sys.modules[name]
    sys.modules.update(saved)
    for name, module in saved.items():
        parent, _, child = name.rpartition(".")
        if parent in saved:
            setattr(saved[parent], child, module)


def _clear_entry_modules() -> None:
    for name in list(sys.modules):
        if name == "jasna.__main__" or name.startswith("jasna.gui"):
            del sys.modules[name]


def _run_entry():
    """Import ``jasna.__main__`` afresh, which dispatches, with the GUI and CLI mocked; returns their mocks."""
    _clear_entry_modules()
    with patch("jasna.gui.run_gui") as gui, patch("jasna.main.main") as cli:
        import jasna.__main__  # noqa: F401
    return gui, cli


@pytest.mark.parametrize(
    ("platform", "argv", "runs"),
    [
        ("linux", ["jasna"], "gui"),
        ("linux", ["jasna", "--version"], "cli"),
        ("win32", [r"C:\Program Files\Jasna\jasna-cli.exe", "--version"], "cli"),
        ("win32", [r"C:\Program Files\Jasna\jasna-gui.exe", "--version"], "gui"),
    ],
    ids=["no-args", "linux-args", "windows-cli", "windows-gui"],
)
def test_entry_dispatches_to_the_gui_or_the_cli(monkeypatch, platform, argv, runs):
    monkeypatch.setattr(sys, "platform", platform, raising=False)
    monkeypatch.setattr(sys, "argv", argv)

    gui, cli = _run_entry()

    assert (gui.called, cli.called) == (runs == "gui", runs == "cli")


@pytest.mark.parametrize("probe", [None, "0", "1"])
def test_gui_font_probe_preserves_console_output(monkeypatch, probe) -> None:
    monkeypatch.setattr(sys, "argv", ["jasna"])
    if probe is None:
        monkeypatch.delenv("JASNA_GUI_FONT_PROBE", raising=False)
    else:
        monkeypatch.setenv("JASNA_GUI_FONT_PROBE", probe)

    with patch("jasna.os_utils.drop_console_window") as drop_console:
        gui, _ = _run_entry()

    gui.assert_called_once()
    assert drop_console.call_count == (0 if probe == "1" else 1)


def test_windows_jasna_cli_no_args_prints_help_and_exits_0(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.setattr(sys, "argv", [r"C:\Program Files\Jasna\jasna-cli.exe"])

    _clear_entry_modules()
    with patch("jasna.main.build_parser") as build_parser, pytest.raises(SystemExit) as exit_info:
        import jasna.__main__  # noqa: F401

    assert exit_info.value.code == 0
    build_parser.return_value.print_help.assert_called_once()
    assert "jasna.gui" not in sys.modules


@pytest.mark.parametrize("child", ["spawned", "other-pid"])
def test_child_processes_exit_without_dispatch(monkeypatch, child) -> None:
    monkeypatch.setattr(sys, "argv", ["jasna"])
    if child == "spawned":
        monkeypatch.setattr("multiprocessing.parent_process", lambda: MagicMock(), raising=False)
    else:
        monkeypatch.setattr(os, "getpid", lambda: 99999)
        monkeypatch.setenv("JASNA_MAIN_PID", "12345")

    _clear_entry_modules()
    with patch("jasna.gui.run_gui") as gui, patch("jasna.main.main") as cli, pytest.raises(SystemExit) as exit_info:
        import jasna.__main__  # noqa: F401

    assert exit_info.value.code == 0
    assert not gui.called and not cli.called


def test_startup_configures_windows_dll_search_paths(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["jasna"])
    with patch("jasna.packaging.windows_dll_paths.configure_windows_dll_search_paths") as configure:
        _run_entry()
    configure.assert_called_once()


def test_ltx_worker_dispatch_precedes_parent_pid_and_gui(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_MAIN_PID", "12345")
    monkeypatch.setattr(sys, "argv", ["jasna", "--ltx-worker"])
    worker = MagicMock()
    monkeypatch.setitem(sys.modules, "jasna.protection.ltx_worker", SimpleNamespace(worker_main=worker))

    _clear_entry_modules()
    with patch("jasna.gui.run_gui") as gui, patch("jasna.main.main") as cli, pytest.raises(SystemExit) as exit_info:
        import jasna.__main__  # noqa: F401

    assert exit_info.value.code == 0
    worker.assert_called_once_with()
    assert not gui.called and not cli.called


def test_ltx_worker_rejects_extra_launch_arguments(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["jasna", "--ltx-worker", "untrusted"])
    with pytest.raises(SystemExit, match="Invalid LTX worker launch"):
        _run_entry()
