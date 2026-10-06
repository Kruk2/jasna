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


def test_no_args_dispatches_to_gui() -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    with patch.object(sys, "argv", ["jasna"]):
        with patch("jasna.gui.run_gui") as run_gui:
            with patch("jasna.main.main"):
                import jasna.__main__  # noqa: F401
                run_gui.assert_called_once()


@pytest.mark.parametrize("probe", [None, "0", "1"])
def test_gui_font_probe_preserves_console_output(monkeypatch, probe) -> None:
    sys.modules.pop("jasna.__main__", None)
    monkeypatch.setattr(sys, "argv", ["jasna"])
    if probe is None:
        monkeypatch.delenv("JASNA_GUI_FONT_PROBE", raising=False)
    else:
        monkeypatch.setenv("JASNA_GUI_FONT_PROBE", probe)

    with patch("jasna.os_utils.drop_console_window") as drop_console:
        with patch("jasna.gui.run_gui") as run_gui:
            import jasna.__main__  # noqa: F401

    run_gui.assert_called_once()
    assert drop_console.call_count == (0 if probe == "1" else 1)


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
            with pytest.raises(SystemExit) as e:
                import jasna.__main__  # noqa: F401
            assert e.value.code == 0
            run_gui.assert_not_called()
            main.assert_not_called()


def test_startup_configures_windows_dll_search_paths() -> None:
    if "jasna.__main__" in sys.modules:
        del sys.modules["jasna.__main__"]
    with patch.object(sys, "argv", ["jasna"]):
        with patch(
            "jasna.packaging.windows_dll_paths.configure_windows_dll_search_paths"
        ) as configure:
            with patch("jasna.gui.run_gui"):
                with patch("jasna.main.main"):
                    import jasna.__main__  # noqa: F401
    configure.assert_called_once()


def test_no_progress_flag_defaults_false() -> None:
    from jasna.main import build_parser
    args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4"])
    assert args.no_progress is False


def test_no_progress_flag_set() -> None:
    from jasna.main import build_parser
    args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4", "--no-progress"])
    assert args.no_progress is True


def test_jasna_main_pid_child_exits_without_dispatch(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "jasna.__main__" or name.startswith("jasna.gui"):
            del sys.modules[name]

    monkeypatch.setattr(os, "getpid", lambda: 99999)
    monkeypatch.setenv("JASNA_MAIN_PID", "12345")
    monkeypatch.setattr(sys, "argv", ["jasna"])

    with patch("jasna.gui.run_gui") as run_gui:
        with patch("jasna.main.main") as main:
            with pytest.raises(SystemExit) as e:
                import jasna.__main__  # noqa: F401
            assert e.value.code == 0
            run_gui.assert_not_called()
            main.assert_not_called()


def test_ltx_worker_dispatch_precedes_parent_pid_and_gui(monkeypatch) -> None:
    sys.modules.pop("jasna.__main__", None)
    monkeypatch.setenv("JASNA_MAIN_PID", "12345")
    monkeypatch.setattr(sys, "argv", ["jasna", "--ltx-worker"])
    worker = MagicMock()
    monkeypatch.setitem(sys.modules, "jasna.protection.ltx_worker", SimpleNamespace(worker_main=worker))
    with patch("jasna.gui.run_gui") as gui, patch("jasna.main.main") as cli:
        with pytest.raises(SystemExit) as exit_info:
            import jasna.__main__  # noqa: F401
    assert exit_info.value.code == 0
    worker.assert_called_once_with()
    gui.assert_not_called()
    cli.assert_not_called()


def test_ltx_worker_rejects_extra_launch_arguments(monkeypatch) -> None:
    sys.modules.pop("jasna.__main__", None)
    monkeypatch.setattr(sys, "argv", ["jasna", "--ltx-worker", "untrusted"])
    with patch("jasna.gui.run_gui") as gui, patch("jasna.main.main") as cli:
        with pytest.raises(SystemExit, match="Invalid LTX worker launch"):
            import jasna.__main__  # noqa: F401
    gui.assert_not_called()
    cli.assert_not_called()
