import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.post_export_action import (
    PostExportVideoCommandCancelled,
    PostExportVideoCommandError,
    _quote_shell_value,
    expand_post_export_video_command,
    run_post_export_action,
    run_post_export_action_safely,
    run_post_export_video_command,
    uses_log_placeholder,
    validate_post_export_action,
)

posix_paths = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell quoting of POSIX paths")


def test_validate_post_export_command_requires_command() -> None:
    with pytest.raises(ValueError, match="post-export-command"):
        validate_post_export_action("command", "")


def test_run_post_export_none_does_not_spawn(monkeypatch) -> None:
    def fail_popen(*_args, **_kwargs):
        raise AssertionError("Popen should not be called")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    run_post_export_action("none")


def test_run_post_export_shutdown_windows(monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "win32")
    monkeypatch.setattr("jasna.post_export_action.subprocess_no_window_kwargs", lambda: {"creationflags": 1})
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)))

    run_post_export_action("shutdown")

    assert calls == [(["shutdown", "/s", "/t", "0"], {"env": None, "creationflags": 1})]


def test_run_post_export_shutdown_linux_without_bundled_libraries(
    monkeypatch, linux_bundled_library_path
) -> None:
    # Issue #822: systemctl failed on the bundled libcrypto.
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)))

    run_post_export_action("shutdown")

    [(cmd, kwargs)] = calls
    assert cmd == ["systemctl", "poweroff"]
    assert kwargs["env"]["LD_LIBRARY_PATH"] == linux_bundled_library_path


def test_run_post_export_custom_command_uses_shell_without_bundled_libraries(
    monkeypatch, linux_bundled_library_path
) -> None:
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)))

    run_post_export_action("command", "  echo done  ")

    [(cmd, kwargs)] = calls
    assert (cmd, kwargs["shell"]) == ("echo done", True)
    assert kwargs["env"]["LD_LIBRARY_PATH"] == linux_bundled_library_path


def test_run_post_export_safely_reports_spawn_error(monkeypatch) -> None:
    errors: list[str] = []

    def fail_popen(*_args, **_kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    assert run_post_export_action_safely("shutdown", "", errors.append) is False
    assert errors == ["Post-export action failed: cannot spawn"]


@posix_paths
def test_expand_post_export_video_command_quotes_paths(monkeypatch) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")

    expanded = expand_post_export_video_command(
        "tool {input} {output} {output_dir}/{output_stem}_remuxed{output_suffix}",
        Path("/videos/source file.mp4"),
        Path("/videos/out/output file.mp4"),
    )

    assert expanded == "tool '/videos/source file.mp4' '/videos/out/output file.mp4' /videos/out/'output file'_remuxed.mp4"


@posix_paths
def test_expand_post_export_video_command_quotes_log_path(monkeypatch) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")

    expanded = expand_post_export_video_command(
        "cp {log} /logs/",
        Path("/videos/in.mp4"),
        Path("/videos/output file.mp4"),
        log_path=Path("/videos/output file.log"),
    )

    assert expanded == "cp '/videos/output file.log' /logs/"


def test_expansion_without_a_log_file_leaves_log_as_typed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")

    expanded = expand_post_export_video_command("cp {log} /logs/", tmp_path / "in.mp4", tmp_path / "out.mp4")

    assert expanded == "cp {log} /logs/"


@pytest.mark.parametrize(
    ("command", "uses_log"),
    [("cp {log} /logs/", True), ("remux {output}", False), ("echo {logs}", False)],
)
def test_uses_log_placeholder(command: str, uses_log: bool) -> None:
    assert uses_log_placeholder(command) is uses_log


@posix_paths
def test_expansion_does_not_reprocess_placeholders_inside_paths(monkeypatch) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")

    expanded = expand_post_export_video_command("tool {input}", Path("/videos/{output_dir}.mp4"), Path("/videos/out.mp4"))

    assert expanded == "tool '/videos/{output_dir}.mp4'"


def test_expand_post_export_video_command_quotes_windows_values(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "win32")
    output_path = tmp_path / "output file.mp4"

    expanded = expand_post_export_video_command(
        "tool {output}",
        tmp_path / "in.mp4",
        output_path,
    )

    assert expanded == f"tool {subprocess.list2cmdline([str(output_path)])}"


def test_expand_post_export_video_command_quotes_cmd_operators_on_windows(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "win32")
    output_path = tmp_path / "Tom&Jerry^1.mp4"

    expanded = expand_post_export_video_command(
        "tool {output}",
        tmp_path / "in.mp4",
        output_path,
    )

    assert expanded == f'tool "{output_path}"'


def test_windows_quoting_keeps_trailing_backslash_inside_quotes(monkeypatch) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "win32")

    assert _quote_shell_value("C:\\") == '"C:\\\\"'


@posix_paths
def test_run_post_export_video_command_runs_the_expanded_command_without_bundled_libraries(
    monkeypatch, linux_bundled_library_path
) -> None:
    process = MagicMock(pid=123)
    process.wait.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("jasna.post_export_action.subprocess.Popen", popen)

    run_post_export_video_command("tool {output}", Path("/videos/in.mp4"), Path("/videos/out file.mp4"), lambda: False)

    assert popen.call_args.args[0] == "tool '/videos/out file.mp4'"
    assert popen.call_args.kwargs["env"]["LD_LIBRARY_PATH"] == linux_bundled_library_path


def test_run_post_export_video_command_reports_nonzero_exit(monkeypatch, tmp_path: Path) -> None:
    process = MagicMock(pid=123)
    process.wait.return_value = 7
    monkeypatch.setattr("jasna.post_export_action.subprocess.Popen", MagicMock(return_value=process))

    with pytest.raises(PostExportVideoCommandError, match="exit code 7"):
        run_post_export_video_command(
            "tool",
            tmp_path / "in.mp4",
            tmp_path / "out.mp4",
            lambda: False,
        )


def test_run_post_export_video_command_reports_spawn_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "jasna.post_export_action.subprocess.Popen",
        MagicMock(side_effect=OSError("cannot spawn")),
    )

    with pytest.raises(PostExportVideoCommandError, match="cannot spawn"):
        run_post_export_video_command(
            "tool",
            tmp_path / "in.mp4",
            tmp_path / "out.mp4",
            lambda: False,
        )


def test_run_post_export_video_command_cancels_process(monkeypatch, tmp_path: Path) -> None:
    process = MagicMock(pid=123)
    terminate = MagicMock()
    monkeypatch.setattr("jasna.post_export_action.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("jasna.post_export_action._terminate_process_tree", terminate)

    with pytest.raises(PostExportVideoCommandCancelled):
        run_post_export_video_command(
            "tool",
            tmp_path / "in.mp4",
            tmp_path / "out.mp4",
            lambda: True,
        )

    terminate.assert_called_once_with(process)
