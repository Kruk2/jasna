import subprocess
import shlex
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.post_export_action import (
    PostExportVideoCommandCancelled,
    PostExportVideoCommandError,
    expand_post_export_video_command,
    run_post_export_action,
    run_post_export_action_safely,
    run_post_export_video_command,
    validate_post_export_action,
)


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

    assert calls == [(["shutdown", "/s", "/t", "0"], {"creationflags": 1})]


def test_run_post_export_shutdown_linux(monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")
    monkeypatch.setattr("jasna.post_export_action.subprocess_no_window_kwargs", lambda: {})
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)))

    run_post_export_action("shutdown")

    assert calls == [(["shutdown", "-h", "now"], {})]


def test_run_post_export_custom_command_uses_shell(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("jasna.post_export_action.subprocess_no_window_kwargs", lambda: {})
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)))

    run_post_export_action("command", "  echo done  ")

    assert calls == [("echo done", {"shell": True})]


def test_run_post_export_safely_reports_spawn_error(monkeypatch) -> None:
    errors: list[str] = []

    def fail_popen(*_args, **_kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    assert run_post_export_action_safely("shutdown", "", errors.append) is False
    assert errors == ["Post-export action failed: cannot spawn"]


def test_expand_post_export_video_command_quotes_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")
    input_path = tmp_path / "source file.mp4"
    output_path = tmp_path / "output file.mp4"

    expanded = expand_post_export_video_command(
        "tool {input} {output} {output_dir}/{output_stem}_remuxed{output_suffix}",
        input_path,
        output_path,
    )

    assert expanded == (
        f"tool {shlex.quote(str(input_path))} {shlex.quote(str(output_path))} "
        f"{shlex.quote(str(tmp_path))}/{shlex.quote('output file')}_remuxed.mp4"
    )


def test_expansion_does_not_reprocess_placeholders_inside_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")
    input_path = tmp_path / "{output_dir}.mp4"

    expanded = expand_post_export_video_command(
        "tool {input}",
        input_path,
        tmp_path / "out.mp4",
    )

    assert expanded == f"tool {shlex.quote(str(input_path))}"


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


def test_run_post_export_video_command_waits_for_success(monkeypatch, tmp_path: Path) -> None:
    process = MagicMock(pid=123)
    process.wait.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("jasna.post_export_action.subprocess.Popen", popen)
    monkeypatch.setattr("jasna.post_export_action.sys.platform", "linux")
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"

    run_post_export_video_command(
        "tool {output}",
        input_path,
        output_path,
        lambda: False,
    )

    popen.assert_called_once_with(
        f"tool {output_path}",
        shell=True,
        cwd=tmp_path,
        start_new_session=True,
    )
    process.wait.assert_called_once_with(timeout=0.1)


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
