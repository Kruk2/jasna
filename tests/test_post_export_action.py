import subprocess

import pytest

from jasna.post_export_action import run_post_export_action, run_post_export_action_safely, validate_post_export_action


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
