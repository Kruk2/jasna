"""Queue-wide post-export action: read at queue end, shutdown behind a cancellable countdown."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jasna.gui import components
from jasna.gui.components import ShutdownCountdownDialog, countdown_tick
from jasna.gui.models import AppSettings


def _use_settings(app, settings: AppSettings) -> None:
    app._preset_manager.create_preset("test", settings)
    app._settings_panel._apply_preset("test")


def _countdown_dialogs(app) -> list[ShutdownCountdownDialog]:
    return [w for w in app.winfo_children() if isinstance(w, ShutdownCountdownDialog)]


def _logged(app) -> list[str]:
    return [message for _, _, message in app._log_panel._entries]


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(components.time, "monotonic", lambda: now[0])
    return now


@pytest.mark.parametrize(
    ("time_left", "expected"),
    [
        (60.0, (60, 1000)),
        (59.75, (60, 750)),
        (0.5, (1, 500)),
    ],
)
def test_countdown_tick_shows_whole_seconds_until_the_next_drop(time_left, expected):
    assert countdown_tick(time_left) == expected


@pytest.mark.parametrize("time_left", [0.0, -5.0])
def test_countdown_tick_shows_zero_once_expired(time_left):
    assert countdown_tick(time_left)[0] == 0


def test_countdown_shows_remaining_seconds_and_expires(tk_root, clock):
    expired: list[bool] = []
    cancelled: list[bool] = []
    dialog = ShutdownCountdownDialog(tk_root, 2, lambda: expired.append(True), lambda: cancelled.append(True))
    assert "2" in dialog._message.cget("text")

    clock[0] += 1
    dialog._tick()
    assert "1" in dialog._message.cget("text")
    assert expired == []

    clock[0] += 1
    dialog._tick()
    assert expired == [True]
    assert cancelled == []
    assert not dialog.winfo_exists()


def _locked_screen_never_shows(window) -> None:
    pytest.fail("waited for a window that a locked screen never shows")


def test_countdown_runs_while_the_screen_is_locked(tk_root, clock, monkeypatch):
    monkeypatch.setattr(ShutdownCountdownDialog, "wait_visibility", _locked_screen_never_shows)
    expired: list[bool] = []
    dialog = ShutdownCountdownDialog(tk_root, 60, lambda: expired.append(True), lambda: None)

    dialog.withdraw()
    tk_root.update()
    assert not dialog.winfo_viewable()

    clock[0] += 60
    dialog._tick()
    assert expired == [True]
    assert not dialog.winfo_exists()


def test_countdown_takes_the_grab_once_shown(tk_root, show_window):
    dialog = ShutdownCountdownDialog(tk_root, 60, lambda: None, lambda: None)

    show_window(dialog)

    assert tk_root.grab_current() is dialog
    dialog.cancel()


def test_countdown_cancel_stops_shutdown(tk_root):
    expired: list[bool] = []
    cancelled: list[bool] = []
    dialog = ShutdownCountdownDialog(tk_root, 1, lambda: expired.append(True), lambda: cancelled.append(True))

    dialog.cancel_button.invoke()

    assert cancelled == [True]
    assert expired == []
    assert not dialog.winfo_exists()


@pytest.mark.parametrize(
    ("settings", "expected_runs"),
    [
        (AppSettings(post_export_action="command", post_export_command="notify"), [("command", "notify")]),
        (AppSettings(post_export_action="none"), []),
    ],
)
def test_queue_end_runs_the_current_action(jasna_app, settings, expected_runs):
    _use_settings(jasna_app, settings)

    with patch("jasna.post_export_action.run_post_export_action") as run:
        jasna_app._handle_complete(queue_finished=True)

    assert [call.args for call in run.call_args_list] == expected_runs
    assert _countdown_dialogs(jasna_app) == []


def test_queue_end_shutdown_waits_for_countdown_and_can_be_cancelled(jasna_app):
    _use_settings(jasna_app, AppSettings(post_export_action="shutdown"))

    with patch("jasna.post_export_action.run_post_export_action") as run:
        jasna_app._handle_complete(queue_finished=True)
        (dialog,) = _countdown_dialogs(jasna_app)
        dialog.cancel_button.invoke()

    run.assert_not_called()
    assert _logged(jasna_app)[-1] == "Shutdown cancelled by user"
