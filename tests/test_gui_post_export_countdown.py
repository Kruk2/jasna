"""Queue-wide post-export action: read at queue end, shutdown behind a cancellable countdown."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jasna.gui.components import ShutdownCountdownDialog
from jasna.gui.models import AppSettings


def _use_settings(app, settings: AppSettings) -> None:
    app._preset_manager.create_preset("test", settings)
    app._settings_panel._apply_preset("test")


def _countdown_dialogs(app) -> list[ShutdownCountdownDialog]:
    return [w for w in app.winfo_children() if isinstance(w, ShutdownCountdownDialog)]


def _logged(app) -> list[str]:
    return [message for _, _, message in app._log_panel._entries]


def test_countdown_shows_remaining_seconds_and_expires(tk_root):
    expired: list[bool] = []
    cancelled: list[bool] = []
    dialog = ShutdownCountdownDialog(tk_root, 2, lambda: expired.append(True), lambda: cancelled.append(True))
    assert "2" in dialog._message.cget("text")

    dialog._tick()
    assert "1" in dialog._message.cget("text")
    assert expired == []

    dialog._tick()
    assert expired == [True]
    assert cancelled == []
    assert not dialog.winfo_exists()


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
