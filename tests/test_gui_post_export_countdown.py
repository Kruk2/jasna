"""Queue-wide post-export action: read at queue end, shutdown behind a cancellable countdown."""

from __future__ import annotations

import time
from tkinter import TclError
from unittest.mock import MagicMock, patch

import customtkinter as ctk
import pytest

from jasna.gui.app import JasnaApp
from jasna.gui.components import ShutdownCountdownDialog
from jasna.gui.models import AppSettings


@pytest.fixture
def root():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


def _pump_until(root, condition, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)


def _app_with_settings(root, settings: AppSettings):
    root._settings_panel = MagicMock(get_settings=MagicMock(return_value=settings))
    root._log_panel = MagicMock()
    return root


def _countdown_dialogs(root) -> list[ShutdownCountdownDialog]:
    return [w for w in root.winfo_children() if isinstance(w, ShutdownCountdownDialog)]


def test_countdown_shows_remaining_seconds_and_expires(root):
    expired: list[bool] = []
    cancelled: list[bool] = []
    dialog = ShutdownCountdownDialog(root, 2, lambda: expired.append(True), lambda: cancelled.append(True))
    assert "2" in dialog._message.cget("text")

    _pump_until(root, lambda: "1" in dialog._message.cget("text"), timeout=3)
    assert "1" in dialog._message.cget("text")
    assert expired == []

    _pump_until(root, lambda: expired, timeout=3)
    assert expired == [True]
    assert cancelled == []
    assert not dialog.winfo_exists()


def test_countdown_cancel_stops_shutdown(root):
    expired: list[bool] = []
    cancelled: list[bool] = []
    dialog = ShutdownCountdownDialog(root, 1, lambda: expired.append(True), lambda: cancelled.append(True))

    dialog.cancel_button.invoke()
    _pump_until(root, lambda: expired, timeout=1.5)

    assert cancelled == [True]
    assert expired == []
    assert not dialog.winfo_exists()


def test_queue_end_uses_current_command_settings(root):
    app = _app_with_settings(root, AppSettings(post_export_action="command", post_export_command="notify"))

    with patch("jasna.post_export_action.run_post_export_action") as run:
        JasnaApp._run_post_export_action(app)

    run.assert_called_once_with("command", "notify")
    assert _countdown_dialogs(root) == []


def test_queue_end_does_nothing_when_action_was_turned_off(root):
    app = _app_with_settings(root, AppSettings(post_export_action="none"))

    with patch("jasna.post_export_action.run_post_export_action") as run:
        JasnaApp._run_post_export_action(app)

    run.assert_not_called()
    assert _countdown_dialogs(root) == []


def test_queue_end_shutdown_waits_for_countdown_and_can_be_cancelled(root):
    app = _app_with_settings(root, AppSettings(post_export_action="shutdown"))

    with patch("jasna.post_export_action.run_post_export_action") as run:
        JasnaApp._run_post_export_action(app)
        (dialog,) = _countdown_dialogs(root)
        dialog.cancel_button.invoke()

    run.assert_not_called()
    app._log_panel.info.assert_called_with("Shutdown cancelled by user")
