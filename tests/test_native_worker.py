from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from jasna.native_worker import (
    AMF_ENCODER_STALL_TIMEOUT_ENV,
    AMF_RENDER_SESSION_SECONDS_ENV,
    DEFAULT_AMF_ENCODER_STALL_TIMEOUT_SECONDS,
    DEFAULT_AMF_RENDER_SESSION_SECONDS,
    ISOLATED_VIDEO_JOB_ENV,
    NATIVE_OPEN_STALL_EXIT_CODE,
    amf_encoder_stall_timeout_seconds,
    amf_render_session_seconds,
    run_amf_decoder_open_with_watchdog,
)


def test_amf_encoder_stall_timeout_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv(AMF_ENCODER_STALL_TIMEOUT_ENV, raising=False)
    assert (
        amf_encoder_stall_timeout_seconds()
        == DEFAULT_AMF_ENCODER_STALL_TIMEOUT_SECONDS
    )
    monkeypatch.setenv(AMF_ENCODER_STALL_TIMEOUT_ENV, "45")
    assert amf_encoder_stall_timeout_seconds() == 45.0
    monkeypatch.setenv(AMF_ENCODER_STALL_TIMEOUT_ENV, "0")
    with pytest.raises(ValueError, match=AMF_ENCODER_STALL_TIMEOUT_ENV):
        amf_encoder_stall_timeout_seconds()


def test_amf_render_session_uses_bounded_default_and_explicit_off(monkeypatch) -> None:
    monkeypatch.delenv(AMF_RENDER_SESSION_SECONDS_ENV, raising=False)
    assert amf_render_session_seconds() == DEFAULT_AMF_RENDER_SESSION_SECONDS
    monkeypatch.setenv(AMF_RENDER_SESSION_SECONDS_ENV, "90")
    assert amf_render_session_seconds() == 90.0
    monkeypatch.setenv(AMF_RENDER_SESSION_SECONDS_ENV, "off")
    assert amf_render_session_seconds() is None


def test_amf_render_session_rejects_invalid_override(monkeypatch) -> None:
    monkeypatch.setenv(AMF_RENDER_SESSION_SECONDS_ENV, "0.5")
    with pytest.raises(ValueError, match=AMF_RENDER_SESSION_SECONDS_ENV):
        amf_render_session_seconds()


def test_amf_open_watchdog_is_disabled_outside_isolated_worker(monkeypatch) -> None:
    import jasna.native_worker as module

    monkeypatch.delenv(ISOLATED_VIDEO_JOB_ENV, raising=False)
    exit_process = MagicMock()
    monkeypatch.setattr(module.os, "_exit", exit_process)

    assert run_amf_decoder_open_with_watchdog(
        lambda: "opened",
        description="test decoder",
        timeout_seconds=0.001,
    ) == "opened"
    time.sleep(0.01)
    exit_process.assert_not_called()


def test_amf_open_watchdog_does_not_fire_after_success(monkeypatch) -> None:
    import jasna.native_worker as module

    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")
    exit_process = MagicMock()
    monkeypatch.setattr(module.os, "_exit", exit_process)

    assert run_amf_decoder_open_with_watchdog(
        lambda: 42,
        description="test decoder",
        timeout_seconds=0.01,
    ) == 42
    time.sleep(0.03)
    exit_process.assert_not_called()


def test_amf_open_watchdog_exits_stalled_isolated_worker(monkeypatch) -> None:
    import jasna.native_worker as module

    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")
    release_open = threading.Event()
    exit_called = threading.Event()
    exit_codes = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        exit_called.set()

    monkeypatch.setattr(module.os, "_exit", fake_exit)
    runner = threading.Thread(
        target=lambda: run_amf_decoder_open_with_watchdog(
            release_open.wait,
            description="stalled decoder",
            timeout_seconds=0.01,
        )
    )
    runner.start()

    assert exit_called.wait(1.0)
    release_open.set()
    runner.join(timeout=1.0)

    assert not runner.is_alive()
    assert exit_codes == [NATIVE_OPEN_STALL_EXIT_CODE]
