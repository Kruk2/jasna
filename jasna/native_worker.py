"""Recovery contracts for isolated Linux AMD video workers.

AMF/PyAV calls can block inside native code after a driver-pressure event.  A
Python thread cannot safely cancel such a call, so GUI video jobs use a child
process as the recovery boundary.  This module is intentionally lightweight so
the decoder, pipeline, and GUI protocol can share the same exit contracts.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import TypeVar


ISOLATED_VIDEO_JOB_ENV = "JASNA_ISOLATED_VIDEO_JOB"
AMF_DECODER_OPEN_TIMEOUT_ENV = "JASNA_AMF_DECODER_OPEN_TIMEOUT_SECONDS"
AMF_RENDER_SESSION_SECONDS_ENV = "JASNA_AMF_RENDER_SESSION_SECONDS"
AMF_ENCODER_STALL_TIMEOUT_ENV = "JASNA_AMF_ENCODER_STALL_TIMEOUT_SECONDS"
NATIVE_PRESSURE_RECYCLE_EXIT_CODE = 75
NATIVE_OPEN_STALL_EXIT_CODE = 86
NATIVE_ENCODE_STALL_EXIT_CODE = 87
DEFAULT_AMF_DECODER_OPEN_TIMEOUT_SECONDS = 60.0
DEFAULT_AMF_RENDER_SESSION_SECONDS = 120.0
DEFAULT_AMF_ENCODER_STALL_TIMEOUT_SECONDS = 90.0

_T = TypeVar("_T")
_log = logging.getLogger(__name__)


class NativeWorkerRecycleRequested(RuntimeError):
    """Ask the GUI parent to restart this worker and resume its workspace."""

    def __init__(self, message: str, *, reason: str = "native_pressure") -> None:
        super().__init__(message)
        self.reason = str(reason)


class HostMemoryPressureError(RuntimeError):
    """Abort a video worker before the host OOM killer terminates it."""

    reason = "host_memory_pressure"


def is_isolated_video_job() -> bool:
    return os.environ.get(ISOLATED_VIDEO_JOB_ENV, "").strip() == "1"


def amf_decoder_open_timeout_seconds() -> float:
    raw = os.environ.get(AMF_DECODER_OPEN_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_AMF_DECODER_OPEN_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_AMF_DECODER_OPEN_TIMEOUT_SECONDS
    return max(5.0, value)


def amf_render_session_seconds() -> float | None:
    """Return the maximum affected AMF render time per isolated process."""

    raw = os.environ.get(AMF_RENDER_SESSION_SECONDS_ENV, "").strip().casefold()
    if not raw:
        return DEFAULT_AMF_RENDER_SESSION_SECONDS
    if raw in {"0", "off", "false", "no"}:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {AMF_RENDER_SESSION_SECONDS_ENV} value {raw!r}; "
            "expected 'off' or seconds >= 1"
        ) from exc
    if value < 1.0:
        raise ValueError(
            f"Invalid {AMF_RENDER_SESSION_SECONDS_ENV} value {raw!r}; "
            "expected 'off' or seconds >= 1"
        )
    return value


def amf_encoder_stall_timeout_seconds() -> float:
    """Return the fail-closed timeout for an isolated native encode call."""

    raw = os.environ.get(AMF_ENCODER_STALL_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_AMF_ENCODER_STALL_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {AMF_ENCODER_STALL_TIMEOUT_ENV} value {raw!r}; "
            "expected seconds >= 30"
        ) from exc
    if value < 30.0:
        raise ValueError(
            f"Invalid {AMF_ENCODER_STALL_TIMEOUT_ENV} value {raw!r}; "
            "expected seconds >= 30"
        )
    return value


def run_amf_decoder_open_with_watchdog(
    open_call: Callable[[], _T],
    *,
    description: str,
    timeout_seconds: float | None = None,
) -> _T:
    """Run one native AMF open, exiting only an isolated worker if it stalls.

    Continuing after a timed-out native call is unsafe because the blocked
    thread can still own AMF/Vulkan resources.  The watchdog therefore exits
    the entire child process.  The GUI parent recognizes the dedicated exit
    code, waits for whole-card VRAM to recover, and resumes verified Smart
    Render fragments in a fresh worker.  CLI and non-isolated callers retain
    the established direct behavior.
    """

    if not is_isolated_video_job():
        return open_call()

    timeout = (
        amf_decoder_open_timeout_seconds()
        if timeout_seconds is None
        else max(0.001, float(timeout_seconds))
    )
    completed = threading.Event()

    def watchdog() -> None:
        if completed.wait(timeout):
            return
        _log.critical(
            "AMF decoder open stalled for %.1f seconds (%s); terminating the "
            "isolated worker so the GUI can release native VRAM and resume",
            timeout,
            description,
        )
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(NATIVE_OPEN_STALL_EXIT_CODE)

    threading.Thread(
        target=watchdog,
        daemon=True,
        name="amf-decoder-open-watchdog",
    ).start()
    try:
        return open_call()
    finally:
        completed.set()
