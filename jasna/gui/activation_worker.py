"""Activation, config access and the update check, kept off the Tk thread."""

import threading
import time
from collections.abc import Callable
from typing import TypeVar

from jasna import license_status, update_check
from jasna.license_api import require_protection, typed_license_code

Result = TypeVar("Result")


def _start(
    operation: Callable[[], Result],
    on_done: Callable[[Result | None, Exception | None], None],
) -> threading.Thread:
    def worker():
        try:
            result = operation()
        except Exception as exc:
            on_done(None, exc)
        else:
            on_done(result, None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def start_license_info(on_done) -> threading.Thread:
    return _start(license_status.read_license_info, on_done)


def start_unofficial_notice_check(on_done) -> threading.Thread:
    return _start(license_status.read_unofficial_notice_due, on_done)


def start_update_check(on_done) -> threading.Thread:
    """Calls ``on_done(version, error)`` with the newer release to show, or None."""
    return _start(lambda: update_check.newer_release(time.time()), on_done)


def start_device_count(on_done) -> threading.Thread:
    """Calls ``on_done(DeviceUse, error)``; asks the license service."""

    def count():
        require_protection()
        from jasna.protection import lease

        return lease.device_count()

    return _start(count, on_done)


def start_lease_activation(code: str, try_offline: bool, on_done) -> threading.Thread:
    """Calls ``on_done(Activated, error)``: with ``try_offline``, works offline on PCs with a TPM; otherwise checks online."""

    def activate():
        require_protection()
        from jasna.protection import lease

        return lease.activate_license_code(typed_license_code(code), try_offline)

    return _start(activate, on_done)
