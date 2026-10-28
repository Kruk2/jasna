"""Activation and config access, kept off the Tk thread."""

import threading
from collections.abc import Callable
from typing import TypeVar

from jasna.license_api import protection_available, require_protection

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


def read_license_info() -> tuple[bool, str | None]:
    """Whether Jasna is activated, and how it checks the license: "offline", "online", or None while an earlier
    activation waits to move on the first supporter-model job."""
    if not protection_available():
        return False, None
    from jasna.protection import lease, license_store, offline

    if offline.saved_grant() is not None:
        mode = "offline"
    elif lease.load_token() is not None:
        mode = "online"
    else:
        mode = None
    return license_store.is_licensed(), mode


def start_license_info(on_done) -> threading.Thread:
    return _start(read_license_info, on_done)


def start_lease_activation(code: str, on_done) -> threading.Thread:
    """Calls ``on_done(Activated, error)``: offline on a certified TPM, else online leases."""

    def activate():
        require_protection()
        from jasna.protection import lease

        return lease.activate_license_code(code)

    return _start(activate, on_done)
