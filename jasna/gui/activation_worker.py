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


def license_status_key(mode: str | None) -> str:
    """Locale key that names how an activated Jasna checks the license; ``mode`` comes from ``read_license_info``."""
    return {"offline": "license_offline_active", "online": "license_lease_active"}.get(mode, "license_active")


def start_license_info(on_done) -> threading.Thread:
    return _start(read_license_info, on_done)


def read_unofficial_notice_due() -> bool:
    """Whether this launch shows the note for buyers of an unofficial copy; only the Chinese UI asks."""
    if not protection_available():
        return False
    from jasna.protection import notice

    return notice.due()


def start_unofficial_notice_check(on_done) -> threading.Thread:
    return _start(read_unofficial_notice_due, on_done)


def start_device_count(on_done) -> threading.Thread:
    """Calls ``on_done((used, limit, offline PCs, online devices), error)``; asks the license service."""

    def count():
        require_protection()
        from jasna.protection import lease

        return lease.device_count()

    return _start(count, on_done)


def start_lease_activation(code: str, on_done) -> threading.Thread:
    """Calls ``on_done(Activated, error)``: works offline on PCs with a TPM, otherwise checks online."""

    def activate():
        require_protection()
        from jasna.protection import lease

        return lease.activate_license_code(code)

    return _start(activate, on_done)
