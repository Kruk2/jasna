"""Activation, config access and the update check, kept off the Tk thread."""

import threading
import time

from jasna import license_status, update_check
from jasna.gui.workers import run_in_background
from jasna.license_api import require_protection, typed_license_code


def start_license_info(on_done) -> threading.Thread:
    return run_in_background(license_status.read_license_info, on_done)


def start_unofficial_notice_check(on_done) -> threading.Thread:
    return run_in_background(license_status.read_unofficial_notice_due, on_done)


def start_update_check(on_done) -> threading.Thread:
    """Calls ``on_done(version, error)`` with the newer release to show, or None."""
    return run_in_background(lambda: update_check.newer_release(time.time()), on_done)


def start_device_count(on_done) -> threading.Thread:
    """Calls ``on_done(DeviceUse, error)``; asks the license service."""

    def count():
        require_protection()
        from jasna.protection import lease

        return lease.device_count()

    return run_in_background(count, on_done)


def start_lease_activation(code: str, try_offline: bool, on_done) -> threading.Thread:
    """Calls ``on_done(Activated, error)``: with ``try_offline``, works offline on PCs with a TPM; otherwise checks online."""

    def activate():
        require_protection()
        from jasna.protection import lease

        return lease.activate_license_code(typed_license_code(code), try_offline)

    return run_in_background(activate, on_done)
