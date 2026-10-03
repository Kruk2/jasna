"""Activation and config access, kept off the Tk thread."""

import threading
from collections.abc import Callable
from typing import TypeVar

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
    def read():
        from jasna.protection import license_store, signing

        key = license_store.load_license()
        identity = None
        if key:
            try:
                proof, _ = signing.decode_license(key)
                identity = signing.license_identity(proof)
            except ValueError:
                identity = None
        return key, license_store.is_licensed(), identity

    return _start(read, on_done)


def start_machine_key(cloud_workspace: bool, on_done) -> threading.Thread:
    def read():
        from jasna.protection import machine

        return machine.machine_key(cloud_workspace=cloud_workspace)

    return _start(read, on_done)


def start_activation(key: str, on_done) -> threading.Thread:
    def activate():
        from jasna.protection import license_store, signing

        license_store.set_license(key)
        proof, _ = signing.decode_license(key)
        return signing.license_identity(proof)

    return _start(activate, on_done)


def start_online_activation(code: str, machine_key: str, on_done) -> threading.Thread:
    from jasna.gui.locales import get_locale

    prefer_hong_kong = get_locale().current_language == "zh"

    def activate():
        from jasna.protection import activation, signing

        key = activation.activate_license_code(code, machine_key, prefer_hong_kong=prefer_hong_kong)
        proof, _ = signing.decode_license(key)
        return key, signing.license_identity(proof)

    return _start(activate, on_done)
