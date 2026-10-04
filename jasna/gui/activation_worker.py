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


def read_license_info() -> tuple[str | None, bool, str | None]:
    if not protection_available():
        return None, False, None
    from jasna.protection import license_store, signing

    key = license_store.load_license()
    identity = None
    if key:
        try:
            proof, _ = signing.decode_license(key)
            identity = signing.license_identity(proof)
        except ValueError:
            identity = None
    licensed = not (identity is not None and identity.startswith("JW1-")) and license_store.is_licensed()
    return key, licensed, identity


def start_license_info(on_done) -> threading.Thread:
    return _start(read_license_info, on_done)


def start_machine_key(on_done) -> threading.Thread:
    def read():
        require_protection()
        from jasna.protection import machine

        return machine.machine_key(cloud_workspace=False)

    return _start(read, on_done)


def start_activation(key: str, on_done) -> threading.Thread:
    def activate():
        require_protection()
        from jasna.protection import license_store, signing
        from jasna.gui.locales import t

        proof, _ = license_store._decode(key)
        identity = signing.license_identity(proof)
        if identity.startswith("JW1-"):
            raise ValueError(t("license_cloud_cli_only"))
        license_store.set_license(key)
        return identity

    return _start(activate, on_done)


def start_online_activation(code: str, machine_key: str, on_done) -> threading.Thread:
    from jasna.gui.locales import get_locale

    prefer_hong_kong = get_locale().current_language == "zh"

    def activate():
        require_protection()
        from jasna.protection import activation, signing
        from jasna.gui.locales import t

        if machine_key.startswith("JW1-"):
            raise ValueError(t("license_cloud_cli_only"))
        key = activation.activate_license_code(code, machine_key, prefer_hong_kong=prefer_hong_kong)
        proof, _ = signing.decode_license(key)
        return key, signing.license_identity(proof)

    return _start(activate, on_done)
