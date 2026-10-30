"""Keep public source usable without the optional private protection package."""

from importlib.util import find_spec

from jasna._frozen import is_frozen


class ProtectionUnavailableError(RuntimeError):
    pass


def protection_available() -> bool:
    if is_frozen():
        return True
    spec = find_spec("jasna.protection")
    # An uninitialized git submodule is an empty namespace, not an implementation.
    return spec is not None and spec.origin is not None


def require_protection() -> None:
    if not protection_available():
        raise ProtectionUnavailableError("Supporter models and activation require the official Jasna release.")


class LicenseCheckStopped(Exception):
    """A license check that every later paid-model job would also fail; ``code`` names the reason
    (expired, offline, devices_full, token_not_found, already_converted, unofficial_key, no_activation, offline_check)."""

    code: str
