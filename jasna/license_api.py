"""Keep public source usable without the optional private protection package."""

import sys
import unicodedata
from dataclasses import dataclass
from importlib.util import find_spec

from jasna._frozen import is_frozen


class ProtectionUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DriverLibraryMismatch:
    """An NVIDIA library LTX would load that is from another driver version than the installed one."""

    path: str
    library_version: str
    driver_version: str


def ltx_driver_mismatch() -> DriverLibraryMismatch | None:
    """What stops LTX from starting on this Linux machine, found before any model download."""
    if not (is_frozen() and sys.platform == "linux"):
        return None
    from jasna.protection.ltx_worker_security import linux_driver_library_mismatch

    return linux_driver_library_mismatch()


def protection_available() -> bool:
    if is_frozen():
        return True
    spec = find_spec("jasna.protection")
    # An uninitialized git submodule is an empty namespace, not an implementation.
    return spec is not None and spec.origin is not None


def require_protection() -> None:
    if not protection_available():
        raise ProtectionUnavailableError("Supporter models and activation require the official Jasna release.")


def typed_license_code(text: str) -> str:
    """The License code as pasted, without the full-width characters, spaces, line breaks and invisible characters
    that input methods and chat apps add."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(c for c in normalized if not c.isspace() and unicodedata.category(c) != "Cf")


class LicenseCheckStopped(Exception):
    """A license check that every later paid-model job would also fail; ``code`` names the reason
    (expired, offline, devices_full, token_not_found, already_converted, unofficial_key, no_activation, offline_check,
    pc_refresh, pc_changed, pc_unreadable)."""

    code: str
