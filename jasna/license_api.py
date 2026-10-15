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
