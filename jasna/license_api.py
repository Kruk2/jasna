"""Safe access to the optional private license implementation.

Public source checkouts intentionally do not contain ``jasna.protection``.
Keep free-model workflows usable without pretending that supporter features
are licensed. Release builds that include the private package keep using its
real store and exception type unchanged.
"""

from __future__ import annotations


try:
    from jasna.protection import ProtectionError, license_store
except ImportError as exc:
    # Do not hide a missing dependency inside an installed protection package.
    # The public checkout is represented either by no package or by the empty
    # gitlink directory, both of which report ``jasna.protection`` here.
    if exc.name != "jasna.protection":
        raise

    class ProtectionError(RuntimeError):
        """Raised when supporter activation is unavailable or invalid."""


    class _PublicSourceLicenseStore:
        @staticmethod
        def load_license() -> None:
            return None

        @staticmethod
        def is_licensed() -> bool:
            return False

        @staticmethod
        def set_license(email: str, key: str) -> None:
            del email, key
            raise ProtectionError(
                "Supporter activation is unavailable in the public source checkout"
            )


    license_store = _PublicSourceLicenseStore()


__all__ = ["ProtectionError", "license_store"]
