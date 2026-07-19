"""Runtime shims for the compiled (Nuitka) build.

torch does source introspection at import time — `torch.jit.interface` decorators and
`torch._inductor` config-comment parsing both call `inspect.getsource`. A compiled
binary has no .py source, so those raise. Make them non-fatal. No-op in source/dev,
where the original calls succeed (the wrappers only swallow the missing-source error).
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_patched = False


def is_frozen() -> bool:
    """True when running from a frozen/compiled build.

    PyInstaller sets ``sys.frozen``; Nuitka does not — it injects a ``__compiled__``
    global into every compiled module instead. Check both so frozen-only code paths
    fire under either packaging tool.
    """
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def patch_frozen_torch() -> None:
    global _patched
    if _patched:
        return
    _patched = True

    import torch.jit
    _interface = torch.jit.interface

    def interface(obj):
        try:
            return _interface(obj)
        except OSError:
            return obj

    torch.jit.interface = interface

    from torch.utils import _config_module
    _assignments = _config_module.get_assignments_with_compile_ignored_comments

    def get_assignments_with_compile_ignored_comments(module):
        try:
            return _assignments(module)
        except Exception:
            logger.debug("Compile-ignored-comment scan failed (no source); returning empty", exc_info=True)
            return set()

    _config_module.get_assignments_with_compile_ignored_comments = get_assignments_with_compile_ignored_comments
