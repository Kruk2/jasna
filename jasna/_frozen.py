"""Runtime shims for the compiled (Nuitka) build.

torch does source introspection at import time — `torch.jit.interface` decorators and
`torch._inductor` config-comment parsing both call `inspect.getsource`. A compiled
binary has no .py source, so those raise. Make them non-fatal. No-op in source/dev,
where the original calls succeed (the wrappers only swallow the missing-source error).

torch.compile in the binary: Inductor compiles in-process (its worker pool would re-launch
the jasna binary) and keys its caches on the torch version (its source-tree hash cannot
walk the bundled torch). Triton finds its GPU backends by scanning its package (no pip
metadata is shipped), and on Linux loads the C helper modules the build ships in
``triton_prebuilt/`` instead of compiling them with gcc, which users may not have.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

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

    if is_frozen():
        from torch.utils import _config_module
        _config_module.get_assignments_with_compile_ignored_comments = lambda module: set()

        os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
        import torch
        import torch._inductor.config as inductor_config
        from torch._inductor import codecache
        inductor_config.compile_threads = 1
        codecache.torch_key.set(hashlib.sha256(f"frozen torch {torch.__version__}".encode()).digest())
        if sys.platform == "linux" and importlib.util.find_spec("triton") is not None:
            use_prebuilt_triton_modules(Path(sys.executable).parent / "triton_prebuilt")


def use_prebuilt_triton_modules(directory: Path) -> None:
    """Make Triton load its C helper modules (``cuda_utils``) from ``directory`` instead of
    building them with a C compiler."""
    from triton import knobs

    suffix = sysconfig.get_config_var("EXT_SUFFIX")

    def prebuilt(name, src, srcdir, library_dirs, include_dirs, libraries):
        return str(directory / f"{name}{suffix}")

    knobs.build.impl = prebuilt
