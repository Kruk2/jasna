"""Runtime shims for the compiled (Nuitka) build.

torch does source introspection at import time — `torch.jit.interface` decorators and
`torch._inductor` config-comment parsing both call `inspect.getsource`. A compiled
binary has no .py source, so those raise. Make them non-fatal. No-op in source/dev,
where the original calls succeed (the wrappers only swallow the missing-source error).
The NVIDIA build preserves the bytecode and introspection data Dynamo needs.

torch.compile in the binary: Inductor compiles in-process (its worker pool would re-launch
the jasna binary) and keys its caches on the torch version (its source-tree hash cannot
walk the bundled torch). Triton finds its GPU backends by scanning its package (no pip
metadata is shipped), and loads the C helper modules the build ships in
``triton_prebuilt/`` instead of compiling them with a C compiler users may not have.
"""
from __future__ import annotations

import os
import sys

_patched = False

LTX_TRACED_MODULES = ("jasna.ltx.transformer", "jasna.ltx.kernels",
                      "jasna.models.ltx_vae.na_triton", "jasna.models.ltx_vae.transformer",
                      "jasna.models.ltx_vae.video_encoder", "jasna.models.ltx_vae.ops")


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
    if is_frozen():
        ignore_triton_cuda_toolkit_lookup_warning()

    import torch.jit
    _interface = torch.jit.interface

    def interface(obj):
        try:
            return _interface(obj)
        except OSError:
            return obj

    torch.jit.interface = interface

    if is_frozen():
        match_bundled_distribution_names_like_pip()
        from torch.utils import _config_module
        _config_module.get_assignments_with_compile_ignored_comments = lambda module: set()

        import hashlib
        import importlib.util
        from pathlib import Path

        os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
        import torch
        import torch._inductor.config as inductor_config
        from torch._inductor import codecache
        inductor_config.compile_threads = 1
        codecache.torch_key.set(hashlib.sha256(f"frozen torch {torch.__version__}".encode()).digest())
        if importlib.util.find_spec("triton") is not None:
            from jasna.os_utils import get_application_executable
            root = Path(get_application_executable()).parent
            register_ltx_sources(root)
            use_prebuilt_triton_modules(root / "triton_prebuilt")


class NormalizedDistributionNameFinder:
    """Nuitka finds bundled package metadata only by its exact, case-insensitive name, so
    transformers' ``importlib.metadata.version("huggingface-hub")`` misses the bundled
    ``huggingface_hub``. Match the normalized name, as installed packages do."""

    @staticmethod
    def find_spec(*_args):
        return None

    @staticmethod
    def find_distributions(context):
        if context.name is None:
            return iter(())
        from importlib.metadata import distributions

        wanted = _normalized_distribution_name(context.name)
        return iter([distribution for distribution in distributions()
                     if distribution.name.lower() != context.name.lower()
                     and _normalized_distribution_name(distribution.name) == wanted])


def _normalized_distribution_name(name: str) -> str:
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def match_bundled_distribution_names_like_pip() -> None:
    sys.meta_path.append(NormalizedDistributionNameFinder)


def ignore_triton_cuda_toolkit_lookup_warning() -> None:
    """Triton on Windows warns when no CUDA Toolkit is installed, but the build bundles ptxas and the helper."""
    import warnings
    warnings.filterwarnings("ignore", message="Failed to find CUDA", category=UserWarning,
                            module="triton.windows_utils")


def register_ltx_sources(root) -> None:
    """Expose source to inspect without a directory named like the executable."""
    import importlib.util
    import linecache
    from pathlib import Path

    for module in LTX_TRACED_MODULES:
        relative = Path(*module.split(".")).with_suffix(".py")
        source_path = str(root / "python_source" / relative)
        lines = linecache.getlines(source_path)
        filename = importlib.util.find_spec(module).origin
        linecache.cache[filename] = (sum(map(len, lines)), None, lines, filename)


def use_prebuilt_triton_modules(directory) -> None:
    import sysconfig
    from triton import knobs

    suffix = sysconfig.get_config_var("EXT_SUFFIX")

    def prebuilt(name, src, srcdir, library_dirs, include_dirs, libraries):
        module = directory / f"{name}{suffix}"
        if not module.is_file():
            raise FileNotFoundError(module)
        return str(module)

    knobs.build.impl = prebuilt
