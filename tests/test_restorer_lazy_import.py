"""Reaching lightweight restorer helpers must not drag in the torch stack.

The GUI builds its Image Restoration section on startup and calls
``sd15_download.bundle_present`` (a pure filesystem check). If importing that helper
also imported ``jasna.restorer``'s torch-backed restorers, the window would block for
several seconds on every launch. These tests pin the import to stay light.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

#: The portable ROCm distribution injects a ``sitecustomize``/``.pth`` shim that
#: imports torch at interpreter startup, so even a bare ``import jasna.restorer``
#: child already has torch in ``sys.modules`` before any Jasna code runs. That is
#: an environment fact, not a regression in the lazy exports.
_PRELOADS_TORCH = importlib.util.find_spec("_jasna_rocm_dist_shim") is not None

requires_no_preloaded_torch = pytest.mark.skipif(
    _PRELOADS_TORCH,
    reason="interpreter startup (ROCm dist shim) preloads torch; the child cannot be torch-free",
)


def _import_keeps_torch_out(import_stmt: str) -> subprocess.CompletedProcess:
    code = (
        "import sys\n"
        f"{import_stmt}\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@requires_no_preloaded_torch
def test_importing_restorer_package_does_not_load_torch():
    result = _import_keeps_torch_out("import jasna.restorer")
    assert result.returncode == 0, result.stderr


@requires_no_preloaded_torch
def test_bundle_present_import_does_not_load_torch():
    result = _import_keeps_torch_out("from jasna.restorer.sd15_download import bundle_present")
    assert result.returncode == 0, result.stderr


def test_lazy_exports_still_resolve():
    code = (
        "from jasna.restorer import RestorationPipeline, BasicvsrppMosaicRestorer, "
        "DenoiseStep, DenoiseStrength\n"
        "assert all(x is not None for x in "
        "(RestorationPipeline, BasicvsrppMosaicRestorer, DenoiseStep, DenoiseStrength))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
