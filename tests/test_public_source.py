"""Exercise public source with no access to the owner's private submodule."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _copy_public_source(root: Path) -> Path:
    shutil.copytree(
        Path(__file__).parents[1] / "jasna", root / "jasna", ignore=shutil.ignore_patterns("protection", "__pycache__")
    )
    return root


@pytest.fixture(scope="module", params=["absent", "empty"])
def public_checkout(tmp_path_factory, request):
    """The public source without the private protection package (absent, or an empty folder).
    One copy serves every test of this module: the tests only read it and keep their settings in their own ``tmp_path``."""
    root = _copy_public_source(tmp_path_factory.mktemp(f"public-{request.param}"))
    if request.param == "empty":
        (root / "jasna" / "protection").mkdir()
    return root


def run_public(checkout, config_dir, script):
    environment = {**os.environ, "PYTHONPATH": str(checkout), "APPDATA": str(config_dir),
                   "XDG_CONFIG_HOME": str(config_dir), "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    environment.pop("JASNA_MAIN_PID", None)
    prefix = """
import sys
sys.meta_path[:] = [finder for finder in sys.meta_path
                   if not getattr(finder, '__module__', type(finder).__module__).startswith('__editable___jasna_')]
"""
    return subprocess.run([sys.executable, "-B", "-c", prefix + script], cwd=checkout,
                          env=environment, capture_output=True, encoding="utf-8", timeout=30)


def test_public_startup_imports_and_license_status(public_checkout, tmp_path):
    result = run_public(public_checkout, tmp_path, """
from jasna.gui import components, license_dialog, activation_worker
from jasna import license_status, session_factory, engine_compiler, image_restore
from jasna.license_api import protection_available
assert not protection_available()
results = []
worker = activation_worker.start_license_info(lambda info, error: results.append((info, error)))
worker.join(5)
assert results == [((False, None), None)], results
license_status.refresh_pc_license()
from jasna import update_check
assert update_check.fetch_latest_release() is None
assert 'jasna.protection.license_store' not in sys.modules
assert 'jasna.protection.lease' not in sys.modules
""")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_public_cli_help_and_version(public_checkout, argument, tmp_path):
    result = run_public(public_checkout, tmp_path, f"""
import runpy
sys.argv = ['jasna', {argument!r}]
runpy.run_module('jasna', run_name='__main__')
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_public_activation_reports_unavailable_without_traceback(public_checkout, tmp_path):
    result = run_public(public_checkout, tmp_path, """
import runpy
sys.argv = ['jasna', '--license-code', 'fake']
runpy.run_module('jasna', run_name='__main__')
""")
    assert result.returncode == 1
    assert "official Jasna release" in result.stderr
    assert "Traceback" not in result.stderr


_ASSERT_EVERY_OPERATION_NEEDS_PROTECTION = """
from jasna.license_api import ProtectionUnavailableError
for operation in operations:
    try:
        operation()
    except ProtectionUnavailableError:
        pass
    else:
        raise AssertionError('Paid model was accessible without protection')
"""


def test_public_encrypted_models_fail_before_opening_data(public_checkout, tmp_path):
    result = run_public(public_checkout, tmp_path, """
from pathlib import Path
from jasna.ltx.model_files import LtxModelFiles, open_tensors
from jasna.restorer.sd15_inpaint_restorer import _read_checkpoint
from jasna.gui.ltx_models import license_missing, read_install_state
operations = (lambda: open_tensors(Path('missing.enc')),
              lambda: LtxModelFiles.from_dir(Path('missing'), 'distilled', fast=False),
              lambda: _read_checkpoint(Path('missing')))
""" + _ASSERT_EVERY_OPERATION_NEEDS_PROTECTION + """
assert license_missing(Path('missing'), 'distilled', False, 'auto', 'auto')
assert not read_install_state(Path('missing'), 'auto', 'auto').downloadable
""")
    assert result.returncode == 0, result.stderr


def test_public_unet4x_fails_before_opening_data(public_checkout, tmp_path):
    pytest.importorskip("tensorrt")
    result = run_public(public_checkout, tmp_path, """
from jasna.restorer.unet4x_secondary_restorer import compile_unet4x_engine, Unet4xSecondaryRestorer
operations = (lambda: compile_unet4x_engine('cpu'), lambda: Unet4xSecondaryRestorer(device='cpu'))
""" + _ASSERT_EVERY_OPERATION_NEEDS_PROTECTION)
    assert result.returncode == 0, result.stderr


def test_public_activation_dialog_stays_disabled(public_checkout, tmp_path):
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        pytest.skip("GUI check needs a display")
    result = run_public(public_checkout, tmp_path, """
import customtkinter as ctk
from jasna.gui.license_dialog import LicenseDialog
from jasna.gui.locales import t
root = ctk.CTk()
root.update()
dialog = LicenseDialog(root, on_activated=lambda: None)
root.update()
assert dialog._status.cget('text') == t('license_source_unavailable')
assert dialog._activate_button.cget('state') == 'disabled'
dialog.destroy()
root.destroy()
""")
    assert result.returncode == 0, result.stderr


def test_frozen_missing_protection_cannot_use_public_fallback(public_checkout, tmp_path):
    result = run_public(public_checkout, tmp_path, """
sys.frozen = True
from jasna.license_api import protection_available
from jasna.gui.activation_worker import start_license_info
assert protection_available()
results = []
worker = start_license_info(lambda info, error: results.append((info, error)))
worker.join(5)
info, error = results[0]
assert info is None and isinstance(error, ImportError), results
""")
    assert result.returncode == 0, result.stderr


def test_installed_broken_protection_keeps_its_error(tmp_path):
    checkout = _copy_public_source(tmp_path / "checkout")
    protection = checkout / "jasna" / "protection"
    protection.mkdir()
    (protection / "__init__.py").write_text("raise RuntimeError('provider initialization failed')\n")
    result = run_public(checkout, tmp_path / "config", """
from jasna.license_api import protection_available
from jasna.gui.activation_worker import start_license_info
assert protection_available()
results = []
worker = start_license_info(lambda info, error: results.append((info, error)))
worker.join(5)
info, error = results[0]
assert info is None and type(error) is RuntimeError, results
assert str(error) == 'provider initialization failed'
""")
    assert result.returncode == 0, result.stderr
