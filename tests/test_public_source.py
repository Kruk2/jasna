"""Exercise public source with no access to the owner's private submodule."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(params=["absent", "empty"])
def public_checkout(tmp_path, request):
    source = Path(__file__).parents[1] / "jasna"
    shutil.copytree(source, tmp_path / "jasna", ignore=shutil.ignore_patterns("protection", "__pycache__"))
    if request.param == "empty":
        (tmp_path / "jasna" / "protection").mkdir()
    return tmp_path


def run_public(checkout, script):
    environment = {**os.environ, "PYTHONPATH": str(checkout),
                   "XDG_CONFIG_HOME": str(checkout / "config"), "CUDA_VISIBLE_DEVICES": ""}
    environment.pop("JASNA_MAIN_PID", None)
    prefix = """
import sys
sys.meta_path[:] = [finder for finder in sys.meta_path
                   if not getattr(finder, '__module__', type(finder).__module__).startswith('__editable___jasna_')]
"""
    return subprocess.run([sys.executable, "-B", "-c", prefix + script], cwd=checkout,
                          env=environment, capture_output=True, text=True, timeout=30)


def test_public_startup_imports_and_license_status(public_checkout):
    result = run_public(public_checkout, """
from jasna.gui import components, activation_worker
from jasna import session_factory, engine_compiler, image_restore
from jasna.license_api import protection_available
assert not protection_available()
results = []
worker = activation_worker.start_license_info(lambda info, error: results.append((info, error)))
worker.join(5)
assert results == [((None, False, None), None)], results
assert 'jasna.protection.license_store' not in sys.modules
""")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_public_cli_help_and_version(public_checkout, argument):
    result = run_public(public_checkout, f"""
import runpy
sys.argv = ['jasna', {argument!r}]
runpy.run_module('jasna', run_name='__main__')
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.parametrize("arguments", [
    ["--machine-key"], ["--activation-key", "fake"], ["--license-code", "fake"],
])
def test_public_activation_reports_unavailable_without_traceback(public_checkout, arguments):
    result = run_public(public_checkout, f"""
import runpy
sys.argv = ['jasna', *{arguments!r}]
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


def test_public_encrypted_models_fail_before_opening_data(public_checkout):
    result = run_public(public_checkout, """
from pathlib import Path
from jasna.ltx.model_files import LtxModelFiles, open_tensors
from jasna.restorer.sd15_inpaint_restorer import _read_checkpoint
from jasna.gui.ltx_models import license_missing, read_install_state
operations = (lambda: open_tensors(Path('missing.enc')),
              lambda: LtxModelFiles.from_dir(Path('missing'), 'distilled', fast=False),
              lambda: _read_checkpoint(Path('missing')))
""" + _ASSERT_EVERY_OPERATION_NEEDS_PROTECTION + """
assert license_missing(Path('missing'), 'distilled', False)
assert not read_install_state(Path('missing')).downloadable
""")
    assert result.returncode == 0, result.stderr


def test_public_unet4x_fails_before_opening_data(public_checkout):
    pytest.importorskip("tensorrt")
    result = run_public(public_checkout, """
from jasna.restorer.unet4x_secondary_restorer import compile_unet4x_engine, Unet4xSecondaryRestorer
operations = (lambda: compile_unet4x_engine('cpu'), lambda: Unet4xSecondaryRestorer(device='cpu'))
""" + _ASSERT_EVERY_OPERATION_NEEDS_PROTECTION)
    assert result.returncode == 0, result.stderr


def test_public_activation_dialog_stays_disabled(public_checkout):
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        pytest.skip("GUI check needs a display")
    result = run_public(public_checkout, """
import customtkinter as ctk
from jasna.gui.components import LicenseDialog
from jasna.gui.locales import t
root = ctk.CTk()
root.update()
dialog = LicenseDialog(root, on_activated=lambda: None)
root.update()
assert dialog._status.cget('text') == t('license_source_unavailable')
for widget in (dialog._activate_button, dialog._online_activate_button, dialog._activation_page_button):
    assert widget.cget('state') == 'disabled'
dialog.destroy()
root.destroy()
""")
    assert result.returncode == 0, result.stderr


def test_frozen_missing_protection_cannot_use_public_fallback(public_checkout):
    result = run_public(public_checkout, """
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


def test_installed_broken_protection_keeps_its_error(public_checkout):
    protection = public_checkout / "jasna" / "protection"
    protection.mkdir(exist_ok=True)
    (protection / "__init__.py").write_text("raise RuntimeError('provider initialization failed')\n")
    result = run_public(public_checkout, """
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
