import importlib
import sys
from unittest.mock import patch

import pytest

from test_main import _base_argv, _main_patches, _make_model_files


def test_importing_pipeline_does_not_patch_frozen_torch():
    with patch("jasna._frozen.patch_frozen_torch") as spy:
        sys.modules.pop("jasna.pipeline", None)
        importlib.import_module("jasna.pipeline")
    spy.assert_not_called()


def test_cli_main_patches_frozen_torch(tmp_path):
    inp, out, rest, det = _make_model_files(tmp_path)
    with patch("jasna._frozen.patch_frozen_torch") as spy:
        with _main_patches():
            with patch.object(sys, "argv", _base_argv(inp, out, rest, det)):
                from jasna.main import main
                main()
    spy.assert_called()


def test_gui_run_gui_patches_frozen_torch():
    from jasna.gui import app as gui_app

    class _Stop(Exception):
        pass

    with patch("jasna._frozen.patch_frozen_torch") as spy:
        with patch.object(gui_app, "JasnaApp", side_effect=_Stop):
            with pytest.raises(_Stop):
                gui_app.run_gui()
    spy.assert_called()
