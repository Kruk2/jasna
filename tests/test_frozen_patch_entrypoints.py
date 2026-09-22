import importlib
import logging
import sys
from unittest.mock import patch

import pytest

from test_main import _base_argv, _main_patches, _make_model_files


def test_frozen_config_scan_skips_source(caplog, monkeypatch):
    import torch.jit
    from torch.utils import _config_module
    from jasna import _frozen

    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: True)
    with (
        patch.object(torch.jit, "interface"),
        patch.object(
            _config_module,
            "get_assignments_with_compile_ignored_comments",
            side_effect=KeyError("torch._inductor.config_comms"),
        ) as scan,
        caplog.at_level(logging.DEBUG, logger="jasna._frozen"),
    ):
        _frozen.patch_frozen_torch()
        assert _config_module.get_assignments_with_compile_ignored_comments(object()) == set()

    scan.assert_not_called()
    assert not caplog.records


def test_importing_pipeline_does_not_patch_frozen_torch():
    orig = sys.modules.pop("jasna.pipeline", None)
    try:
        with patch("jasna._frozen.patch_frozen_torch") as spy:
            importlib.import_module("jasna.pipeline")
        spy.assert_not_called()
    finally:
        if orig is not None:
            sys.modules["jasna.pipeline"] = orig
        else:
            sys.modules.pop("jasna.pipeline", None)


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
