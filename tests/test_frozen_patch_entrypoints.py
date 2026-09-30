import importlib
import logging
import os
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
    monkeypatch.setattr(_frozen, "use_prebuilt_triton_modules", lambda directory: None)
    monkeypatch.setenv("TRITON_BACKENDS_IN_TREE", "1")
    import torch._inductor.config as inductor_config
    from torch._inductor import codecache
    monkeypatch.setattr(inductor_config, "compile_threads", inductor_config.compile_threads)
    monkeypatch.setattr(codecache, "torch_key", codecache.torch_key_cache(lambda: b"source"))
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


def test_source_build_keeps_config_scan(monkeypatch):
    import torch.jit
    from torch.utils import _config_module
    from jasna import _frozen

    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: False)
    scan = _config_module.get_assignments_with_compile_ignored_comments
    with patch.object(torch.jit, "interface"):
        _frozen.patch_frozen_torch()
    assert _config_module.get_assignments_with_compile_ignored_comments is scan


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


def test_frozen_build_compiles_in_process_with_prebuilt_triton_modules(monkeypatch, tmp_path):
    import torch._inductor.config as inductor_config
    import torch.jit
    from jasna import _frozen

    triton = pytest.importorskip("triton")
    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "jasna"))
    monkeypatch.delenv("TRITON_BACKENDS_IN_TREE", raising=False)
    monkeypatch.setattr(inductor_config, "compile_threads", inductor_config.compile_threads)
    monkeypatch.setattr(triton.knobs.build, "impl", None)
    from torch._inductor import codecache
    monkeypatch.setattr(codecache, "torch_key", codecache.torch_key_cache(lambda: b"source"))
    with patch.object(torch.jit, "interface"):
        _frozen.patch_frozen_torch()

    assert inductor_config.compile_threads == 1
    assert codecache.torch_key() != b"source"
    assert os.environ["TRITON_BACKENDS_IN_TREE"] == "1"
    module = triton.runtime.build._build("cuda_utils", "src.c", str(tmp_path), [], [], [], [])
    assert module.startswith(str(tmp_path / "triton_prebuilt" / "cuda_utils"))
