from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from jasna.ltx.model_files import LtxModelFiles, open_tensors


def test_source_model_bundle_and_reader(tmp_path):
    value = torch.arange(6).reshape(2, 3)
    for name in ("transformer", "vae", "vae-decoder"):
        save_file({"weight": value}, str(tmp_path / f"{name}.safetensors"), metadata={"config": "example"})
    files = LtxModelFiles.from_dir(tmp_path, fast=False)
    with open_tensors(files.transformer) as handle:
        assert list(handle.keys()) == ["weight"]
        assert handle.metadata() == {"config": "example"}
        assert torch.equal(handle.get_tensor("weight"), value)


def test_model_check_precedes_engine_compilation(tmp_path, monkeypatch):
    from factories import session_config
    from jasna import accelerator, engine_compiler, session_factory

    compile_engines = Mock()
    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda device: True)
    monkeypatch.setattr(engine_compiler, "ensure_engines_compiled", compile_engines)
    monkeypatch.setattr(LtxModelFiles, "from_dir", Mock(side_effect=ValueError("model unavailable")))
    config = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path)
    with pytest.raises(ValueError, match="model unavailable"):
        session_factory._build_ltx_session(config, torch.device("cuda:0"), log_callback=None)
    compile_engines.assert_not_called()


def test_fast_bundle_uses_the_fast_transformer(tmp_path):
    for name in ("transformer-fast", "vae", "vae-decoder"):
        save_file({"weight": torch.zeros(1)}, str(tmp_path / f"{name}.safetensors"))
    assert LtxModelFiles.from_dir(tmp_path, fast=True).transformer == tmp_path / "transformer-fast.safetensors"


def test_bundle_present_needs_every_quality_file(tmp_path):
    from jasna.ltx.model_files import bundle_present

    for name in ("transformer-fast", "vae", "vae-decoder"):
        (tmp_path / f"{name}.safetensors").touch()
    assert not bundle_present(tmp_path)
    (tmp_path / "transformer.safetensors.enc").touch()
    assert bundle_present(tmp_path)
