import hashlib
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from jasna.ltx import model_files
from jasna.ltx.model_files import LtxModelFiles, bundle_names, open_tensors


def test_source_model_bundle_and_reader(tmp_path):
    value = torch.arange(6).reshape(2, 3)
    for name in bundle_names("distilled", fast=False):
        save_file({"weight": value}, str(tmp_path / name), metadata={"config": "example"})
    files = LtxModelFiles.from_dir(tmp_path, "distilled", fast=False)
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
        session_factory._ltx_model_files(config, torch.device("cuda:0"), log_callback=None)
    compile_engines.assert_not_called()


def test_every_model_choice_has_its_own_transformer_file():
    transformers = {bundle_names(model, fast=fast)[0] for model in model_files.LTX_MODELS for fast in (False, True)}
    assert len(transformers) == 4
    assert all(name in model_files.LTX_DOWNLOADS for name in transformers)
    assert bundle_names("undistilled", fast=True)[1:] == bundle_names("distilled", fast=False)[1:]


def test_installed_checks_the_chosen_model_only(tmp_path):
    for name in bundle_names("undistilled", fast=True):
        (tmp_path / name).touch()
    assert model_files.model_installed(tmp_path, "undistilled", fast=True)
    assert not model_files.model_installed(tmp_path, "distilled", fast=True)
    assert model_files.bundle_present(tmp_path)
    assert [f.name for f in model_files.missing_downloads(tmp_path, "distilled", fast=True)] == [
        "ltx-restore-alpha1-distill8-nvfp4.safetensors.enc"
    ]


def test_download_refuses_without_a_published_location(tmp_path, monkeypatch):
    monkeypatch.setattr(model_files, "LTX_DOWNLOAD_URL", None)
    with pytest.raises(RuntimeError, match="not published"):
        model_files.download_files(tmp_path, model_files.missing_downloads(tmp_path, "distilled", fast=False))


def test_download_verifies_and_replaces_atomically(tmp_path, monkeypatch):
    source = tmp_path / "published"
    source.mkdir()
    good, bad = b"model bytes", b"tampered"
    (source / "a.safetensors.enc").write_bytes(good)
    (source / "b.safetensors.enc").write_bytes(bad)
    monkeypatch.setattr(model_files, "LTX_DOWNLOAD_URL", source.as_uri())
    target = tmp_path / "models"
    progress = []
    ok = model_files.DownloadableFile("a.safetensors.enc", len(good), hashlib.sha256(good).hexdigest())
    model_files.download_files(target, [ok], lambda done, total: progress.append((done, total)))
    assert (target / "a.safetensors.enc").read_bytes() == good and progress[-1] == (len(good), len(good))
    wrong = model_files.DownloadableFile("b.safetensors.enc", len(bad), hashlib.sha256(good).hexdigest())
    with pytest.raises(RuntimeError, match="checksum"):
        model_files.download_files(target, [wrong])
    assert sorted(p.name for p in target.iterdir()) == ["a.safetensors.enc"]


def test_cli_offers_the_missing_model_only_when_ltx_runs(tmp_path, monkeypatch):
    import sys

    from factories import session_config
    from jasna import main

    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
    main._ensure_ltx_model(session_config(restoration_model_name="basicvsrpp"), None)
    assert asked == []

    ltx = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(FileNotFoundError, match="Run jasna in a terminal"):
        main._ensure_ltx_model(ltx, None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with pytest.raises(FileNotFoundError, match="missing"):
        main._ensure_ltx_model(ltx, None)
    assert "GB" in asked[0] and "ltx-restore-alpha1-distill8-int8" in asked[0]
