import hashlib
import io
import runpy
import ssl
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import certifi
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


def test_model_check_precedes_engine_compilation(tmp_path, monkeypatch, nvidia_vendor):
    from factories import session_config
    from jasna import engine_compiler, session_factory

    compile_engines = Mock()
    monkeypatch.setattr("jasna.ltx.transformer.require_kernels", lambda device: None)
    monkeypatch.setattr(engine_compiler, "ensure_engines_compiled", compile_engines)
    monkeypatch.setattr(LtxModelFiles, "from_dir", Mock(side_effect=ValueError("model unavailable")))
    config = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path)
    with pytest.raises(ValueError, match="model unavailable"):
        session_factory._ltx_model_files(config, torch.device("cuda:0"), log_callback=None)
    compile_engines.assert_not_called()


def test_every_model_choice_has_its_own_transformer_file():
    transformers = {bundle_names(model, fast=fast)[0] for model in model_files.LTX_MODELS for fast in (False, True)}
    assert len(transformers) == 8
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


@pytest.mark.parametrize("old_size", [64, 96])
def test_outdated_encrypted_model_is_missing_until_verified_download_replaces_it(tmp_path, monkeypatch, old_size):
    from jasna.protection import protected_tensors

    names = bundle_names("distilled2", fast=False)
    current = b"m" * 80
    entries = {name: replace(model_files.LTX_DOWNLOADS[name], size_bytes=len(current),
                            sha256=hashlib.sha256(current).hexdigest()) for name in names}
    monkeypatch.setattr(model_files, "LTX_DOWNLOADS", {**model_files.LTX_DOWNLOADS, **entries})
    monkeypatch.setattr(model_files, "is_frozen", lambda: True)
    monkeypatch.setattr("jasna.license_api.require_protection", lambda: None)
    checked = Mock()
    monkeypatch.setattr(protected_tensors, "check_ltx_license", checked)
    for name in names:
        (tmp_path / (name + ".enc")).write_bytes(current)
    transformer = tmp_path / (names[0] + ".enc")
    transformer.write_bytes(b"o" * old_size)
    with monkeypatch.context() as stat_only:
        stat_only.setattr(Path, "open", lambda *a, **k: pytest.fail("availability read model bytes"))
        assert not model_files.model_installed(tmp_path, "distilled2", fast=False)
        needed = model_files.missing_downloads(tmp_path, "distilled2", fast=False)
        assert needed == [replace(entries[names[0]], name=transformer.name)]
        with pytest.raises(FileNotFoundError, match="missing or outdated"):
            LtxModelFiles.from_dir(tmp_path, "distilled2", fast=False)
    assert checked.call_count == 1
    monkeypatch.setattr(model_files.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(current))
    model_files.download_files(tmp_path, needed, None, mirror_first=False)
    assert transformer.read_bytes() == current
    assert model_files.model_installed(tmp_path, "distilled2", fast=False)
    assert model_files.missing_downloads(tmp_path, "distilled2", fast=False) == []
    assert LtxModelFiles.from_dir(tmp_path, "distilled2", fast=False).transformer == transformer


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
    model_files.download_files(target, [ok], lambda done, total: progress.append((done, total)), mirror_first=False)
    assert (target / "a.safetensors.enc").read_bytes() == good and progress[-1] == (len(good), len(good))
    wrong = model_files.DownloadableFile("b.safetensors.enc", len(bad), hashlib.sha256(good).hexdigest())
    with pytest.raises(RuntimeError, match="checksum"):
        model_files.download_files(target, [wrong], None, mirror_first=False)
    assert sorted(p.name for p in target.iterdir()) == ["a.safetensors.enc"]


def _published(directory, content: bytes) -> str:
    directory.mkdir()
    (directory / "model.enc").write_bytes(content)
    return directory.as_uri()


def test_download_falls_back_to_the_other_source(tmp_path, monkeypatch):
    content = b"model bytes"
    file = model_files.DownloadableFile("model.enc", len(content), hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(model_files, "LTX_DOWNLOAD_URL", (tmp_path / "unreachable").as_uri())
    monkeypatch.setattr(model_files, "LTX_MIRROR_URL", _published(tmp_path / "mirror", content))
    model_files.download_files(tmp_path / "a", [file], None, mirror_first=False)
    assert (tmp_path / "a" / "model.enc").read_bytes() == content

    monkeypatch.setattr(model_files, "LTX_DOWNLOAD_URL", _published(tmp_path / "main", content))
    monkeypatch.setattr(model_files, "LTX_MIRROR_URL", (tmp_path / "unreachable").as_uri())
    model_files.download_files(tmp_path / "b", [file], None, mirror_first=True)
    assert (tmp_path / "b" / "model.enc").read_bytes() == content


@pytest.mark.parametrize("version,expected", [
    ("0.10.0", "main"), ("0.11.3", "main"), ("0.11.4", "main"),
    ("0.12.0", "0.12.0"), ("0.12.1", "0.12.0"), ("0.13.0", "0.12.0"),
])
def test_model_branch_uses_latest_known_version_not_newer_than_jasna(version, expected):
    assert model_files._model_branch(version) == expected


def test_model_branch_orders_versions_numerically_and_ignores_future_models(monkeypatch):
    monkeypatch.setattr(model_files, "_MODEL_BRANCHES", ("0.12.10", "0.13.0", "0.12.0", "0.12.2"))
    assert model_files._model_branch("0.12.1") == "0.12.0"
    assert model_files._model_branch("0.12.9") == "0.12.2"
    assert model_files._model_branch("0.12.10") == "0.12.10"
    assert model_files._model_branch("0.13.0") == "0.13.0"


@pytest.mark.parametrize("mirror_first", [False, True])
def test_replacement_models_use_release_branch_on_both_sources(tmp_path, monkeypatch, mirror_first):
    content = b"replacement model"
    name = "ltx-restore-alpha1-distill2-int8.safetensors.enc"
    file = model_files.DownloadableFile(name, len(content), hashlib.sha256(content).hexdigest())
    requests = []

    def open_url(url, **kwargs):
        requests.append(url)
        if len(requests) == 1:
            raise urllib.error.URLError("primary unreachable")
        return io.BytesIO(content)

    monkeypatch.setattr(model_files.urllib.request, "urlopen", open_url)
    model_files.download_files(tmp_path, [file], None, mirror_first=mirror_first)
    expected = [f"https://{host}/Kruk2/ltx-restore/resolve/0.12.0/{name}"
                for host in ("huggingface.co", "hf-mirror.com")]
    assert requests == (list(reversed(expected)) if mirror_first else expected)
    assert (tmp_path / name).read_bytes() == content


@pytest.mark.parametrize("mirror_first", [False, True])
def test_download_uses_bundled_ca_for_primary_and_fallback_with_empty_system_trust(tmp_path, monkeypatch, mirror_first):
    real_create_context = ssl.create_default_context
    monkeypatch.setattr(ssl.SSLContext, "load_default_certs", lambda *args, **kwargs: None)
    empty_default_context = real_create_context()
    assert empty_default_context.cert_store_stats()["x509_ca"] == 0
    create_context = Mock(wraps=real_create_context)
    monkeypatch.setattr(ssl, "create_default_context", create_context)
    requests = []
    content = b"model bytes"
    files = [model_files.DownloadableFile(name, len(content), hashlib.sha256(content).hexdigest())
             for name in ("first.enc", "second.enc")]

    def open_url(url, *, timeout, context=None):
        requests.append((url, timeout, context or empty_default_context))
        if len(requests) == 1:
            raise urllib.error.URLError("primary unavailable")
        return io.BytesIO(content)

    monkeypatch.setattr(model_files.urllib.request, "urlopen", open_url)
    target = tmp_path / "models"
    model_files.download_files(target, files, None, mirror_first=mirror_first)
    primary, fallback = ((model_files.LTX_MIRROR_URL, model_files.LTX_DOWNLOAD_URL) if mirror_first
                         else (model_files.LTX_DOWNLOAD_URL, model_files.LTX_MIRROR_URL))
    assert [url for url, _, _ in requests] == [f"{primary}/first.enc", f"{fallback}/first.enc", f"{primary}/second.enc"]
    assert all(timeout == 30 for _, timeout, _ in requests)
    for _, _, context in requests:
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        assert context.cert_store_stats()["x509_ca"] > 0
    assert all(context is requests[0][2] for _, _, context in requests)
    create_context.assert_called_once_with(cafile=certifi.where())
    assert all((target / file.name).read_bytes() == content for file in files)
    assert not list(target.glob("*.part"))


def test_mirror_first_tries_the_mirror_before_the_main_source(tmp_path, monkeypatch):
    mirrored = b"mirror bytes"
    monkeypatch.setattr(model_files, "LTX_DOWNLOAD_URL", _published(tmp_path / "main", b"main bytes"))
    monkeypatch.setattr(model_files, "LTX_MIRROR_URL", _published(tmp_path / "mirror", mirrored))
    file = model_files.DownloadableFile("model.enc", len(mirrored), hashlib.sha256(mirrored).hexdigest())
    model_files.download_files(tmp_path / "models", [file], None, mirror_first=True)
    assert (tmp_path / "models" / "model.enc").read_bytes() == mirrored
    with pytest.raises(RuntimeError, match="checksum"):
        model_files.download_files(tmp_path / "models", [file], None, mirror_first=False)


def test_cli_offers_the_missing_model_only_when_ltx_runs(tmp_path, monkeypatch, nvidia_vendor):
    import sys

    from factories import session_config
    from jasna import main

    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
    monkeypatch.setattr("jasna.license_api.require_protection", lambda: None)
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


def test_import_does_not_load_ca_or_start_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("module import performed download work")

    monkeypatch.setattr(ssl, "create_default_context", forbidden)
    monkeypatch.setattr(certifi, "where", forbidden)
    monkeypatch.setattr(model_files.urllib.request, "urlopen", forbidden)
    runpy.run_path(model_files.__file__)


def test_cli_offers_only_the_fast_models_files(tmp_path, monkeypatch, nvidia_vendor):
    import sys

    from factories import session_config
    from jasna import main

    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("jasna.license_api.require_protection", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    fast = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path, ltx_fast=True)

    with pytest.raises(FileNotFoundError, match="missing"):
        main._ensure_ltx_model(fast, None)

    assert "ltx-restore-alpha1-distill8-nvfp4" in asked[0] and "int8" not in asked[0]
    assert "9.2 GB" in asked[0]


@pytest.mark.parametrize("interactive", [False, True])
def test_cli_explains_models_copied_from_older_release_need_updating(tmp_path, monkeypatch, nvidia_vendor, interactive):
    import sys

    from factories import session_config
    from jasna import main

    monkeypatch.setattr(model_files, "is_frozen", lambda: True)
    monkeypatch.setattr("jasna.license_api.require_protection", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: interactive)
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
    names = bundle_names("distilled2", fast=False)
    for index, name in enumerate(names):
        with (tmp_path / (name + ".enc")).open("wb") as stream:
            stream.truncate(13_368_048_749 if index == 0 else model_files.LTX_DOWNLOADS[name].size_bytes)
    config = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path, ltx_model="distilled2")

    with pytest.raises(FileNotFoundError) as error:
        main._ensure_ltx_model(config, None)
    message = asked[0] if interactive else str(error.value)
    assert "need updating" in message and "older Jasna" in message
    assert "License code still works" in message
    assert names[0] in message and names[1] not in message and names[2] not in message
    assert (tmp_path / (names[0] + ".enc")).stat().st_size == 13_368_048_749
    if not interactive:
        assert asked == [] and "Run jasna in a terminal" in message


@pytest.mark.parametrize("nvidia", [False, True], ids=["amd", "older-nvidia"])
@pytest.mark.parametrize("trial", [False, True])
def test_cli_fast_ltx_checks_hardware_before_models_or_downloads(tmp_path, monkeypatch, nvidia, trial):
    from factories import session_config
    from jasna import accelerator, main

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda device: nvidia)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda device: not nvidia)
    def capability(device):
        assert nvidia, "AMD capability aliases cannot establish NVIDIA FP4 support"
        return (8, 9)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(model_files, "missing_downloads", lambda *args, **kwargs: pytest.fail("checked/downloaded incompatible files"))
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("offered an incompatible download"))
    config = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path,
                            ltx_fast=True, ltx_trial=trial)
    with pytest.raises(SystemExit, match="Blackwell"):
        main._ensure_ltx_model(config, None)


@pytest.mark.parametrize(("trial", "source_files", "refused"), [(False, False, True), (True, False, False), (False, True, False)])
@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_licensed_ltx_on_amd_requires_a_supported_worker(tmp_path, monkeypatch, trial, source_files, refused, platform):
    from factories import session_config
    from jasna import accelerator, session_factory

    monkeypatch.setattr(accelerator, "is_amd_device", lambda device=None: True)
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    if source_files:
        for name in bundle_names("distilled", fast=False):
            (tmp_path / name).touch()
    config = session_config(
        restoration_model_name="ltx", restoration_model_path=tmp_path, ltx_model="distilled", ltx_trial=trial
    )
    assert session_factory.unsupported_licensed_ltx_on_amd(config) is (refused and platform == "darwin")
    monkeypatch.setattr(accelerator, "is_amd_device", lambda device=None: False)
    assert not session_factory.unsupported_licensed_ltx_on_amd(config)


def test_licensed_ltx_on_amd_stops_before_the_download_offer_and_the_session(tmp_path, monkeypatch):
    from factories import session_config
    from jasna import accelerator, main, session_factory

    monkeypatch.setattr(accelerator, "is_amd_device", lambda device=None: True)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("offered a download"))
    monkeypatch.setattr("jasna.ltx.transformer.require_kernels", lambda device: pytest.fail("loaded LTX"))
    monkeypatch.setattr(accelerator.sys, "platform", "darwin")
    config = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path)

    with pytest.raises(SystemExit, match="LTX quality models are unavailable for this AMD setup"):
        main._ensure_ltx_model(config, None)
    with pytest.raises(ValueError, match="LTX quality models are unavailable for this AMD setup"):
        session_factory._ltx_model_files(config, torch.device("cuda:0"), log_callback=None)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_supported_licensed_amd_reaches_the_normal_model_preflight(tmp_path, monkeypatch, platform):
    from factories import session_config
    from jasna import accelerator, main, session_factory

    monkeypatch.setattr(accelerator, "is_amd_device", lambda device=None: True)
    monkeypatch.setattr(model_files, "missing_downloads", Mock(side_effect=RuntimeError("normal download preflight")))
    monkeypatch.setattr("jasna.ltx.transformer.require_kernels", Mock(side_effect=RuntimeError("normal kernel preflight")))
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    config = session_config(restoration_model_name="ltx", restoration_model_path=tmp_path)
    with pytest.raises(RuntimeError, match="normal download preflight"):
        main._ensure_ltx_model(config, None)
    with pytest.raises(RuntimeError, match="normal kernel preflight"):
        session_factory._ltx_model_files(config, torch.device("cuda:0"), log_callback=None)
