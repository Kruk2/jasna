import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from jasna.restorer import basicvsrpp_migraphx_b1 as module


def test_auto_skips_ineligible_cpu_without_gpu_probe(monkeypatch):
    monkeypatch.setattr(
        module,
        "is_amd_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("vendor queried")),
    )

    assert not module.basicvsrpp_migraphx_b1_enabled(
        torch.device("cpu"), fp16=True, checkpoint_path="model.pth", environ={}
    )


def test_invalid_switch_is_rejected():
    with pytest.raises(ValueError, match=module.BASICVSRPP_MIGRAPHX_B1_ENV):
        module.basicvsrpp_migraphx_b1_enabled(
            torch.device("cpu"),
            fp16=True,
            checkpoint_path="model.pth",
            environ={module.BASICVSRPP_MIGRAPHX_B1_ENV: "maybe"},
        )


def _mock_supported_host(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(module.torch.version, "hip", "7.2.1")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1100"),
    )


def test_auto_selects_an_installed_artifact_set(monkeypatch, tmp_path):
    _mock_supported_host(monkeypatch)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    for name in (
        "B1_COLD_MANIFEST.json",
        "B1_COLD_MANIFEST.sha256",
        *(f"b1_{direction}.torch" for direction in module._DIRECTIONS),
    ):
        (directory / name).write_bytes(b"artifact")

    assert module.basicvsrpp_migraphx_b1_enabled(
        torch.device("cuda:0"),
        fp16=True,
        checkpoint_path=tmp_path / "model.pth",
        environ={module.BASICVSRPP_MIGRAPHX_B1_DIR_ENV: str(directory)},
    )


def test_explicit_enable_fails_when_artifacts_are_missing(monkeypatch, tmp_path):
    _mock_supported_host(monkeypatch)

    with pytest.raises(RuntimeError, match="missing artifact files"):
        module.basicvsrpp_migraphx_b1_enabled(
            torch.device("cuda:0"),
            fp16=True,
            checkpoint_path=tmp_path / "model.pth",
            environ={module.BASICVSRPP_MIGRAPHX_B1_ENV: "1"},
        )


def test_preload_reuses_torch_cpp_extension_owned_module(monkeypatch, tmp_path):
    extension = tmp_path / "cache" / "_torch_migraphx.so"
    extension.parent.mkdir()
    extension.write_bytes(b"accepted extension")
    expected_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    monkeypatch.delitem(sys.modules, "_torch_migraphx", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch_migraphx._C",
        SimpleNamespace(_mod=SimpleNamespace(__file__=str(extension))),
    )

    selected = module._preload_torch_migraphx_extension(
        tmp_path / "artifacts", expected_sha256
    )

    assert selected == extension.resolve()


def test_preload_accepts_equal_bytes_cache_copy(monkeypatch, tmp_path):
    artifact = tmp_path / "artifacts" / "_torch_migraphx.so"
    cache = tmp_path / "cache" / "_torch_migraphx.so"
    artifact.parent.mkdir()
    cache.parent.mkdir()
    artifact.write_bytes(b"accepted extension")
    cache.write_bytes(artifact.read_bytes())
    expected_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.delitem(sys.modules, "_torch_migraphx", raising=False)
    monkeypatch.delitem(sys.modules, "torch_migraphx._C", raising=False)
    monkeypatch.setattr(module, "_extension_candidates", lambda _directory: [artifact])
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(cache)),
    )

    selected = module._preload_torch_migraphx_extension(
        artifact.parent, expected_sha256
    )

    assert selected == cache.resolve()


def test_preload_rejects_different_loaded_cache_copy(monkeypatch, tmp_path):
    artifact = tmp_path / "artifacts" / "_torch_migraphx.so"
    cache = tmp_path / "cache" / "_torch_migraphx.so"
    artifact.parent.mkdir()
    cache.parent.mkdir()
    artifact.write_bytes(b"accepted extension")
    cache.write_bytes(b"different extension")
    expected_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.delitem(sys.modules, "_torch_migraphx", raising=False)
    monkeypatch.delitem(sys.modules, "torch_migraphx._C", raising=False)
    monkeypatch.setattr(module, "_extension_candidates", lambda _directory: [artifact])
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(cache)),
    )

    with pytest.raises(RuntimeError, match="differs from the selected binary"):
        module._preload_torch_migraphx_extension(
            artifact.parent, expected_sha256
        )


def test_strict_artifact_clones_reused_output():
    direction = "backward_1"
    output = torch.zeros((1, 64, 64, 64), dtype=torch.float16)
    artifact = module._StrictArtifact(
        direction, lambda *_values: output, torch.device("cpu")
    )
    inputs = tuple(
        torch.zeros(shape, dtype=torch.float16)
        for shape in module._artifact_shapes(direction)
    )

    result = artifact(*inputs)

    assert torch.equal(result, output)
    assert result.data_ptr() != output.data_ptr()
