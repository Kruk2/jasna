from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest
import torch

from jasna import engine_paths
from jasna.trt import torch_tensorrt_export as exports

export_cache = pytest.importorskip("jasna.protection.export_cache")


def test_frozen_engine_preflight_checks_each_cached_module(monkeypatch):
    monkeypatch.setattr(engine_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    paths = engine_paths.get_basicvsrpp_sub_engine_paths("models/example.pth", True, "sm89")
    seen = []

    def available(path):
        seen.append(path)
        return True

    monkeypatch.setattr(export_cache, "available", available)
    assert engine_paths.all_basicvsrpp_sub_engines_exist("models/example.pth", True, "sm89")
    assert seen == list(paths.values())
    monkeypatch.setattr(export_cache, "available", lambda path: False)
    assert not engine_paths.all_basicvsrpp_sub_engines_exist("models/example.pth", True, "sm89")


def test_frozen_module_loading_uses_the_runtime_reader(monkeypatch):
    monkeypatch.setattr(exports, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(exports, "_mute_torch_tensorrt", lambda: None)
    module = torch.nn.Identity()
    seen = []

    def load(path, device):
        seen.append((path, device))
        return module

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected export reader")

    monkeypatch.setattr(export_cache, "load", load)
    monkeypatch.setattr(torch.export, "load", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    device = torch.device("cpu")
    assert exports.load_torchtrt_export(checkpoint_path="module.engine", device=device) is module
    assert seen == [("module.engine", device)]


def test_frozen_compilation_uses_the_runtime_writer(monkeypatch):
    monkeypatch.setattr(exports, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(exports, "_mute_torch_tensorrt", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    module = torch.nn.Identity()
    monkeypatch.setitem(sys.modules, "torch_tensorrt", SimpleNamespace(
        Input=type("Input", (), {}), compile=lambda *args, **kwargs: module))
    seen = []
    monkeypatch.setattr(export_cache, "save", lambda value, path: seen.append((value, path)))
    path = exports.compile_and_save_torchtrt_dynamo(
        module=module, inputs=[torch.ones(2)], output_path="module.engine",
        dtype=torch.float32, workspace_size_bytes=1024, message="compile",
        device=torch.device("cpu"))
    assert path == "module.engine"
    assert seen == [(module, path)]


@pytest.mark.parametrize("fails", [False, True])
def test_a_source_engine_file_is_saved_whole_or_not_at_all(tmp_path, monkeypatch, fails):
    monkeypatch.setattr(exports, "is_frozen", lambda: False)
    monkeypatch.setattr(exports, "_mute_torch_tensorrt", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def save(module, path, inputs):
        with open(path, "wb") as out:
            out.write(b"half")
            if fails:
                raise OSError("disk full")
            out.write(b" and the rest")

    monkeypatch.setitem(sys.modules, "torch_tensorrt", SimpleNamespace(
        Input=type("Input", (), {}), compile=lambda *args, **kwargs: torch.nn.Identity(), save=save))
    engine = tmp_path / "module.engine"

    def compile_engine():
        exports.compile_and_save_torchtrt_dynamo(
            module=torch.nn.Identity(), inputs=[torch.ones(2)], output_path=str(engine),
            dtype=torch.float32, workspace_size_bytes=1024, message="compile", device=torch.device("cpu"))

    if fails:
        with pytest.raises(OSError, match="disk full"):
            compile_engine()
    else:
        compile_engine()

    saved = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert saved == ({} if fails else {"module.engine": b"half and the rest"})


def test_source_export_remains_loadable_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(exports, "is_frozen", lambda: False)
    monkeypatch.setattr(exports, "_mute_torch_tensorrt", lambda: None)
    value = torch.ones(2)
    program = torch.export.export(torch.nn.Identity(), (value,))
    path = tmp_path / "module.engine"
    torch.export.save(program, path)
    module = exports.load_torchtrt_export(checkpoint_path=str(path), device=torch.device("cpu"))
    assert torch.equal(module(value), value)
