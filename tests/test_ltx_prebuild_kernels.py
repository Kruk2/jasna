"""Release kernel profile selection without compiling code or initializing CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from jasna.ltx import native
from jasna.ltx.model_files import PlaceholderHandle
from jasna.ltx.native.aot import BlockDims


PROFILES = [(7, 5), (8, 6), (8, 9), (12, 0), (12, 1)]


def _forbidden(*args, **kwargs):
    raise AssertionError("kernel-only prebuild invoked CUDA, an extension build, or tensor allocation")


def _no_runtime_work(monkeypatch):
    for name in ("init", "_lazy_init", "current_device", "get_device_capability", "get_device_properties",
                 "get_arch_list", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, _forbidden)
    for name in ("empty", "zeros", "randn"):
        monkeypatch.setattr(torch, name, _forbidden)
    monkeypatch.setattr(PlaceholderHandle, "get_tensor_into", _forbidden)
    monkeypatch.setattr(native, "_build_extension", _forbidden)
    monkeypatch.setattr(native.subprocess, "run", _forbidden)


def test_kernel_only_prebuild_uses_exact_public_profiles_and_shapes(monkeypatch):
    _no_runtime_work(monkeypatch)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "unchanged")
    monkeypatch.setattr(native, "cuda_arch_list", _forbidden)
    calls = []
    monkeypatch.setattr(native, "build_kernels", lambda capability, dims, *, fp4, cache:
                        calls.append((capability, dims, fp4, cache)))
    directory = Path("owned-kernel-prebuild")
    assert native.prebuild_kernels(directory) is None
    expected = [(capability, BlockDims(4096, 16384, 32), False, directory / "kernels") for capability in PROFILES]
    expected += [(capability, BlockDims(4096, 16384, 32), True, directory / "kernels") for capability in [(12, 0), (12, 1)]]
    assert native.release_architectures() == PROFILES
    assert calls == expected
    assert native.os.environ["TORCH_CUDA_ARCH_LIST"] == "unchanged"


def test_prebuild_profiles_are_seven_pure_public_architecture_specs(monkeypatch):
    _no_runtime_work(monkeypatch)
    monkeypatch.setattr(native, "build_kernels", _forbidden)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "unchanged")
    expected = [(capability, BlockDims(4096, 16384, 32), False) for capability in PROFILES]
    expected += [(capability, BlockDims(4096, 16384, 32), True) for capability in [(12, 0), (12, 1)]]
    profiles = native.prebuild_profiles()
    assert profiles == expected
    profiles.clear()
    assert native.prebuild_profiles() == expected
    assert native.os.environ["TORCH_CUDA_ARCH_LIST"] == "unchanged"


def test_kernel_only_route_delegates_profile_selection(monkeypatch):
    profile = ((8, 6), BlockDims(4096, 16384, 32), False)
    calls = []
    monkeypatch.setattr(native, "prebuild_profiles", lambda: [profile])
    monkeypatch.setattr(native, "release_architectures", _forbidden)
    monkeypatch.setattr(native, "build_kernels", lambda capability, dims, *, fp4, cache:
                        calls.append((capability, dims, fp4, cache)))
    native.prebuild_kernels(Path("owned-kernel-prebuild"))
    assert calls == [(*profile, Path("owned-kernel-prebuild/kernels"))]


def test_no_import_time_prebuild_work():
    tree = ast.parse(Path(native.__file__).read_text())
    names = {"prebuild", "prebuild_profiles", "prebuild_kernels", "build_kernels", "_build_extension"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for item in ast.walk(node):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                assert item.func.id not in names


@pytest.mark.parametrize("failure_index", [0, 3, 6])
def test_kernel_failure_propagates_without_later_compilation(monkeypatch, failure_index):
    _no_runtime_work(monkeypatch)
    calls = []
    failure = RuntimeError("owned compiler failure")

    def compile_kernel(capability, dims, *, fp4, cache):
        calls.append((capability, fp4))
        if len(calls) == failure_index + 1:
            raise failure

    monkeypatch.setattr(native, "build_kernels", compile_kernel)
    with pytest.raises(RuntimeError) as caught:
        native.prebuild_kernels(Path("owned-kernel-prebuild"))
    assert caught.value is failure
    assert len(calls) == failure_index + 1


def test_prebuild_delegates_then_preserves_extension_copy_and_cleanup(monkeypatch):
    directory = Path("owned-public-prebuild")
    module = SimpleNamespace(__file__=str(directory / "build" / "jasna_ltx_native.pyd"))
    calls = []
    monkeypatch.setattr(native, "prebuild_kernels", lambda path: calls.append(("kernels", path)))
    monkeypatch.setattr(native, "cuda_arch_list", lambda: "7.5;8.6;8.9;12.0;12.1")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "previous")

    def extension(path):
        calls.append(("extension", path, native.os.environ["TORCH_CUDA_ARCH_LIST"]))
        return module

    monkeypatch.setattr(native, "_build_extension", extension)
    monkeypatch.setattr(native.shutil, "copy2", lambda source, target: calls.append(("copy", source, target)))
    monkeypatch.setattr(native.shutil, "rmtree", lambda path: calls.append(("cleanup", path)))
    assert native.prebuild(directory) is None
    assert calls == [("kernels", directory),
                     ("extension", directory / "build", "7.5;8.6;8.9;12.0;12.1"),
                     ("copy", module.__file__, directory / "jasna_ltx_native.pyd"),
                     ("cleanup", directory / "build")]


def test_prebuild_kernel_failure_prevents_extension_and_env_change(monkeypatch):
    failure = RuntimeError("owned kernel failure")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "previous")

    def fail(path):
        raise failure

    monkeypatch.setattr(native, "prebuild_kernels", fail)
    monkeypatch.setattr(native, "_build_extension", _forbidden)
    monkeypatch.setattr(native, "cuda_arch_list", _forbidden)
    with pytest.raises(RuntimeError) as caught:
        native.prebuild(Path("owned-public-prebuild"))
    assert caught.value is failure
    assert native.os.environ["TORCH_CUDA_ARCH_LIST"] == "previous"


def test_prebuild_extension_failure_propagates_without_copy_or_cleanup(monkeypatch):
    calls = []
    failure = RuntimeError("owned extension failure")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "previous")
    monkeypatch.setattr(native, "prebuild_kernels", lambda directory: calls.append(directory))

    def fail(directory):
        raise failure

    monkeypatch.setattr(native, "_build_extension", fail)
    monkeypatch.setattr(native.shutil, "copy2", _forbidden)
    monkeypatch.setattr(native.shutil, "rmtree", _forbidden)
    with pytest.raises(RuntimeError) as caught:
        native.prebuild(Path("owned-public-prebuild"))
    assert caught.value is failure
    assert calls == [Path("owned-public-prebuild")]
