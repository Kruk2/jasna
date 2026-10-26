"""Release kernel prebuild without compiling code or initializing CUDA."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from jasna.ltx import native
from jasna.ltx.kernel_policy import BlockDims
from jasna.ltx.model_files import PlaceholderHandle
from jasna.ltx.native import aot


LTX = BlockDims(4096, 16384, 32)
NVIDIA_PROFILES = [*((capability, LTX, False) for capability in [(7, 5), (8, 6), (8, 9), (12, 0), (12, 1)]),
                   ((12, 0), LTX, True), ((12, 1), LTX, True)]
AMD_PROFILES = [(gfx, LTX, False) for gfx in
                ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"]]


def _forbidden(*args, **kwargs):
    raise AssertionError("kernel-only prebuild invoked CUDA, an extension build, or tensor allocation")


@pytest.mark.parametrize("hip, profiles", [(None, NVIDIA_PROFILES), ("7.15", AMD_PROFILES)])
def test_prebuild_compiles_every_release_profile_without_a_gpu(monkeypatch, hip, profiles):
    for name in ("init", "_lazy_init", "current_device", "get_device_capability", "get_device_properties",
                 "get_arch_list", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, _forbidden)
    for name in ("empty", "zeros", "randn"):
        monkeypatch.setattr(torch, name, _forbidden)
    monkeypatch.setattr(PlaceholderHandle, "get_tensor_into", _forbidden)
    monkeypatch.setattr(native, "_build_extension", _forbidden)
    monkeypatch.setattr(native.subprocess, "run", _forbidden)
    monkeypatch.setattr(torch.version, "hip", hip)
    built = []
    monkeypatch.setattr(aot, "build_kernels", lambda architecture, dims, *, fp4, cache:
                        built.append((architecture, dims, fp4, cache)))
    native.prebuild_kernels(Path("prebuild"))
    assert built == [(*profile, Path("prebuild/kernels")) for profile in profiles]


@pytest.mark.parametrize("hip, variable, targets", [
    (None, "TORCH_CUDA_ARCH_LIST", "7.5;8.6;8.9;12.0;12.1"),
    ("7.15", "PYTORCH_ROCM_ARCH", ";".join(architecture for architecture, _, _ in AMD_PROFILES)),
])
def test_prebuild_builds_the_extension_for_every_release_gpu(monkeypatch, tmp_path, hip, variable, targets):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setenv(variable, "previous")
    monkeypatch.setattr(native, "prebuild_kernels", lambda directory: None)
    seen = []

    def build(directory):
        seen.append(native.os.environ[variable])
        directory.mkdir()
        (directory / "jasna_ltx_native.so").write_bytes(b"extension")
        return SimpleNamespace(__file__=str(directory / "jasna_ltx_native.so"))

    monkeypatch.setattr(native, "_build_extension", build)
    native.prebuild(tmp_path)
    assert seen == [targets] and native.os.environ[variable] == "previous"
    assert (tmp_path / "jasna_ltx_native.so").read_bytes() == b"extension"


def test_a_failed_kernel_stops_the_prebuild(monkeypatch):
    failure = RuntimeError("compiler failure")
    built = []

    def compile_kernel(architecture, dims, *, fp4, cache):
        built.append(architecture)
        raise failure

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "previous")
    monkeypatch.setattr(aot, "build_kernels", compile_kernel)
    monkeypatch.setattr(native, "_build_extension", _forbidden)
    with pytest.raises(RuntimeError) as caught:
        native.prebuild(Path("prebuild"))
    assert caught.value is failure and len(built) == 1
    assert native.os.environ["TORCH_CUDA_ARCH_LIST"] == "previous"
