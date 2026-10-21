import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from factories import offload_bundle_targets

import jasna.media.cuda_kernel as cuda_kernel
from jasna.media.cuda_kernel import HIP_KERNEL_ARCHITECTURES, Kernel, has_precompiled_kernels

MEDIA = Path(cuda_kernel.__file__).resolve().parent
KERNEL_SOURCES = sorted(MEDIA.glob("*.cu"))


def _use_rocm(monkeypatch, architecture: str = "gfx1201") -> None:
    monkeypatch.setattr(torch.version, "hip", "7.15.26333")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName=f"{architecture}:sramecc-:xnack-"),
    )


def test_nvidia_always_runs_the_kernels(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    assert has_precompiled_kernels(torch.device("cuda:0"))
    assert not has_precompiled_kernels(torch.device("cpu"))


@pytest.mark.parametrize("architecture", sorted(HIP_KERNEL_ARCHITECTURES))
def test_amd_runs_the_kernels_on_bundled_architectures(monkeypatch, architecture):
    _use_rocm(monkeypatch, architecture)

    assert has_precompiled_kernels(torch.device("cuda:0"))


def test_amd_falls_back_to_torch_on_an_architecture_missing_from_the_bundle(monkeypatch):
    _use_rocm(monkeypatch, "gfx1036")

    assert not has_precompiled_kernels(torch.device("cuda:0"))
    assert not has_precompiled_kernels(torch.device("cpu"))


@pytest.mark.parametrize("source", KERNEL_SOURCES, ids=lambda path: path.stem)
def test_every_kernel_ships_for_both_vendors(source):
    assert source.with_suffix(".fatbin").is_file()
    targets = offload_bundle_targets(source.with_suffix(".hipfb"))
    assert {target.rpartition("--")[2] for target in targets if target.startswith("hipv4-")} == (
        HIP_KERNEL_ARCHITECTURES
    )


class _FakeHip:
    def __init__(self, launch_result: int = 0):
        self.loaded: list[bytes] = []
        self.functions: list[bytes] = []
        self.launches: list[tuple] = []
        self.launch_result = launch_result

    def _named(self, name, body):
        body.__name__ = name
        setattr(self, name, body)

    def install(self, monkeypatch):
        def get_device(device):
            device._obj.value = 1
            return 0

        def load_data(module, image):
            self.loaded.append(image.raw[:-1])
            module._obj.value = 0x1000
            return 0

        def get_function(function, module, name):
            self.functions.append(name)
            function._obj.value = 0x2000
            return 0

        def launch(function, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
            self.launches.append((function.value, (gx, gy, gz), (bx, by, bz), shared, stream.value))
            return self.launch_result

        for name, body in (
            ("hipGetDevice", get_device),
            ("hipModuleLoadData", load_data),
            ("hipModuleGetFunction", get_function),
            ("hipModuleLaunchKernel", launch),
            ("hipGetErrorName", lambda result: b"hipErrorLaunchFailure"),
            ("hipGetErrorString", lambda result: b"unspecified launch failure"),
        ):
            self._named(name, body)
        monkeypatch.setattr(cuda_kernel, "hip_runtime", lambda: self)
        monkeypatch.setattr(cuda_kernel, "_modules", {})


def test_amd_kernel_loads_the_hip_bundle_and_launches_on_the_given_stream(monkeypatch):
    _use_rocm(monkeypatch)
    hip = _FakeHip()
    hip.install(monkeypatch)
    kernel = Kernel("cas", "cas_luma8", (ctypes.c_uint64, ctypes.c_float))

    kernel.launch((4, 2, 1), (32, 8, 1), (123, 0.5), 0xBEEF)
    kernel.launch((4, 2, 1), (32, 8, 1), (456, 0.25), 0xBEEF)

    assert hip.loaded == [(MEDIA / "cas.hipfb").read_bytes()]
    assert hip.functions == [b"cas_luma8"]
    assert hip.launches == [(0x2000, (4, 2, 1), (32, 8, 1), 0, 0xBEEF)] * 2


def test_kernel_launches_with_its_dynamic_shared_memory(monkeypatch):
    _use_rocm(monkeypatch)
    hip = _FakeHip()
    hip.install(monkeypatch)
    kernel = Kernel("cas", "cas_luma8", (ctypes.c_uint64,), 8192)

    kernel.launch((128, 1, 1), (128, 1, 1), (0,), 0xBEEF)

    assert hip.launches == [(0x2000, (128, 1, 1), (128, 1, 1), 8192, 0xBEEF)]


def test_amd_launch_failure_names_the_hip_error(monkeypatch):
    _use_rocm(monkeypatch)
    _FakeHip(launch_result=719).install(monkeypatch)
    kernel = Kernel("cas", "cas_luma8", (ctypes.c_uint64,))

    with pytest.raises(RuntimeError, match=r"hipModuleLaunchKernel\(cas_luma8\) failed: hipErrorLaunchFailure"):
        kernel.launch((1, 1, 1), (1, 1, 1), (0,), 0)


class _HostRegisterRecorder:
    def __init__(self):
        self.calls = []

    def hipHostRegister(self, pointer, nbytes, flags):
        self.calls.append(("hipHostRegister", pointer, nbytes, flags))
        return 0

    def hipHostUnregister(self, pointer):
        self.calls.append(("hipHostUnregister", pointer))
        return 0

    def cuMemHostRegister_v2(self, pointer, nbytes, flags):
        self.calls.append(("cuMemHostRegister_v2", pointer, nbytes, flags))
        return 0

    def cuMemHostUnregister(self, pointer):
        self.calls.append(("cuMemHostUnregister", pointer))
        return 0


@pytest.mark.parametrize(
    ("hip", "register", "unregister"),
    [("7.15.26333", "hipHostRegister", "hipHostUnregister"), (None, "cuMemHostRegister_v2", "cuMemHostUnregister")],
)
def test_host_memory_is_page_locked_portably_through_the_vendor_runtime(monkeypatch, hip, register, unregister):
    recorder = _HostRegisterRecorder()
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", None if hip else "13.0")
    monkeypatch.setattr(cuda_kernel, "hip_runtime", lambda: recorder)
    monkeypatch.setattr(cuda_kernel, "cuda_driver", lambda: recorder)

    assert cuda_kernel.host_register(4096, 1 << 20) == 0
    cuda_kernel.host_unregister(4096)

    assert recorder.calls == [(register, 4096, 1 << 20, 1), (unregister, 4096)]
