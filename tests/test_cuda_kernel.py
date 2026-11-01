import ctypes
from pathlib import Path

import pytest
import torch
from factories import offload_bundle_targets

import jasna.media.cuda_kernel as cuda_kernel
from jasna.accelerator import HIP_KERNEL_ARCHITECTURES, TRITON_HIP_ARCHITECTURES
from jasna.media.cuda_kernel import Kernel, TritonKernel, aot_kernel, has_aot_kernel, has_precompiled_kernels
from jasna.models.ltx_vae import hip_launch as ltx_vae_launch
from jasna.mosaic import rfdetr_hip_launch
from jasna.restorer import basicvsrpp_hip_launch

MEDIA = Path(cuda_kernel.__file__).resolve().parent
KERNEL_SOURCES = sorted(MEDIA.glob("*.cu"))
TRITON_LAUNCH_TABLES = {
    table.__name__.rpartition(".")[2]: table.KERNEL_LAUNCH
    for table in (ltx_vae_launch, rfdetr_hip_launch, basicvsrpp_hip_launch)
}


def test_nvidia_always_runs_the_kernels(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    assert has_precompiled_kernels(torch.device("cuda:0"))
    assert not has_precompiled_kernels(torch.device("cpu"))


@pytest.mark.parametrize("architecture", sorted(HIP_KERNEL_ARCHITECTURES))
def test_amd_runs_the_kernels_on_bundled_architectures(amd_gpu, architecture):
    amd_gpu(architecture)

    assert has_precompiled_kernels(torch.device("cuda:0"))


def test_amd_falls_back_to_torch_on_an_architecture_missing_from_the_bundle(amd_gpu):
    amd_gpu("gfx1036")

    assert not has_precompiled_kernels(torch.device("cuda:0"))
    assert not has_precompiled_kernels(torch.device("cpu"))


@pytest.mark.parametrize("source", KERNEL_SOURCES, ids=lambda path: path.stem)
def test_every_kernel_ships_for_both_vendors(source):
    assert source.with_suffix(".fatbin").is_file()
    targets = offload_bundle_targets(source.with_suffix(".hipfb"))
    assert {target.rpartition("--")[2] for target in targets if target.startswith("hipv4-")} == (
        HIP_KERNEL_ARCHITECTURES
    )


@pytest.mark.parametrize(
    ("table", "module_name"),
    [(name, module) for name, table in TRITON_LAUNCH_TABLES.items() for module in sorted(table["gfx1201"])],
)
def test_every_triton_kernel_ships_for_every_triton_architecture(table, module_name):
    launch = TRITON_LAUNCH_TABLES[table]
    targets = offload_bundle_targets(MEDIA / f"{module_name}.hipfb")

    assert {target.rpartition("--")[2] for target in targets if target.startswith("hipv4-")} == set(TRITON_HIP_ARCHITECTURES)
    assert set(launch) == set(TRITON_HIP_ARCHITECTURES)
    assert all(module_name in kernels for kernels in launch.values())


def test_the_triton_architectures_are_media_kernel_architectures():
    assert set(TRITON_HIP_ARCHITECTURES) < HIP_KERNEL_ARCHITECTURES


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


def test_amd_kernel_loads_the_hip_bundle_and_launches_on_the_given_stream(amd_gpu, monkeypatch):
    amd_gpu()
    hip = _FakeHip()
    hip.install(monkeypatch)
    kernel = Kernel("cas", "cas_luma8", (ctypes.c_uint64, ctypes.c_float))

    kernel.launch((4, 2, 1), (32, 8, 1), (123, 0.5), 0xBEEF)
    kernel.launch((4, 2, 1), (32, 8, 1), (456, 0.25), 0xBEEF)

    assert hip.loaded == [(MEDIA / "cas.hipfb").read_bytes()]
    assert hip.functions == [b"cas_luma8"]
    assert hip.launches == [(0x2000, (4, 2, 1), (32, 8, 1), 0, 0xBEEF)] * 2


def test_kernel_launches_with_its_dynamic_shared_memory(amd_gpu, monkeypatch):
    amd_gpu()
    hip = _FakeHip()
    hip.install(monkeypatch)
    kernel = Kernel("cas", "cas_luma8", (ctypes.c_uint64,), 8192)

    kernel.launch((128, 1, 1), (128, 1, 1), (0,), 0xBEEF)

    assert hip.launches == [(0x2000, (128, 1, 1), (128, 1, 1), 8192, 0xBEEF)]


def test_amd_launch_failure_names_the_hip_error(amd_gpu, monkeypatch):
    amd_gpu()
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


_LAUNCH_TABLE = {"gfx1201": {"demo": ("demo_kernel", (4, 1, 1), 128, 2048)}}


def test_triton_kernel_passes_tensors_as_pointers_then_ints_floats_and_the_unused_scratch(amd_gpu, monkeypatch):
    amd_gpu()
    hip = _FakeHip()
    hip.install(monkeypatch)
    kernel = TritonKernel("cas", ("cas_luma8", (4, 1, 1), 128, 2048), {"x": "*bf16:wide", "n": "i32:16", "scale": "fp32"})
    x = torch.zeros(4)

    kernel.launch((4, 1, 1), 0xBEEF, x, 32, 0.5)

    assert [value.value for value in kernel._kernel._values] == [x.data_ptr(), 32, 0.5, None, None]
    assert hip.launches == [(0x2000, (4, 1, 1), (128, 1, 1), 2048, 0xBEEF)]


def test_aot_kernel_is_built_once_per_architecture(amd_gpu, monkeypatch):
    amd_gpu()
    monkeypatch.setattr(cuda_kernel, "_aot_kernels", {})

    kernel = aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cuda:0"), {"x": "*fp16"})

    assert aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cuda:0"), {"x": "*fp16"}) is kernel
    assert (kernel.grid, kernel.block) == ((4, 1, 1), (128, 1, 1))


def test_a_module_without_a_built_kernel_fails_loudly(amd_gpu):
    amd_gpu("gfx1100")

    with pytest.raises(RuntimeError, match="no ahead-of-time kernel demo on gfx1100"):
        aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cuda:0"), {"x": "*fp16"})


def test_only_amd_gpus_in_the_launch_table_have_an_aot_kernel(amd_gpu, monkeypatch):
    amd_gpu()
    assert has_aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cuda:0"))
    assert not has_aot_kernel(_LAUNCH_TABLE, "other", torch.device("cuda:0"))
    assert not has_aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cpu"))

    amd_gpu("gfx1030")
    assert not has_aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cuda:0"))

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert not has_aot_kernel(_LAUNCH_TABLE, "demo", torch.device("cuda:0"))


def _pe_image_importing(*dlls: tuple[bytes, int]) -> bytes:
    """A PE32+ image whose import table lists ``dlls`` as (name, first import address) pairs."""
    image = bytearray(0x1000)
    image[0x3C:0x40] = (0x80).to_bytes(4, "little")
    descriptors = 0x200
    image[0x80 + 24 + 112 + 8:0x80 + 24 + 112 + 12] = descriptors.to_bytes(4, "little")
    for index, (name, address) in enumerate(dlls):
        descriptor, name_at, thunk_at = descriptors + 20 * index, 0x400 + 0x40 * index, 0x800 + 8 * index
        image[name_at:name_at + len(name) + 1] = name + b"\0"
        image[thunk_at:thunk_at + 8] = address.to_bytes(8, "little")
        image[descriptor + 12:descriptor + 16] = name_at.to_bytes(4, "little")
        image[descriptor + 16:descriptor + 20] = thunk_at.to_bytes(4, "little")
    return bytes(image)


def _reader(image: bytes):
    return lambda offset, size: image[offset:offset + size]


def test_finds_the_runtime_a_dll_imports_by_name_prefix():
    image = _pe_image_importing((b"KERNEL32.dll", 0x7FF8_0000_1000), (b"AMDHIP64_7.dll", 0x7FF8_1234_5678))
    assert cuda_kernel.imported_function_address(_reader(image), b"amdhip64") == 0x7FF8_1234_5678


def test_a_dll_that_does_not_import_the_runtime_has_no_address():
    image = _pe_image_importing((b"KERNEL32.dll", 0x7FF8_0000_1000), (b"amd_comgr.dll", 0x7FF8_0000_2000))
    assert cuda_kernel.imported_function_address(_reader(image), b"amdhip64") is None
