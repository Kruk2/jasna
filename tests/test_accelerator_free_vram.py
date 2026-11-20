import ctypes
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from jasna import accelerator

GIB = 1 << 30


@pytest.mark.usefixtures("nvidia_vendor")
def test_windows_caps_the_process_view_with_the_device_wide_free_vram(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (30 * GIB, 32 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: 12 * GIB)
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == 12 * GIB


def test_linux_reads_cuda_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (11 * GIB, 32 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: pytest.fail("NVML read on Linux"))
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == 11 * GIB


_DGPU_LUID = bytes.fromhex("8f6a010000000000")
_IGPU_LUID = bytes.fromhex("1c7b010000000000")


@pytest.fixture
def amd_windows(monkeypatch):
    """Windows AMD with an RX 9070 XT and an iGPU; ``set_adapters`` sets what DXGI lists, or the error it raises."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: pytest.fail("NVML read on AMD"))
    monkeypatch.setattr(accelerator, "_hip_device_luid", lambda index: _DGPU_LUID)

    def set_adapters(adapters):
        def listed():
            if isinstance(adapters, OSError):
                raise adapters
            return adapters

        monkeypatch.setattr(accelerator, "windows_video_memory", listed)

    return set_adapters


@pytest.mark.parametrize(
    ("free", "budget", "usage", "expected"),
    [(20 * GIB, 15 * GIB, 3 * GIB, 12 * GIB), (10 * GIB, 15 * GIB, 3 * GIB, 10 * GIB), (20 * GIB, 15 * GIB, 16 * GIB, 0)],
)
def test_free_vram_is_capped_at_what_the_windows_budget_has_left(free, budget, usage, expected):
    assert accelerator.budget_capped_free_bytes(free, budget, usage) == expected


@pytest.mark.parametrize(
    ("adapters", "expected"),
    [
        ({_IGPU_LUID: ("AMD Radeon(TM) Graphics", 30 * GIB, 0), _DGPU_LUID: ("AMD Radeon RX 9070 XT", 15 * GIB, 3 * GIB)},
         12 * GIB),
        # no adapter with the GPU's LUID: HIP's figure, as before
        ({_IGPU_LUID: ("AMD Radeon(TM) Graphics", 30 * GIB, 0)}, 20 * GIB),
        # DXGI refused (as it might inside the restricted LTX worker): HIP's figure, the job goes on
        (OSError("DXGI CreateDXGIFactory1 failed (HRESULT 0x80070005)"), 20 * GIB),
    ],
)
def test_amd_windows_caps_hip_with_the_gpus_windows_budget(monkeypatch, amd_windows, adapters, expected):
    amd_windows(adapters)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * GIB, 32 * GIB))
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == expected


def test_worker_free_vram_on_amd_windows_is_capped_by_its_own_budget(monkeypatch, amd_windows):
    from jasna.ltx import inference

    amd_windows({_DGPU_LUID: ("AMD Radeon RX 9070 XT", 13 * GIB, 3 * GIB)})
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (15 * GIB, 16 * GIB))
    free_vram = inference.free_vram_beside_parent(torch.device("cuda:0"), 12 * GIB)
    assert free_vram(torch.device("cuda:0")) == 10 * GIB


@pytest.mark.parametrize(("usage", "over"), [(16 * GIB, True), (14 * GIB, False)])
def test_amd_windows_warns_when_windows_moved_vram_to_system_ram(caplog, amd_windows, usage, over):
    amd_windows({_DGPU_LUID: ("AMD Radeon RX 9070 XT", 15 * GIB, usage)})
    with caplog.at_level("WARNING", logger=accelerator.log.name):
        assert accelerator.warn_over_windows_budget(torch.device("cuda:0")) is over
    assert bool(caplog.records) is over


_NvmlQuery = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(accelerator._NvmlMemory))


def _fake_nvml(total: int, free: int):
    def query(handle, out):
        out.contents.total = total
        out.contents.free = free
        out.contents.used = total - free
        return 0

    return SimpleNamespace(nvmlDeviceGetMemoryInfo=_NvmlQuery(query))


@pytest.mark.parametrize(
    ("nvml_total", "cuda_total", "disagrees"),
    [
        # Measured GB10 row: NVML answers the v1 query with a bogus ~62.5 GiB pool while CUDA sees 86.68 GiB.
        (67_108_864_000, 93_066_629_120, True),
        # A real dGPU: NVML's total stays within the WDDM reservation, well under 5% of CUDA's.
        (105 * GIB, 110 * GIB, False),
    ],
)
def test_nvml_total_disagrees_with_cuda(nvml_total, cuda_total, disagrees):
    assert accelerator._nvml_total_disagrees_with_cuda(nvml_total, cuda_total) is disagrees


@pytest.mark.usefixtures("nvidia_vendor")
def test_windows_soc_gpu_caps_the_process_view_at_the_host_available_ram(monkeypatch):
    # Same measured GB10 row through device_free_bytes.
    monkeypatch.setattr(sys, "platform", "win32")
    props = SimpleNamespace(uuid="abc", total_memory=93_066_629_120)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (92_648_583_168, 93_066_629_120))
    monkeypatch.setattr(accelerator, "_nvml_device", lambda uuid: (_fake_nvml(67_108_864_000, 475_320_320), None))
    monkeypatch.setattr(accelerator.psutil, "virtual_memory", lambda: SimpleNamespace(available=14_832_648_192))
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == 14_832_648_192


def test_nvml_total_within_wddm_reserve_keeps_nvml_figure(monkeypatch):
    # A real dGPU: NVML's total stays within 5% of CUDA's (the WDDM reservation).
    props = SimpleNamespace(uuid="abc", total_memory=110 * GIB)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (109 * GIB, 110 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_device", lambda uuid: (_fake_nvml(105 * GIB, 40 * GIB), None))
    assert accelerator._nvml_free_bytes(torch.device("cuda:0")) == 40 * GIB


_HOLD_VRAM = """
import sys
import torch
held = torch.ones(int(sys.argv[1]), dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
print("held", flush=True)
sys.stdin.read()
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_vram_held_by_another_process_is_not_free():
    device = torch.device("cuda:0")
    # On a SoC GPU the NVML clamp is replaced by the host's available-RAM bound, which
    # a commit-backed unified pool does not track, so the device-wide check cannot apply.
    if sys.platform == "win32" and not torch.version.hip:
        props, memory = accelerator._nvml_memory(device)
        if accelerator._nvml_total_disagrees_with_cuda(memory.total, props.total_memory):
            pytest.skip("SoC GPU (NVML/CUDA total mismatch): the NVML clamp is replaced by the host available bound")
    before = accelerator.device_free_bytes(device)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_VRAM, str(2 * GIB)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert accelerator.device_free_bytes(device) <= before - int(1.5 * GIB)
    finally:
        holder.communicate("")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_allocatable_vram_stops_where_allocations_fail_and_frees_the_probe():
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(device)
    torch.cuda.set_per_process_memory_fraction((reserved + GIB) / total, device)
    try:
        allocatable = accelerator.allocatable_bytes(device, total)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device)
    assert GIB // 2 < allocatable <= GIB
    assert torch.cuda.memory_reserved(device) == reserved


@pytest.mark.parametrize(("free", "resident"), [(40 * GIB, 4), (3 * GIB + 3 * GIB + GIB, 2)])
def test_resident_blocks_probe_only_what_all_resident_needs(free, resident):
    from jasna.ltx import sampler

    limits = []
    assert sampler.resident_block_count([GIB] * 4, lambda limit: limits.append(limit) or min(free, limit)) == resident
    assert limits == [4 * GIB + sampler.ACTIVATION_RESERVE_BYTES + 2 * GIB]


@pytest.mark.usefixtures("nvidia_vendor")
def test_used_vram_is_what_the_whole_gpu_does_not_have_free(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (30 * GIB, 32 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: 12 * GIB)
    assert accelerator.device_used_bytes(torch.device("cuda:0")) == 20 * GIB
