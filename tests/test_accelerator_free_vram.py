import subprocess
import sys

import pytest
import torch

from jasna import accelerator

GIB = 1 << 30


def test_windows_caps_the_process_view_with_the_device_wide_free_vram(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (30 * GIB, 32 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: 12 * GIB)
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == 12 * GIB


def test_linux_reads_cuda_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (11 * GIB, 32 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: pytest.fail("NVML read on Linux"))
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == 11 * GIB


def test_amd_windows_reads_hip_without_nvml(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.15.26333")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * GIB, 32 * GIB))
    monkeypatch.setattr(accelerator, "_nvml_free_bytes", lambda device: pytest.fail("NVML read on AMD"))
    assert accelerator.device_free_bytes(torch.device("cuda:0")) == 20 * GIB


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
def test_resident_blocks_probe_only_what_all_resident_needs(monkeypatch, free, resident):
    from types import SimpleNamespace

    from jasna.ltx import sampler

    limits = []
    monkeypatch.setattr(sampler, "allocatable_bytes", lambda device, limit: limits.append(limit) or limit)
    owner = SimpleNamespace(device=torch.device("cuda:0"), _free_vram=lambda device: free)
    assert sampler.LtxTransformer._resident_blocks(owner, [GIB] * 4) == resident
    assert limits == [min(free, 4 * GIB + sampler.ACTIVATION_RESERVE_BYTES + 2 * GIB)]
