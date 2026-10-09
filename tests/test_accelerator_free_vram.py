import subprocess
import sys

import pytest
import torch

from jasna import accelerator

GIB = 1 << 30


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
