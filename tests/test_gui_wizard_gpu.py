from __future__ import annotations

import pytest
import torch

from jasna.gui.system_checks import check_gpu


@pytest.mark.parametrize(
    ("is_available", "capability", "passed", "message_parts"),
    [
        (True, (7, 5), True, ["Fake GPU"]),
        (True, (8, 0), True, []),
        (True, (12, 1), True, []),
        (True, (7, 0), False, ["7.5", "7.0"]),
        (True, (6, 1), False, ["7.5", "6.1"]),
        (False, (7, 5), False, ["compatible GPU"]),
    ],
)
def test_check_gpu_requires_cuda_compute_7_5(
    monkeypatch, cuda_capability, is_available, capability, passed, message_parts
):
    cuda_capability(capability)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: is_available)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")

    result, message = check_gpu()

    assert result is passed
    assert all(part in message for part in message_parts)


def test_check_gpu_names_unsupported_amd_arch(monkeypatch):
    from jasna import system_requirements

    monkeypatch.setattr(system_requirements, "check_supported_gpu", lambda: (False, ("arch_unsupported", "gfx1103")))
    passed, msg = check_gpu()
    assert passed is False
    assert "gfx1103" in msg
    assert "AMD" in msg


def test_check_gpu_names_unsupported_datacenter_gpu(monkeypatch):
    from jasna import system_requirements

    monkeypatch.setattr(system_requirements, "check_supported_gpu", lambda: (False, ("datacenter_unsupported", "NVIDIA H100 80GB")))
    passed, msg = check_gpu()
    assert passed is False
    assert "NVIDIA H100 80GB" in msg
    assert "RTX" in msg
