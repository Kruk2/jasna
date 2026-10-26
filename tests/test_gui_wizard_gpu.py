from __future__ import annotations

import sys
import types

import pytest

from jasna.gui.system_checks import check_gpu


def _fake_torch(is_available: bool, capability: tuple[int, int]):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: is_available,
            get_device_capability=lambda device=0: capability,
            get_device_name=lambda device=0: "Fake GPU",
        )
    )


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
def test_check_gpu_requires_cuda_compute_7_5(monkeypatch, is_available, capability, passed, message_parts):
    import jasna.accelerator as accelerator

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(is_available, capability))
    monkeypatch.setattr(accelerator, "vendor_for_device", lambda _device: accelerator.AcceleratorVendor.NVIDIA)

    result, message = check_gpu()

    assert result is passed
    assert all(part in message for part in message_parts)


def test_check_gpu_names_unsupported_amd_arch(monkeypatch):
    from jasna import os_utils

    monkeypatch.setattr(os_utils, "check_supported_gpu", lambda: (False, ("arch_unsupported", "gfx1103")))
    passed, msg = check_gpu()
    assert passed is False
    assert "gfx1103" in msg
    assert "AMD" in msg
