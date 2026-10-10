"""ROCm-only environment defaults applied at process entry points."""
import os

import torch

from jasna.accelerator import apply_rocm_env_defaults, configure_rocm_process_env

ROCM_DEFAULTS = {
    "MIOPEN_FIND_MODE": "FAST",
    "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:False",
    # Opt-in for the AOTriton flash-attention kernels: `configure_rocm_process_env`
    # keeps FLASH SDPA enabled when this GPU's kernel images are installed (2x on
    # attention-bound models) and only falls back to MATH when they are missing.
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
}


def test_rocm_pins_the_allocator_and_miopen_find_mode():
    environ: dict[str, str] = {}
    apply_rocm_env_defaults(environ)
    assert environ == ROCM_DEFAULTS


def test_a_user_setting_wins():
    environ = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    apply_rocm_env_defaults(environ)
    assert environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert environ["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:False"


def test_process_env_is_left_alone_off_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    for name in ROCM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    configure_rocm_process_env()
    assert not any(name in os.environ for name in ROCM_DEFAULTS)


def test_process_env_gets_rocm_defaults_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    for name in ROCM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    configure_rocm_process_env()
    assert {name: os.environ[name] for name in ROCM_DEFAULTS} == ROCM_DEFAULTS
