"""ROCm-only environment defaults applied at process entry points."""
import os
from pathlib import Path

import pytest
import torch

from jasna import accelerator
from jasna.accelerator import apply_rocm_env_defaults, configure_rocm_process_env, has_aotriton_flash_images

ROCM_DEFAULTS = {
    "MIOPEN_FIND_MODE": "HYBRID",
    "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:False",
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


@pytest.fixture
def sdpa_switches(monkeypatch) -> dict[str, bool]:
    switches: dict[str, bool] = {}
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", lambda on: switches.__setitem__("flash", on))
    monkeypatch.setattr(torch.backends.cuda, "enable_mem_efficient_sdp", lambda on: switches.__setitem__("mem_efficient", on))
    return switches


def _torch_lib(root: Path, *image_files: str) -> Path:
    lib = root / "lib"
    for name in image_files:
        (lib / "aotriton.images" / name).parent.mkdir(parents=True, exist_ok=True)
        (lib / "aotriton.images" / name).write_bytes(b"")
    lib.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    return lib


def test_aotriton_flash_images_need_the_forward_kernel(tmp_path):
    assert not has_aotriton_flash_images(_torch_lib(tmp_path / "missing"))
    assert not has_aotriton_flash_images(_torch_lib(tmp_path / "partial", "amd-gfx120x/__signature__"))
    assert has_aotriton_flash_images(_torch_lib(tmp_path / "installed", "amd-gfx120x/flash/attn_fwd.zip"))


@pytest.mark.parametrize(("image_files", "expected"), [((), {"flash": False, "mem_efficient": False}), (("amd-gfx110x/flash/attn_fwd.zip",), {})])
def test_rocm_keeps_flash_sdpa_only_with_aotriton_images(monkeypatch, tmp_path, sdpa_switches, image_files, expected):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    _torch_lib(tmp_path, *image_files)
    monkeypatch.setattr(accelerator.torch, "__file__", str(tmp_path / "__init__.py"))
    configure_rocm_process_env()
    assert sdpa_switches == expected


def test_process_env_gets_rocm_defaults_on_rocm(monkeypatch, sdpa_switches):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    for name in ROCM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    configure_rocm_process_env()
    assert {name: os.environ[name] for name in ROCM_DEFAULTS} == ROCM_DEFAULTS
