"""AOTriton SDPA availability, tested without a GPU.

Windows ROCm splits the AOTriton payload across two device packages, and with
only the exact-arch one installed the images are absent: the first FLASH
attention launch then dies with hipErrorInvalidImage (ROCm/TheRock#7315). These
tests pin down which image directory is accepted, and that the SDPA backends are
only switched off when they genuinely cannot run — switching them off
unconditionally costs ~2x on attention-bound models such as RF-DETR's DINOv2
backbone.
"""

from __future__ import annotations

import pytest
import torch

from jasna import accelerator


def make_images(lib_dir, name):
    """Create a minimal but complete image set and return its directory."""
    flash = lib_dir / "aotriton.images" / name / "flash"
    flash.mkdir(parents=True, exist_ok=True)
    (flash / "attn_fwd.zip").write_bytes(b"kernel")
    return flash.parent


def test_no_images_directory(tmp_path):
    assert accelerator.aotriton_image_dir(tmp_path) is None


def test_directory_without_the_flash_kernel_is_ignored(tmp_path):
    (tmp_path / "aotriton.images" / "amd-gfx110x" / "flash").mkdir(parents=True)
    assert accelerator.aotriton_image_dir(tmp_path) is None


def test_single_set_is_used_without_touching_the_device(tmp_path, monkeypatch):
    expected = make_images(tmp_path, "amd-gfx110x")

    def explode():  # pragma: no cover - must not run
        raise AssertionError("a single image set must not trigger a device query")

    monkeypatch.setattr(accelerator, "_device_gcn_arch", explode)
    assert accelerator.aotriton_image_dir(tmp_path) == expected


def test_family_directory_matches_the_architecture(tmp_path):
    make_images(tmp_path, "amd-gfx110x")
    make_images(tmp_path, "amd-gfx120x")

    assert accelerator.aotriton_image_dir(tmp_path, arch="gfx1102").name == "amd-gfx110x"
    assert accelerator.aotriton_image_dir(tmp_path, arch="gfx1201").name == "amd-gfx120x"


def test_exact_architecture_wins_over_the_family(tmp_path):
    make_images(tmp_path, "amd-gfx1100")
    make_images(tmp_path, "amd-gfx110x")

    assert accelerator.aotriton_image_dir(tmp_path, arch="gfx1100").name == "amd-gfx1100"


def test_architecture_without_images_keeps_the_math_backend(tmp_path):
    make_images(tmp_path, "amd-gfx110x")
    make_images(tmp_path, "amd-gfx120x")

    # RDNA2 has no AOTriton support upstream: images exist, but not for this GPU.
    assert accelerator.aotriton_image_dir(tmp_path, arch="gfx1030") is None


def test_several_sets_with_unknown_arch_stay_conservative(tmp_path):
    make_images(tmp_path, "amd-gfx110x")
    make_images(tmp_path, "amd-gfx120x")

    assert accelerator.aotriton_image_dir(tmp_path, arch="") is None


class _Recorder:
    def __init__(self):
        self.calls: list[bool] = []

    def __call__(self, enabled):
        self.calls.append(bool(enabled))


@pytest.fixture
def rocm_process(monkeypatch):
    """A ROCm build whose SDPA switches and env defaults are observable."""
    monkeypatch.setattr(torch.version, "hip", "7.16.0")
    monkeypatch.setattr(accelerator, "apply_rocm_env_defaults", lambda environ: None)
    monkeypatch.delenv(accelerator._FORCE_MATH_ENV, raising=False)
    flash, mem_efficient = _Recorder(), _Recorder()
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", flash)
    monkeypatch.setattr(torch.backends.cuda, "enable_mem_efficient_sdp", mem_efficient)
    return flash, mem_efficient


def test_images_present_keeps_the_fast_backends(rocm_process, monkeypatch, tmp_path):
    flash, mem_efficient = rocm_process
    monkeypatch.setattr(accelerator, "aotriton_image_dir", lambda *a, **k: tmp_path)

    accelerator.configure_rocm_process_env()

    assert flash.calls == []
    assert mem_efficient.calls == []


def test_images_missing_falls_back_to_math(rocm_process, monkeypatch):
    flash, mem_efficient = rocm_process
    monkeypatch.setattr(accelerator, "aotriton_image_dir", lambda *a, **k: None)

    accelerator.configure_rocm_process_env()

    assert flash.calls == [False]
    assert mem_efficient.calls == [False]


def test_force_env_overrides_present_images(rocm_process, monkeypatch, tmp_path):
    flash, mem_efficient = rocm_process
    monkeypatch.setattr(accelerator, "aotriton_image_dir", lambda *a, **k: tmp_path)
    monkeypatch.setenv(accelerator._FORCE_MATH_ENV, "1")

    accelerator.configure_rocm_process_env()

    assert flash.calls == [False]
    assert mem_efficient.calls == [False]


def test_non_rocm_build_is_left_alone(rocm_process, monkeypatch):
    flash, mem_efficient = rocm_process
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(accelerator, "aotriton_image_dir", lambda *a, **k: None)

    accelerator.configure_rocm_process_env()

    assert flash.calls == []
    assert mem_efficient.calls == []
