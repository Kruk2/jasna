"""ROCm-only environment defaults applied at process entry points."""
import os
from pathlib import Path

import pytest
import torch

from jasna import accelerator
from jasna.accelerator import (
    apply_rocm_env_defaults,
    configure_rocm_process_env,
    has_aotriton_flash_images,
    windows_miopen_system_db,
)

ROCM_DEFAULTS = {
    "MIOPEN_FIND_MODE": "HYBRID",
    "MIOPEN_DEBUG_CONV_GEMM": "0",
    "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:False",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
}


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_rocm_pins_the_allocator_and_miopen_find_mode(monkeypatch, platform):
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    environ: dict[str, str] = {}
    apply_rocm_env_defaults(environ)
    assert environ == ROCM_DEFAULTS


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize(("name", "value"), [
    ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0"),
    ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
    ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
    ("MIOPEN_DEBUG_CONV_GEMM", "1"),
])
def test_a_user_setting_wins(monkeypatch, platform, name, value):
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    environ = {name: value}
    apply_rocm_env_defaults(environ)
    assert environ == {**ROCM_DEFAULTS, name: value}


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


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize(("image_files", "expected"), [((), {"flash": False, "mem_efficient": False}), (("amd-gfx120x/flash/attn_fwd.zip",), {})])
def test_rocm_keeps_flash_sdpa_only_with_aotriton_images(monkeypatch, tmp_path, sdpa_switches, platform, image_files, expected):
    monkeypatch.setattr(accelerator, "has_triton", lambda: False)
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    _torch_lib(tmp_path, *image_files)
    monkeypatch.setattr(accelerator.torch, "__file__", str(tmp_path / "__init__.py"))
    configure_rocm_process_env()
    assert sdpa_switches == expected


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_process_env_gets_rocm_defaults_on_rocm(monkeypatch, sdpa_switches, platform):
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    for name in ROCM_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    configure_rocm_process_env()
    assert {name: os.environ.get(name) for name in ROCM_DEFAULTS} == ROCM_DEFAULTS


@pytest.mark.parametrize(("platform", "shipped", "expected"), [
    ("win32", True, True), ("win32", False, False), ("linux", True, False),
])
def test_windows_miopen_reads_the_wheels_system_databases(tmp_path, platform, shipped, expected):
    database = tmp_path / "share" / "miopen" / "db"
    if shipped:
        database.mkdir(parents=True)
    assert windows_miopen_system_db(platform, tmp_path) == (database if expected else None)


def test_windows_miopen_system_database_needs_the_rocm_libraries():
    assert windows_miopen_system_db("win32", None) is None


@pytest.mark.parametrize(("platform", "expected"), [("win32", True), ("linux", False)])
def test_process_env_points_windows_miopen_at_the_system_databases(monkeypatch, tmp_path, sdpa_switches, platform, expected):
    (tmp_path / "share" / "miopen" / "db").mkdir(parents=True)
    monkeypatch.setattr(accelerator.sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(accelerator, "rocm_libraries_dir", lambda: tmp_path)
    monkeypatch.setenv("MIOPEN_SYSTEM_DB_PATH", "restored after the test")
    monkeypatch.delenv("MIOPEN_SYSTEM_DB_PATH")
    configure_rocm_process_env()
    assert os.environ.get("MIOPEN_SYSTEM_DB_PATH") == (str(tmp_path / "share" / "miopen" / "db") if expected else None)


def test_a_user_miopen_system_database_wins(monkeypatch, tmp_path, sdpa_switches):
    (tmp_path / "share" / "miopen" / "db").mkdir(parents=True)
    monkeypatch.setattr(accelerator.sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(accelerator, "rocm_libraries_dir", lambda: tmp_path)
    monkeypatch.setenv("MIOPEN_SYSTEM_DB_PATH", "D:/own")
    configure_rocm_process_env()
    assert os.environ["MIOPEN_SYSTEM_DB_PATH"] == "D:/own"


@pytest.mark.parametrize("triton", [True, False])
def test_miopen_runs_channels_last_convs_natively_with_or_without_triton(monkeypatch, sdpa_switches, triton):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setenv("PYTORCH_MIOPEN_SUGGEST_NHWC", "restored after the test")
    monkeypatch.delenv("PYTORCH_MIOPEN_SUGGEST_NHWC")
    monkeypatch.setattr(accelerator, "has_triton", lambda: triton)
    configure_rocm_process_env()
    assert os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC") == "1"


def test_process_env_is_left_alone_off_rocm_including_the_channels_last_convs(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delenv("PYTORCH_MIOPEN_SUGGEST_NHWC", raising=False)
    configure_rocm_process_env()
    assert "PYTORCH_MIOPEN_SUGGEST_NHWC" not in os.environ


@pytest.mark.parametrize(
    ("image_files", "amf", "level"),
    [
        (("amd-gfx120x/flash/attn_fwd.zip",), True, "INFO"),
        ((), True, "WARNING"),
        (("amd-gfx120x/flash/attn_fwd.zip",), False, "WARNING"),
    ],
)
def test_amd_backend_line_warns_when_a_runtime_is_missing(monkeypatch, tmp_path, caplog, image_files, amf, level):
    from jasna import os_utils

    _torch_lib(tmp_path, *image_files)
    monkeypatch.setattr(accelerator.torch, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(os_utils, "check_amd_amf_runtime", lambda: (amf, ""))
    monkeypatch.setattr(accelerator, "has_triton", lambda: False)
    monkeypatch.setattr(accelerator, "amd_gpu_architecture", lambda device: "gfx1201")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "AMD Radeon AI PRO R9700")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setenv("MIOPEN_FIND_MODE", "HYBRID")
    accelerator.log_amd_backend.cache_clear()

    with caplog.at_level("INFO", logger="jasna.accelerator"):
        accelerator.log_amd_backend(torch.device("cuda:0"))
        accelerator.log_amd_backend(torch.device("cuda:0"))

    [record] = caplog.records
    assert record.levelname == level
    assert "R9700 (gfx1201)" in record.message
    assert "prebuilt kernels" in record.message
    assert "MIOPEN_FIND_MODE=HYBRID" in record.message
