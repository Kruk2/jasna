import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from jasna.media import hip_kernel


def _mock_supported_host(monkeypatch, architecture="gfx1100", platform="linux"):
    monkeypatch.setattr(hip_kernel.sys, "platform", platform)
    monkeypatch.setattr(hip_kernel, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(hip_kernel.torch.version, "hip", "7.2.1")
    monkeypatch.setattr(hip_kernel.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        hip_kernel.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName=architecture),
    )


def test_colour_kernels_auto_skips_an_ineligible_device(monkeypatch):
    monkeypatch.setattr(
        hip_kernel,
        "is_amd_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("vendor queried")),
    )

    assert not hip_kernel.hip_color_kernels_enabled(
        torch.device("cpu"), environ={}
    )


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_colour_kernels_accept_supported_explicit_enable(monkeypatch, value):
    _mock_supported_host(monkeypatch, "gfx1100:sramecc+:xnack-")

    assert hip_kernel.hip_color_kernels_enabled(
        torch.device("cuda:0"),
        environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: value},
    )


def test_colour_kernels_reject_invalid_switch():
    with pytest.raises(ValueError, match=hip_kernel.AMD_HIP_COLOR_KERNELS_ENV):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "maybe"},
        )


def test_colour_kernels_fail_closed_on_unsupported_architecture(monkeypatch):
    _mock_supported_host(monkeypatch, "gfx1201")

    with pytest.raises(RuntimeError, match="gfx1201"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_colour_kernels_fail_closed_when_one_code_object_is_missing(
    monkeypatch, tmp_path
):
    _mock_supported_host(monkeypatch)
    monkeypatch.setattr(hip_kernel, "code_object_path", lambda name: tmp_path / name)
    (tmp_path / hip_kernel._REQUIRED_COLOR_CODE_OBJECTS[0]).write_bytes(b"code")

    with pytest.raises(RuntimeError, match="missing precompiled kernels"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_colour_kernels_fail_closed_off_supported_platforms(monkeypatch):
    monkeypatch.setattr(hip_kernel.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="Linux or Windows AMD/ROCm"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_colour_kernels_auto_select_installed_supported_target(monkeypatch, tmp_path):
    _mock_supported_host(monkeypatch)
    monkeypatch.setattr(hip_kernel, "code_object_path", lambda name: tmp_path / name)
    for name in hip_kernel._REQUIRED_COLOR_CODE_OBJECTS:
        (tmp_path / name).write_bytes(b"code")

    assert hip_kernel.hip_color_kernels_enabled(torch.device("cuda:0"), environ={})


def _write_windows_bundle(monkeypatch, tmp_path, *, hip_major=7):
    _mock_supported_host(monkeypatch, platform="win32")
    monkeypatch.setattr(hip_kernel, "code_object_path", lambda name: tmp_path / name)
    monkeypatch.setattr(
        hip_kernel,
        "_windows_runtime_file_identity",
        lambda: {
            "runtime_dll": "amdhip64_7.dll",
            "runtime_path": str(tmp_path / "amdhip64_7.dll"),
            "runtime_dll_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(hip_kernel, "_windows_runtime_api_version", lambda: 70200000)
    source_hashes = {}
    for name in ("yuv_to_rgb.cu", "rgb_to_yuv.cu"):
        payload = name.encode("ascii")
        (tmp_path / name).write_bytes(payload)
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
    artifact_hashes = {}
    for name in hip_kernel.required_color_code_objects("win32"):
        payload = name.encode("ascii")
        (tmp_path / name).write_bytes(payload)
        artifact_hashes[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": hip_kernel._WINDOWS_MANIFEST_SCHEMA,
        "platform": "win32",
        "architecture": "gfx1100",
        "hip_major": hip_major,
        "runtime_dll": f"amdhip64_{hip_major}.dll",
        "runtime_dll_sha256": "a" * 64,
        "runtime_api_version": 70200000,
        "torch_hip_runtime": "7.2.1",
        "code_object_abi": 4,
        "parameter_abi": hip_kernel._WINDOWS_PARAMETER_ABI,
        "sources": source_hashes,
        "artifacts": artifact_hashes,
    }
    (tmp_path / hip_kernel._WINDOWS_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _rewrite_windows_manifest(tmp_path, manifest):
    (tmp_path / hip_kernel._WINDOWS_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_windows_colour_kernels_remain_off_in_auto_mode(monkeypatch, tmp_path):
    _write_windows_bundle(monkeypatch, tmp_path)

    assert not hip_kernel.hip_color_kernels_enabled(
        torch.device("cuda:0"), environ={}
    )


def test_windows_colour_kernels_accept_valid_explicit_bundle(monkeypatch, tmp_path):
    _write_windows_bundle(monkeypatch, tmp_path)

    assert hip_kernel.hip_color_kernels_enabled(
        torch.device("cuda:0"),
        environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
    )


def test_windows_colour_kernels_reject_runtime_major_mismatch(monkeypatch, tmp_path):
    _write_windows_bundle(monkeypatch, tmp_path, hip_major=6)

    with pytest.raises(RuntimeError, match="hip_major mismatch"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_windows_colour_kernels_reject_artifact_hash_mismatch(monkeypatch, tmp_path):
    _write_windows_bundle(monkeypatch, tmp_path)
    name = hip_kernel.required_color_code_objects("win32")[0]
    (tmp_path / name).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_windows_colour_kernels_reject_runtime_api_mismatch(monkeypatch, tmp_path):
    _write_windows_bundle(monkeypatch, tmp_path)
    monkeypatch.setattr(hip_kernel, "_windows_runtime_api_version", lambda: 70200001)

    with pytest.raises(RuntimeError, match="runtime API version mismatch"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "unexpected", "schema mismatch"),
        ("platform", "linux", "platform mismatch"),
        ("architecture", "gfx1101", "architecture mismatch"),
        ("torch_hip_runtime", "7.2.2", "torch_hip_runtime mismatch"),
        ("code_object_abi", 5, "code_object_abi mismatch"),
        ("parameter_abi", "unexpected", "parameter_abi mismatch"),
        ("runtime_dll", "amdhip64_6.dll", "runtime DLL mismatch"),
        ("runtime_dll_sha256", "b" * 64, "runtime DLL SHA256 mismatch"),
    ],
)
def test_windows_colour_kernels_reject_manifest_contract_mismatch(
    monkeypatch, tmp_path, field, value, message
):
    manifest = _write_windows_bundle(monkeypatch, tmp_path)
    manifest[field] = value
    _rewrite_windows_manifest(tmp_path, manifest)

    with pytest.raises(RuntimeError, match=message):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_windows_colour_kernels_reject_missing_or_malformed_manifest(
    monkeypatch, tmp_path
):
    _write_windows_bundle(monkeypatch, tmp_path)
    manifest_path = tmp_path / hip_kernel._WINDOWS_MANIFEST
    manifest_path.unlink()

    with pytest.raises(RuntimeError, match="missing Windows HIP"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )

    manifest_path.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot read Windows HIP"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_windows_colour_kernels_reject_source_hash_mismatch(monkeypatch, tmp_path):
    _write_windows_bundle(monkeypatch, tmp_path)
    (tmp_path / "yuv_to_rgb.cu").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="SHA256 mismatch for yuv_to_rgb.cu"):
        hip_kernel.hip_color_kernels_enabled(
            torch.device("cuda:0"),
            environ={hip_kernel.AMD_HIP_COLOR_KERNELS_ENV: "1"},
        )


def test_platform_specific_code_object_names():
    assert hip_kernel.color_code_object_name("yuv_to_rgb", "linux").endswith(
        ".gfx1100.hsaco"
    )
    assert hip_kernel.color_code_object_name("yuv_to_rgb", "win32").endswith(
        ".gfx1100.windows.co"
    )
