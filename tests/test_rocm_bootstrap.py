"""Windows AMD startup filters unsupported adapters before PyTorch loads."""
import os
import subprocess

import pytest

from jasna import rocm_bootstrap


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("gfx1036\ngfx1200\n", "1"),
        ("gfx1200\ngfx1036\n", "0"),
        ("gfx1036\ngfx1200\ngfx1201\n", "1,2"),
        ("gfx1036\ngfx1100:sramecc-:xnack-\n", "1"),
        ("gfx1200\n", None),
        ("gfx1200\ngfx1201\n", None),
        ("gfx1151\n", None),
        ("gfx1036\n", None),
        ("", None),
        ("unexpected diagnostic\ngfx1200\n", None),
        (subprocess.TimeoutExpired("offload-arch", 10), None),
        (subprocess.CalledProcessError(1, "offload-arch"), None),
        (OSError("GPU probe could not start"), None),
    ],
)
def test_windows_amd_filters_only_mixed_supported_and_unsupported_adapters(monkeypatch, tmp_path, output, expected):
    tool = tmp_path / "_rocm_sdk_core" / "lib" / "llvm" / "bin" / "offload-arch.exe"
    tool.parent.mkdir(parents=True)
    tool.touch()
    monkeypatch.setattr(rocm_bootstrap.sys, "platform", "win32")
    monkeypatch.setattr(rocm_bootstrap.sys, "executable", str(tmp_path / "jasna.exe"))
    monkeypatch.setattr(rocm_bootstrap.sys, "argv", ["jasna.exe"])
    monkeypatch.setattr(rocm_bootstrap, "is_frozen", lambda: True)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    def probe(*args, **kwargs):
        if isinstance(output, Exception):
            raise output
        return subprocess.CompletedProcess(args[0], 0, output)

    monkeypatch.setattr(subprocess, "run", probe)
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        monkeypatch.delenv(name, raising=False)

    rocm_bootstrap.configure_windows_rocm_devices()

    assert os.environ.get("HIP_VISIBLE_DEVICES") == expected


@pytest.mark.parametrize(
    ("platform", "frozen", "selection", "arguments"),
    [
        ("linux", True, {}, []),
        ("win32", False, {}, []),
        ("win32", True, {"HIP_VISIBLE_DEVICES": "1"}, []),
        ("win32", True, {"HIP_VISIBLE_DEVICES": ""}, []),
        ("win32", True, {"CUDA_VISIBLE_DEVICES": "1"}, []),
        ("win32", True, {"ROCR_VISIBLE_DEVICES": "1"}, []),
        ("win32", True, {"GPU_DEVICE_ORDINAL": "1"}, []),
        ("win32", True, {}, ["--device", "cuda:1"]),
        ("win32", True, {}, ["--device=cuda:1"]),
    ],
)
def test_other_platforms_and_explicit_device_choices_are_preserved(monkeypatch, tmp_path, platform, frozen, selection, arguments):
    tool = tmp_path / "_rocm_sdk_core" / "lib" / "llvm" / "bin" / "offload-arch.exe"
    tool.parent.mkdir(parents=True)
    tool.touch()
    monkeypatch.setattr(rocm_bootstrap.sys, "platform", platform)
    monkeypatch.setattr(rocm_bootstrap.sys, "executable", str(tmp_path / "jasna.exe"))
    monkeypatch.setattr(rocm_bootstrap.sys, "argv", ["jasna.exe", *arguments])
    monkeypatch.setattr(rocm_bootstrap, "is_frozen", lambda: frozen)
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in selection.items():
        monkeypatch.setenv(name, value)

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Explicit GPU choices and other platforms must not probe or remap adapters")

    monkeypatch.setattr(subprocess, "run", unexpected_probe)

    rocm_bootstrap.configure_windows_rocm_devices()

    assert {name: os.environ[name] for name in selection} == selection
    assert os.environ.get("HIP_VISIBLE_DEVICES") == selection.get("HIP_VISIBLE_DEVICES")
