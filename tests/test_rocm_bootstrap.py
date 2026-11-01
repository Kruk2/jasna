"""AMD startup filters unsupported adapters before PyTorch loads."""
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
        ("gfx1031\ngfx1200\n", "1"),
        ("gfx1036\ngfx1031\n", "0"),
        ("gfx1030\ngfx1201\n", "1"),
        ("gfx1201\ngfx1030\ngfx1036\n", "0"),
        ("gfx1030\ngfx1036\n", "0"),
        ("gfx1036\ngfx1151\n", "1"),
        (("gfx1201\n", 1), "0"),
        (("gfx1036\ngfx1201\n", 1), "1"),
        (("", 1), None),
        ("gfx1200\n", None),
        ("gfx1200\ngfx1201\n", None),
        ("gfx1151\n", None),
        ("gfx1036\n", None),
        ("", None),
        ("unexpected diagnostic\ngfx1200\n", None),
        (subprocess.TimeoutExpired("offload-arch", 10), None),
        (OSError("GPU probe could not start"), None),
    ],
)
def test_windows_amd_keeps_the_best_adapters_the_build_has_kernels_for(rocm_probe, output, expected):
    """#668: an R9700 next to an adapter offload-arch could not read, or next to an RDNA2 card, refused LTX."""
    rocm_probe(output, "win32")

    rocm_bootstrap.configure_rocm_devices()

    assert os.environ.get("HIP_VISIBLE_DEVICES") == expected


LINUX_CPU = "  Name: AMD Ryzen Processor\n  Uuid: CPU-XX\n  Device Type: CPU\n"
LINUX_IGPU = "  Name: gfx1036\n  Uuid: GPU-0000000000000001\n  Marketing Name: AMD Radeon Graphics\n"
LINUX_DGPU = "  Name: gfx1200\n  Uuid: GPU-0000000000000002\n  Marketing Name: AMD Radeon RX 9060 XT\n"
LINUX_OTHER_DGPU = "  Name: gfx1201\n  Uuid: GPU-0000000000000003\n  Marketing Name: AMD Radeon RX 9070 XT\n"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (LINUX_CPU + LINUX_IGPU + LINUX_DGPU, "GPU-0000000000000002"),
        (LINUX_DGPU + LINUX_IGPU, "GPU-0000000000000002"),
        (LINUX_IGPU + LINUX_DGPU + LINUX_OTHER_DGPU, "GPU-0000000000000002,GPU-0000000000000003"),
        (LINUX_IGPU + LINUX_DGPU.replace("gfx1200", "gfx1100:sramecc-:xnack-"), "GPU-0000000000000002"),
        (LINUX_IGPU + LINUX_DGPU.replace("gfx1200", "gfx1031"), "GPU-0000000000000001"),
        (LINUX_DGPU.replace("gfx1200", "gfx1030") + LINUX_OTHER_DGPU, "GPU-0000000000000003"),
        (LINUX_DGPU, None),
        (LINUX_DGPU + LINUX_OTHER_DGPU, None),
        (LINUX_IGPU, None),
        (LINUX_CPU, None),
        ("", None),
        (LINUX_IGPU + LINUX_DGPU.replace("  Uuid: GPU-0000000000000002\n", ""), None),
        (LINUX_IGPU + LINUX_DGPU.replace("GPU-0000000000000002", "GPU-XX"), None),
        (LINUX_IGPU + LINUX_DGPU.replace("GPU-0000000000000002", "GPU-0000000000000001"), None),
        (subprocess.TimeoutExpired("rocminfo", 10), None),
        (subprocess.CalledProcessError(1, "rocminfo"), None),
        (OSError("GPU probe could not start"), None),
    ],
)
def test_linux_amd_filters_by_gpu_uuid_without_using_agent_order(rocm_probe, output, expected):
    rocm_probe(output, "linux")

    rocm_bootstrap.configure_rocm_devices()

    assert os.environ.get("ROCR_VISIBLE_DEVICES") == expected
    assert "HIP_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize(
    "platform", ["linux", "win32"],
)
@pytest.mark.parametrize(
    ("frozen", "selection", "arguments"),
    [
        (False, {}, []),
        (True, {"HIP_VISIBLE_DEVICES": "1"}, []),
        (True, {"HIP_VISIBLE_DEVICES": ""}, []),
        (True, {"CUDA_VISIBLE_DEVICES": "1"}, []),
        (True, {"ROCR_VISIBLE_DEVICES": "1"}, []),
        (True, {"GPU_DEVICE_ORDINAL": "1"}, []),
        (True, {}, ["--device", "cuda:1"]),
        (True, {}, ["--device=cuda:1"]),
        (True, {}, ["--ltx-worker", "1", "2", "3", "4"]),
    ],
)
def test_explicit_device_choices_and_source_runs_are_preserved(rocm_probe, monkeypatch, platform, frozen, selection, arguments):
    rocm_probe("", platform)
    monkeypatch.setattr(rocm_bootstrap.sys, "argv", ["jasna.exe", *arguments])
    monkeypatch.setattr(rocm_bootstrap, "is_frozen", lambda: frozen)
    for name, value in selection.items():
        monkeypatch.setenv(name, value)

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Explicit GPU choices and other platforms must not probe or remap adapters")

    monkeypatch.setattr(subprocess, "run", unexpected_probe)

    rocm_bootstrap.configure_rocm_devices()

    assert {name: os.environ[name] for name in selection} == selection
    assert os.environ.get("HIP_VISIBLE_DEVICES") == selection.get("HIP_VISIBLE_DEVICES")


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_amd_build_without_kernel_packs_leaves_adapters_alone(rocm_probe, platform):
    rocm_probe(LINUX_IGPU + LINUX_DGPU if platform == "linux" else "gfx1036\ngfx1200\n", platform, kernel_packs=())

    rocm_bootstrap.configure_rocm_devices()

    assert os.environ.get("HIP_VISIBLE_DEVICES") is None
    assert os.environ.get("ROCR_VISIBLE_DEVICES") is None


def test_kernel_pack_architectures_lists_installed_device_packs(tmp_path):
    assert rocm_bootstrap.kernel_pack_architectures(tmp_path) is None
    packs = tmp_path / ".kpack"
    packs.mkdir()
    for name in ("torch_gfx1031.kpack", "torch_gfx1201.kpack", "blas_lib_gfx1100.kpack"):
        (packs / name).touch()

    assert rocm_bootstrap.kernel_pack_architectures(tmp_path) == {"gfx1031", "gfx1201"}
