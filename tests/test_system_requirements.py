import os
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from jasna import accelerator, os_utils, system_requirements

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.mark.parametrize(
    "output",
    [
        "ffmpeg version 8.0.1 Copyright (c) ...",
        "ffprobe version n8.1.2-12-gdeadbeef Copyright (c) ...",
        "ffmpeg version N-113224-gdeadbeef Copyright (c) ...\nlibavutil      60.  3.100 / 60.  3.100",
    ],
    ids=["semver", "n-prefix", "nightly-libavutil"],
)
def test_parse_ffmpeg_major_version(output) -> None:
    assert system_requirements.parse_ffmpeg_major_version(output) == 8


@pytest.mark.parametrize(
    ("version_output", "message"),
    [
        ("ffprobe version 8.1.0", None),
        ("ffprobe version 7.1.0", "major version must be exactly 8"),
        ("ffprobe version 9.0.0", "major version must be exactly 8"),
        ("ffprobe version N-113224-gdeadbeef", "could not detect major version"),
    ],
)
def test_check_required_executables_accepts_only_ffprobe_8(monkeypatch, capsys, version_output, message) -> None:
    monkeypatch.setattr(os_utils.shutil, "which", lambda exe: f"/fake/{exe}")
    monkeypatch.setattr(system_requirements.subprocess, "run", lambda cmd, **kwargs: _completed(version_output))

    with nullcontext() if message is None else pytest.raises(SystemExit, match="^1$"):
        system_requirements.check_required_executables()

    out = capsys.readouterr().out
    assert message in out if message else out == ""


def test_check_required_executables_logs_stdout_stderr_when_exe_fails(monkeypatch, caplog) -> None:
    monkeypatch.setattr(system_requirements, "find_executable", lambda exe: f"/fake/{exe}")
    monkeypatch.setattr(
        system_requirements.subprocess, "run", lambda cmd, **kwargs: _completed("ffprobe stdout", 1, "ffprobe stderr")
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        system_requirements.check_required_executables()

    assert any(
        "ffprobe failed" in rec.message and "ffprobe stdout" in rec.message and "ffprobe stderr" in rec.message
        for rec in caplog.records
    )


def _raise_oserror(setting_id):
    raise OSError("nvdrsdb0.bin not found")


@pytest.mark.parametrize(
    ("read_setting", "ok", "info"),
    [
        (lambda setting_id: system_requirements._PREFER_NO_SYSMEM_FALLBACK, True, "Prefer No Sysmem Fallback"),
        (lambda setting_id: 0, False, "Driver Default"),
        (lambda setting_id: system_requirements._PREFER_SYSMEM_FALLBACK, False, "Prefer Sysmem Fallback"),
        (lambda setting_id: None, False, "Driver Default"),
        (_raise_oserror, False, "nvdrsdb0.bin not found"),
    ],
    ids=["prefer-no-sysmem", "driver-default", "prefer-sysmem", "not-set", "unreadable"],
)
def test_check_sysmem_fallback_policy_reads_the_nvidia_setting(monkeypatch, read_setting, ok, info) -> None:
    monkeypatch.setattr(system_requirements.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(system_requirements, "_read_drs_setting", read_setting)

    result_ok, result_info = system_requirements.check_windows_nvidia_sysmem_fallback_policy()

    assert result_ok is ok
    assert info in result_info


def test_check_sysmem_fallback_skips_nvidia_drs_on_amd(monkeypatch) -> None:
    monkeypatch.setattr(system_requirements.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda: True)

    def _read_drs_setting(setting_id):
        raise AssertionError("AMD must not read the NVIDIA DRS database")

    monkeypatch.setattr(system_requirements, "_read_drs_setting", _read_drs_setting)

    assert system_requirements.check_windows_nvidia_sysmem_fallback_policy() == (True, "N/A (AMD)")


def test_check_sysmem_fallback_returns_na_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(system_requirements.sys, "platform", "linux", raising=False)

    assert system_requirements.check_windows_nvidia_sysmem_fallback_policy() == (True, "N/A")


@pytest.mark.parametrize(
    ("available", "capability", "name", "expected"),
    [
        (True, (8, 0), "RTX 4090", (True, "RTX 4090")),
        (True, (7, 5), "RTX 2070", (True, "RTX 2070")),
        (True, (6, 1), "GTX 1060", (False, ("compute_too_low", 6, 1))),
        (False, None, None, (False, "no_cuda")),
    ],
    ids=["supported", "minimum", "too-old", "no-cuda"],
)
def test_check_supported_gpu_needs_compute_capability_7_5(monkeypatch, available, capability, name, expected) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            get_device_capability=lambda device: capability,
            get_device_name=lambda device: name,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert system_requirements.check_supported_gpu() == expected


def _fake_rocm_torch(gcn_arch_name: str, arch_list: list[str], torch_dir: Path):
    def _no_kernels(*_args, **_kwargs):
        raise AssertionError("the GPU check must not launch a kernel")

    return types.SimpleNamespace(
        __file__=str(torch_dir / "__init__.py"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda device: types.SimpleNamespace(gcnArchName=gcn_arch_name),
            get_arch_list=lambda: arch_list,
            get_device_name=lambda device: "AMD Radeon 780M Graphics",
        ),
        zeros=_no_kernels,
    )


@pytest.mark.parametrize(
    ("arch", "arch_list", "kernel_packs", "expected"),
    [
        ("gfx1103", ["gfx1100", "gfx1101", "gfx1102", "gfx1201"], (), (False, ("arch_unsupported", "gfx1103"))),
        ("gfx90a:sramecc+:xnack-", ["gfx90a:xnack-", "gfx1100"], (), (True, "AMD Radeon 780M Graphics")),
        ("gfx1103", [], (), (True, "AMD Radeon 780M Graphics")),
        ("gfx1031", ["gfx1030", "gfx1031", "gfx1201"], ("gfx1030", "gfx1201"), (False, ("arch_unsupported", "gfx1031"))),
        ("gfx1031", ["gfx1030", "gfx1031", "gfx1201"], ("gfx1031",), (True, "AMD Radeon 780M Graphics")),
    ],
    ids=["missing-from-build", "feature-suffixes", "no-arch-list", "kernel-pack-missing", "kernel-pack-installed"],
)
def test_check_supported_gpu_needs_an_amd_arch_the_build_contains(
    monkeypatch, tmp_path, arch, arch_list, kernel_packs, expected,
) -> None:
    (tmp_path / ".kpack").mkdir()
    for architecture in kernel_packs:
        (tmp_path / ".kpack" / f"torch_{architecture}.kpack").touch()
    fake_torch = _fake_rocm_torch(arch, arch_list, tmp_path)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(accelerator, "torch", fake_torch)
    monkeypatch.setattr(accelerator, "vendor_for_device", lambda _device: accelerator.AcceleratorVendor.AMD)

    assert system_requirements.check_supported_gpu() == expected


@pytest.mark.parametrize(
    ("frozen", "capability", "builders", "expected"),
    [
        (True, (9, 0), ("tensorrt_libs/nvinfer_builder_resource_sm86_10.dll",),
         (False, ("datacenter_unsupported", "NVIDIA H100"))),
        (True, (8, 0), ("libnvinfer_builder_resource_sm86.so.10.16.1",), (False, ("datacenter_unsupported", "NVIDIA H100"))),
        (True, (8, 0), ("libnvinfer_builder_resource_sm80.so.10.16.1",), (True, "NVIDIA H100")),
        (True, (9, 0), ("tensorrt_libs/nvinfer_builder_resource_sm90_10.dll",), (True, "NVIDIA H100")),
        (True, (8, 6), (), (True, "NVIDIA H100")),
        (False, (9, 0), (), (True, "NVIDIA H100")),
    ],
    ids=["release-without-h100", "release-without-a100", "release-with-a100", "release-with-h100",
         "release-rtx-3090", "source-h100"],
)
def test_release_refuses_datacenter_gpus_it_cannot_build_engines_for(
    monkeypatch, tmp_path, frozen, capability, builders, expected,
) -> None:
    for name in builders:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: capability,
            get_device_name=lambda device: "NVIDIA H100",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(system_requirements, "is_frozen", lambda: frozen)
    monkeypatch.setattr(system_requirements, "get_application_executable", lambda: str(tmp_path / "jasna"))

    assert system_requirements.check_supported_gpu() == expected


def test_gpu_check_error_names_each_failure() -> None:
    assert "No compatible GPU" in system_requirements.gpu_check_error("no_cuda")
    assert "H100" in system_requirements.gpu_check_error(("datacenter_unsupported", "NVIDIA H100"))
    assert "gfx1103" in system_requirements.gpu_check_error(("arch_unsupported", "gfx1103"))
    assert "6.1" in system_requirements.gpu_check_error(("compute_too_low", 6, 1))


def _nvidia_smi_raises(cmd, **kwargs):
    raise OSError("permission denied")


@pytest.mark.parametrize(
    ("smi", "run", "ok", "info"),
    [
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed(f"{system_requirements.MIN_DRIVER_VERSION}.00\n"), True, f"{system_requirements.MIN_DRIVER_VERSION}.00"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed("611.12\n"), True, "611.12"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed(f"{system_requirements.MIN_DRIVER_VERSION - 1}.99\n"), False,
         f"{system_requirements.MIN_DRIVER_VERSION - 1}.99 (requires {system_requirements.MIN_DRIVER_VERSION}+)"),
        (None, None, False, "nvidia-smi not found"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed(returncode=1), False, "nvidia-smi exited with code 1"),
        ("/fake/nvidia-smi", _nvidia_smi_raises, False, "permission denied"),
        ("/fake/nvidia-smi", lambda cmd, **kw: _completed("garbage\n"), False, "Could not parse driver version: 'garbage'"),
    ],
    ids=["minimum", "newer", "too-old", "no-nvidia-smi", "smi-fails", "smi-unrunnable", "unparseable"],
)
def test_check_gpu_driver_version(monkeypatch, smi, run, ok, info) -> None:
    monkeypatch.setattr(system_requirements, "find_executable", lambda name: smi)
    monkeypatch.setattr(system_requirements.subprocess, "run", run)

    assert system_requirements.check_gpu_driver_version() == (ok, info)


def test_check_ascii_install_path_passes_for_ascii(monkeypatch, tmp_path) -> None:
    ascii_path = tmp_path / "jasna"
    ascii_path.mkdir()
    monkeypatch.setattr(system_requirements, "__file__", str(ascii_path / "system_requirements.py"))
    ok, info = system_requirements.check_ascii_install_path()
    assert ok is True


def test_check_ascii_install_path_fails_for_non_ascii(monkeypatch, tmp_path) -> None:
    non_ascii_path = tmp_path / "プロジェクト"
    non_ascii_path.mkdir()
    monkeypatch.setattr(system_requirements, "__file__", str(non_ascii_path / "system_requirements.py"))
    ok, info = system_requirements.check_ascii_install_path()
    assert ok is False
    assert "プロジェクト" in info


def test_check_ascii_install_path_uses_executable_when_frozen(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(system_requirements, "is_frozen", lambda: True)
    monkeypatch.setattr(system_requirements, "__file__", str(tmp_path / "プロジェクト" / "system_requirements.py"))
    monkeypatch.setattr(system_requirements.sys, "executable", str(tmp_path / "jasna.exe"))
    assert system_requirements.check_ascii_install_path() == (True, str(tmp_path))


def _sysfs_disk(root, name: str, rotational: str):
    disk = root / "devices" / name
    (disk / "queue").mkdir(parents=True)
    (disk / "queue" / "rotational").write_text(rotational + "\n")
    partition = disk / f"{name}1"
    partition.mkdir()
    (partition / "partition").write_text("1\n")
    return disk, partition


@pytest.mark.skipif(sys.platform == "win32", reason="sysfs device names like 8:1 are not valid Windows file names")
def test_partition_on_a_spinning_disk_reports_its_disk_as_rotational(tmp_path) -> None:
    _hdd, hdd_partition = _sysfs_disk(tmp_path, "sda", "1")
    ssd, _ssd_partition = _sysfs_disk(tmp_path, "nvme0n1", "0")
    links = tmp_path / "dev" / "block"
    links.mkdir(parents=True)
    (links / "8:1").symlink_to(hdd_partition)
    (links / "259:0").symlink_to(ssd)

    assert system_requirements.sysfs_block_is_rotational(links / "8:1")
    assert not system_requirements.sysfs_block_is_rotational(links / "259:0")
    assert not system_requirements.sysfs_block_is_rotational(links / "0:79")


def test_virtio_disks_are_not_reported_as_rotational_because_their_drive_type_is_unknown(tmp_path) -> None:
    pci_bus = tmp_path / "devices" / "pci0000_00"
    _virtio_blk, virtio_blk_partition = _sysfs_disk(pci_bus / "0000_00_04.0" / "virtio2", "vda", "1")
    virtio_scsi, _ = _sysfs_disk(pci_bus / "0000_00_03.0" / "virtio0" / "host0" / "target0_0_1" / "0_0_1_0", "sda", "1")
    _sata_hdd, sata_hdd_partition = _sysfs_disk(pci_bus / "0000_00_1f.2" / "ata1", "sdb", "1")

    assert not system_requirements.sysfs_block_is_rotational(virtio_blk_partition)
    assert not system_requirements.sysfs_block_is_rotational(virtio_scsi)
    assert system_requirements.sysfs_block_is_rotational(sata_hdd_partition)


def test_mount_source_is_the_innermost_mount_holding_the_path() -> None:
    mountinfo = "\n".join([
        "29 1 0:26 / / rw,relatime shared:1 - btrfs /dev/mapper/root rw,subvol=/root",
        "40 29 8:2 / /mnt/media rw,relatime shared:5 - ext4 /dev/sda2 rw",
        "41 40 0:50 / /mnt/media/old\\040disk rw,relatime shared:6 - btrfs /dev/sdb1 rw",
        "42 29 0:79 / /mnt/pool rw,relatime shared:7 - fuse.mergerfs pool rw",
    ])

    assert system_requirements.mount_source(mountinfo, Path("/mnt/media/old disk/jasna")) == "/dev/sdb1"
    assert system_requirements.mount_source(mountinfo, Path("/mnt/media/jasna")) == "/dev/sda2"
    assert system_requirements.mount_source(mountinfo, Path("/mnt/mediaX/jasna")) == "/dev/mapper/root"
    assert system_requirements.mount_source(mountinfo, Path("/mnt/pool/jasna")) == "pool"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mounts")
def test_a_mount_device_missing_in_a_container_leaves_the_btrfs_device(monkeypatch, tmp_path) -> None:
    anonymous = os.makedev(0, 79)
    real_stat = os.stat

    def stat(path, **kwargs):
        if Path(path) == tmp_path:
            return os.stat_result((0o40755, 1, anonymous, 1, 0, 0, 0, 0, 0, 0))
        return real_stat(path, **kwargs)

    monkeypatch.setattr(system_requirements.os, "stat", stat)
    monkeypatch.setattr(system_requirements, "mount_source", lambda mountinfo, path: "/dev/mapper/luks-not-in-this-container")
    assert system_requirements._linux_block_device(tmp_path) == anonymous


@pytest.mark.skipif(sys.platform != "win32", reason="Windows storage query")
def test_windows_seek_penalty_query_skips_network_paths() -> None:
    assert not system_requirements._windows_drive_has_seek_penalty(Path(r"\\server\share\jasna"))


@pytest.mark.parametrize(
    ("release", "supported"),
    [("4.4.302+", False), ("5.4.0-216-generic", False), ("5.7.19", False), ("5.8.0-63-generic", True),
     ("5.10.55+", True), ("5.15.167.4-microsoft-standard-WSL2", True), ("6.8.0-45-generic", True), ("10.0.0", True)],
)
def test_ltx_runs_on_linux_kernels_from_5_8(release, supported) -> None:
    assert system_requirements.kernel_runs_ltx(release) is supported
