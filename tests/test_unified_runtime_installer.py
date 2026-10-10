from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import sys
import zipfile
from pathlib import Path

import pytest

from jasna import runtime_contract
from scripts import install_unified_runtime as installer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str = "linux",
) -> tuple[Path, Path, argparse.Namespace]:
    platform_key = runtime_contract.runtime_platform_key(platform)
    build_root = tmp_path / "build"
    repo_root = tmp_path / "repo"
    target_root = tmp_path / "installed/runtime"
    wheel_dir = build_root / "wheels"
    bin_dir = build_root / "ffmpeg-install/bin"
    wheel_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    repo_root.mkdir()
    bridge_source = repo_root / "scripts/amf_surface_probe.pyx"
    bridge_source.parent.mkdir()
    bridge_source.write_bytes(b"accepted-amf-bridge-source")

    wheel = wheel_dir / "av-18.1.0-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("av/__init__.py", "__version__ = '18.1.0'\n")
        archive.writestr("av-18.1.0.dist-info/METADATA", "Name: av\n")

    if platform_key == "linux-amd":
        executable = bin_dir / "ffmpeg"
        library = build_root / "ffmpeg-install/lib/libavcodec.so"
        library.parent.mkdir()
        library_directory = "lib"
        bridge = (
            build_root
            / "amf-interop-bridge"
            / f"_jasna_amf_surface_probe.{sys.implementation.cache_tag}-test.so"
        )
        bridge.parent.mkdir()
        bridge.write_bytes(b"accepted-amf-bridge")
    else:
        executable = bin_dir / "ffmpeg.exe"
        library = bin_dir / "avcodec-62.dll"
        library_directory = "bin"
    executable.write_bytes(b"accepted-ffmpeg")
    library.write_bytes(b"accepted-libavcodec")

    policy = runtime_contract.RuntimePolicy(
        wheel_sha256=_sha256(wheel),
        executables={executable.name: _sha256(executable)},
        libraries={library.name: _sha256(library)},
        library_directory=library_directory,
    )
    monkeypatch.setitem(runtime_contract.RUNTIME_POLICIES, platform_key, policy)

    pins = dict(runtime_contract.EXPECTED_SOURCE_PINS)
    manifest_values = dict(pins)
    if platform_key == "linux-amd":
        manifest_values.update(
            {
                "AMF_INTEROP_BRIDGE_SHA256": _sha256(bridge),
                "AMF_INTEROP_BRIDGE_SOURCE_SHA256": _sha256(bridge_source),
            }
        )
    (build_root / "build-manifest.txt").write_text(
        "".join(f"{name}={value}\n" for name, value in manifest_values.items()),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        build_root=str(build_root),
        repo_root=str(repo_root),
        target_root=str(target_root),
        platform=platform,
        force=False,
    )
    return build_root, target_root, args


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_install_runtime_builds_a_valid_atomic_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    _build_root, target_root, args = _fixture(
        tmp_path,
        monkeypatch,
        platform=platform,
    )

    result = installer.install_runtime(args)

    assert result.target_root == target_root
    assert result.backup_root is None
    manifest = runtime_contract.validate_runtime_layout(target_root, platform=platform)
    assert manifest["product_defaults"] == {"batch_size": 4}
    if platform == "linux":
        bridge = manifest["amf_interop_bridge"]
        assert (target_root / "bridge" / bridge["filename"]).is_file()
    else:
        assert "amf_interop_bridge" not in manifest
    assert "migraphx_manifest" not in manifest
    assert not list(target_root.parent.glob(f".{target_root.name}.staging-*"))


def test_windows_install_stages_optional_resident_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is a Windows layout/ABI contract even when the test host is Linux.
    monkeypatch.setattr(importlib.machinery, "EXTENSION_SUFFIXES", [".cp312-win_amd64.pyd"])
    build_root, target_root, args = _fixture(
        tmp_path,
        monkeypatch,
        platform="win32",
    )
    repo_root = Path(args.repo_root)
    for index, relative in enumerate(installer.WINDOWS_RESIDENT_SOURCE_PATHS):
        source = repo_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"accepted-resident-source-{index}".encode())
    bridge = (
        build_root
        / "amf-d3d11-hip-resident-bridge"
        / f"_jasna_amf_d3d11_hip_resident{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    bridge.parent.mkdir()
    bridge.write_bytes(b"accepted-resident-bridge")
    with (build_root / "build-manifest.txt").open("a", encoding="utf-8") as manifest:
        manifest.write(
            "WINDOWS_D3D11_HIP_RESIDENT_BRIDGE_SHA256="
            f"{_sha256(bridge)}\n"
        )
        manifest.write(
            "WINDOWS_D3D11_HIP_RESIDENT_SOURCE_SHA256="
            f"{installer.windows_resident_source_sha256(repo_root)}\n"
        )

    installer.install_runtime(args)

    manifest = runtime_contract.validate_runtime_layout(
        target_root,
        platform="win32",
    )
    record = manifest["windows_d3d11_hip_resident_bridge"]
    assert record["api_version"] == 1
    assert (target_root / "bridge" / record["filename"]).read_bytes() == (
        b"accepted-resident-bridge"
    )


def test_existing_target_requires_force_and_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_root, target_root, args = _fixture(tmp_path, monkeypatch)
    target_root.mkdir(parents=True)
    marker = target_root / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(runtime_contract.RuntimeContractError, match="already exists"):
        installer.install_runtime(args)

    assert marker.read_text(encoding="utf-8") == "original"


def test_force_replacement_preserves_previous_runtime_as_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_root, target_root, args = _fixture(tmp_path, monkeypatch)
    first = installer.install_runtime(args)
    marker = first.target_root / "previous-only.txt"
    marker.write_text("preserved", encoding="utf-8")
    args.force = True

    second = installer.install_runtime(args)

    assert second.backup_root is not None
    assert second.backup_root.is_dir()
    assert (second.backup_root / "previous-only.txt").read_text(encoding="utf-8") == "preserved"
    assert not (second.target_root / "previous-only.txt").exists()
    runtime_contract.validate_runtime_layout(second.target_root, platform="linux")


def test_failed_publish_restores_previous_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_root, target_root, args = _fixture(tmp_path, monkeypatch)
    installer.install_runtime(args)
    marker = target_root / "previous-only.txt"
    marker.write_text("preserved", encoding="utf-8")
    args.force = True
    original_replace = Path.replace

    def fail_staging_publish(self: Path, target: Path) -> Path:
        if self.name.startswith(f".{target_root.name}.staging-"):
            raise OSError("simulated publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_staging_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        installer.install_runtime(args)

    assert marker.read_text(encoding="utf-8") == "preserved"
    assert not list(target_root.parent.glob(f".{target_root.name}.staging-*"))


def test_wrong_source_pin_fails_before_target_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root, target_root, args = _fixture(tmp_path, monkeypatch)
    (build_root / "build-manifest.txt").write_text(
        "FFMPEG_COMMIT=wrong\n",
        encoding="utf-8",
    )

    with pytest.raises(runtime_contract.RuntimeContractError, match="FFMPEG_COMMIT"):
        installer.install_runtime(args)

    assert not target_root.exists()


def test_wheel_path_traversal_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root, target_root, args = _fixture(tmp_path, monkeypatch)
    wheel = next((build_root / "wheels").glob("*.whl"))
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("av/__init__.py", "")
        archive.writestr("../escaped.py", "bad")
    old_policy = runtime_contract.RUNTIME_POLICIES["linux-amd"]
    monkeypatch.setitem(
        runtime_contract.RUNTIME_POLICIES,
        "linux-amd",
        runtime_contract.RuntimePolicy(
            wheel_sha256=_sha256(wheel),
            executables=old_policy.executables,
            libraries=old_policy.libraries,
            library_directory=old_policy.library_directory,
        ),
    )

    with pytest.raises(runtime_contract.RuntimeContractError, match="unsafe path"):
        installer.install_runtime(args)

    assert not target_root.exists()
    assert not (target_root.parent / "escaped.py").exists()


def test_external_runtime_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root, target_root, args = _fixture(tmp_path, monkeypatch)
    executable = build_root / "ffmpeg-install/bin/ffmpeg"
    executable.unlink()
    external = tmp_path / "external-ffmpeg"
    external.write_bytes(b"accepted-ffmpeg")
    try:
        executable.symlink_to(external)
    except OSError as exc:
        if sys.platform == "win32" and exc.winerror == 1314:
            pytest.skip("Windows file-symlink privilege is unavailable")
        raise

    with pytest.raises(runtime_contract.RuntimeContractError, match="symlink escapes"):
        installer.install_runtime(args)

    assert not target_root.exists()


@pytest.mark.parametrize("unsafe", ["repo", "build"])
def test_repository_and_build_roots_are_rejected_as_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe: str,
) -> None:
    build_root, _target_root, args = _fixture(tmp_path, monkeypatch)
    args.target_root = args.repo_root if unsafe == "repo" else str(build_root)

    with pytest.raises(runtime_contract.RuntimeContractError, match="unsafe runtime target"):
        installer.install_runtime(args)
