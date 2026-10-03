#!/usr/bin/env python3
"""Atomically install one accepted PyAV/FFmpeg runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jasna import runtime_contract  # noqa: E402


WINDOWS_RESIDENT_SOURCE_PATHS = (
    "scripts/amf_d3d11_hip_resident.pyx",
    "scripts/native/amf_d3d11_hip_resident_native.hpp",
    "scripts/build_amf_d3d11_hip_resident.py",
)


@dataclass(frozen=True)
class InstallResult:
    target_root: Path
    backup_root: Path | None


def _find_single(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise runtime_contract.RuntimeContractError(
            f"{label} must resolve to exactly one file below {directory}; "
            f"observed {len(matches)}"
        )
    return matches[0]


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise runtime_contract.RuntimeContractError(f"{label} is missing: {path}")
    actual = runtime_contract.sha256_file(path)
    if actual != expected:
        raise runtime_contract.RuntimeContractError(
            f"{label} SHA256 mismatch: expected {expected}, observed {actual}"
        )


def windows_resident_source_sha256(repo_root: Path) -> str:
    """Hash the full native bridge source set using a stable manifest."""

    rows: list[str] = []
    for relative in WINDOWS_RESIDENT_SOURCE_PATHS:
        path = repo_root / Path(relative)
        if not path.is_file():
            raise runtime_contract.RuntimeContractError(
                f"Windows D3D11/HIP resident bridge source is missing: {path}"
            )
        rows.append(f"{relative}={runtime_contract.sha256_file(path)}")
    payload = ("\n".join(rows) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _matches_current_extension_abi(path: Path) -> bool:
    return any(
        path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _copy_contents(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise runtime_contract.RuntimeContractError(
            f"runtime source directory is missing: {source}"
        )
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)


def _extract_wheel(wheel: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise runtime_contract.RuntimeContractError(
                    f"PyAV wheel contains an unsafe path: {member.filename!r}"
                )
            extracted = (destination / member_path).resolve(strict=False)
            if not extracted.is_relative_to(destination_root):
                raise runtime_contract.RuntimeContractError(
                    f"PyAV wheel path escapes site-packages: {member.filename!r}"
                )
        archive.extractall(destination)


def _require_contained_symlinks(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(resolved_root):
            raise runtime_contract.RuntimeContractError(
                f"runtime symlink escapes the installation: {path} -> {resolved}"
            )


def _validate_target_root(
    requested_target: Path,
    *,
    build_root: Path,
    repo_root: Path,
) -> Path:
    if requested_target.is_symlink():
        raise runtime_contract.RuntimeContractError(
            f"target runtime must not be a symlink: {requested_target}"
        )
    target = requested_target.resolve(strict=False)
    filesystem_root = Path(target.anchor)
    home = Path.home().resolve()
    if target in {filesystem_root, home, build_root, repo_root}:
        raise runtime_contract.RuntimeContractError(
            f"refusing unsafe runtime target: {target}"
        )
    if build_root.is_relative_to(target) or target.is_relative_to(build_root):
        raise runtime_contract.RuntimeContractError(
            "runtime target must not contain or be contained by the build root"
        )
    if repo_root.is_relative_to(target) or target.is_relative_to(repo_root):
        raise runtime_contract.RuntimeContractError(
            "runtime target must not contain or be contained by the repository"
        )
    if home.is_relative_to(target):
        raise runtime_contract.RuntimeContractError(
            f"runtime target must not contain the user home: {target}"
        )
    return target


def _next_backup_path(target_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = target_root.with_name(f"{target_root.name}.backup-{timestamp}")
    candidate = base
    counter = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = base.with_name(f"{base.name}-{counter}")
        counter += 1
    return candidate


def install_runtime(args: argparse.Namespace) -> InstallResult:
    platform_key = runtime_contract.runtime_platform_key(args.platform)
    policy = runtime_contract.RUNTIME_POLICIES[platform_key]
    build_root = Path(args.build_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    requested_target = Path(args.target_root).expanduser()
    target_root = _validate_target_root(
        requested_target,
        build_root=build_root,
        repo_root=repo_root,
    )
    build_manifest = runtime_contract.parse_build_manifest(
        build_root / "build-manifest.txt"
    )

    expected_pins = {
        **runtime_contract.EXPECTED_SOURCE_PINS,
        **policy.extra_source_pins,
    }
    for name, expected in expected_pins.items():
        actual = build_manifest.get(name)
        if actual != expected:
            raise runtime_contract.RuntimeContractError(
                f"{name}: expected accepted pin {expected}, observed {actual!r}"
            )

    wheel = _find_single(build_root / "wheels", "av-*.whl", "PyAV wheel")
    _require_hash(wheel, policy.wheel_sha256, "PyAV wheel")

    bridge: Path | None = None
    bridge_sha256: str | None = None
    bridge_source_sha256: str | None = None
    windows_resident_bridge: Path | None = None
    windows_resident_bridge_sha256: str | None = None
    windows_resident_expected_source_sha256: str | None = None
    if platform_key == "linux-amd":
        bridge = _find_single(
            build_root / "amf-interop-bridge",
            "_jasna_amf_surface_probe.*.so",
            "AMF interop bridge",
        )
        bridge_sha256 = build_manifest.get("AMF_INTEROP_BRIDGE_SHA256")
        bridge_source_sha256 = build_manifest.get(
            "AMF_INTEROP_BRIDGE_SOURCE_SHA256"
        )
        if not bridge_sha256 or not bridge_source_sha256:
            raise runtime_contract.RuntimeContractError(
                "build manifest is missing AMF bridge hashes"
            )
        _require_hash(bridge, bridge_sha256, "AMF interop bridge")
        source = repo_root / "scripts/amf_surface_probe.pyx"
        _require_hash(source, bridge_source_sha256, "AMF interop bridge source")
        if sys.implementation.cache_tag not in bridge.name:
            raise runtime_contract.RuntimeContractError(
                "AMF interop bridge Python ABI does not match this installer: "
                f"{bridge.name} vs {sys.implementation.cache_tag}"
            )
    elif platform_key == "windows-amd" and (
        build_manifest.get("WINDOWS_D3D11_HIP_RESIDENT_BRIDGE_SHA256")
        or build_manifest.get("WINDOWS_D3D11_HIP_RESIDENT_SOURCE_SHA256")
    ):
        windows_resident_bridge = _find_single(
            build_root / "amf-d3d11-hip-resident-bridge",
            "_jasna_amf_d3d11_hip_resident.*.pyd",
            "Windows D3D11/HIP resident bridge",
        )
        windows_resident_bridge_sha256 = build_manifest.get(
            "WINDOWS_D3D11_HIP_RESIDENT_BRIDGE_SHA256"
        )
        windows_resident_expected_source_sha256 = build_manifest.get(
            "WINDOWS_D3D11_HIP_RESIDENT_SOURCE_SHA256"
        )
        if (
            not windows_resident_bridge_sha256
            or not windows_resident_expected_source_sha256
        ):
            raise runtime_contract.RuntimeContractError(
                "build manifest has incomplete Windows D3D11/HIP resident "
                "bridge hashes"
            )
        _require_hash(
            windows_resident_bridge,
            windows_resident_bridge_sha256,
            "Windows D3D11/HIP resident bridge",
        )
        observed_source_sha256 = windows_resident_source_sha256(repo_root)
        if observed_source_sha256 != windows_resident_expected_source_sha256:
            raise runtime_contract.RuntimeContractError(
                "Windows D3D11/HIP resident bridge source manifest SHA256 "
                f"mismatch: expected {windows_resident_expected_source_sha256}, "
                f"observed {observed_source_sha256}"
            )
        if not _matches_current_extension_abi(windows_resident_bridge):
            raise runtime_contract.RuntimeContractError(
                "Windows D3D11/HIP resident bridge Python ABI does not match "
                f"this installer: {windows_resident_bridge.name} vs one of "
                f"{importlib.machinery.EXTENSION_SUFFIXES}"
            )

    if (target_root.exists() or target_root.is_symlink()) and not args.force:
        raise runtime_contract.RuntimeContractError(
            f"target runtime already exists: {target_root}; use --force to replace it"
        )

    target_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.staging-",
            dir=target_root.parent,
        )
    )
    backup_root: Path | None = None
    try:
        site_packages = staging / "site-packages"
        site_packages.mkdir()
        _extract_wheel(wheel, site_packages)

        ffmpeg_install = build_root / "ffmpeg-install"
        _copy_contents(ffmpeg_install / "bin", staging / "bin")
        if platform_key == "linux-amd":
            _copy_contents(ffmpeg_install / "lib", staging / "lib")
            assert bridge is not None
            (staging / "bridge").mkdir()
            shutil.copy2(bridge, staging / "bridge" / bridge.name)
        elif windows_resident_bridge is not None:
            (staging / "bridge").mkdir()
            shutil.copy2(
                windows_resident_bridge,
                staging / "bridge" / windows_resident_bridge.name,
            )
        _require_contained_symlinks(staging)

        source_pins = {name: build_manifest[name] for name in expected_pins}
        runtime_manifest: dict[str, object] = {
            "schema_version": runtime_contract.RUNTIME_SCHEMA_VERSION,
            "platform": platform_key,
            "installed_unix_seconds": time.time(),
            "source_build_root": str(build_root),
            "repo_root": str(repo_root),
            "source_pins": source_pins,
            "wheel_filename": wheel.name,
            "wheel_sha256": runtime_contract.sha256_file(wheel),
            "product_defaults": {"batch_size": 4},
        }
        if bridge is not None:
            runtime_manifest["amf_interop_bridge"] = {
                "filename": bridge.name,
                "sha256": bridge_sha256,
                "source_sha256": bridge_source_sha256,
            }
        if windows_resident_bridge is not None:
            runtime_manifest["windows_d3d11_hip_resident_bridge"] = {
                "filename": windows_resident_bridge.name,
                "sha256": windows_resident_bridge_sha256,
                "source_sha256": windows_resident_expected_source_sha256,
                "api_version": (
                    runtime_contract.WINDOWS_D3D11_HIP_RESIDENT_API_VERSION
                ),
            }
        (staging / "runtime.json").write_text(
            json.dumps(runtime_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_contract.validate_runtime_layout(staging, platform=args.platform)
        runtime_contract.validate_ffmpeg_capabilities(
            staging,
            platform=args.platform,
        )

        if target_root.exists():
            backup_root = _next_backup_path(target_root)
            target_root.replace(backup_root)
        try:
            staging.replace(target_root)
        except Exception:
            if backup_root is not None and not target_root.exists():
                backup_root.replace(target_root)
            raise
        return InstallResult(target_root=target_root, backup_root=backup_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--target-root",
        default=str(runtime_contract.default_runtime_root()),
    )
    parser.add_argument("--platform", default=sys.platform)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    try:
        result = install_runtime(build_parser().parse_args())
    except (
        OSError,
        KeyError,
        ValueError,
        runtime_contract.RuntimeContractError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Unified runtime installation failed: {exc}", file=sys.stderr)
        return 1
    print(result.target_root)
    if result.backup_root is not None:
        print(f"Previous runtime preserved at: {result.backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
