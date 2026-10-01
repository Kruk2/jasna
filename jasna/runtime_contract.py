"""Validation helpers for Jasna's pinned PyAV/FFmpeg runtime.

PyAV and the FFmpeg libraries it loads form one native ABI unit.  A regular
development environment may still provide the rest of Jasna's Python
dependencies, but an explicitly selected unified runtime must not silently mix
its PyAV wheel, command-line tools, or shared libraries with ambient copies.

This module is deliberately independent from decoder and encoder routing.  It
only validates and prepares a runtime; callers opt in through the launchers in
``scripts/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


RUNTIME_SCHEMA_VERSION = 1
EXPECTED_SOURCE_PINS: Mapping[str, str] = {
    "FFMPEG_COMMIT": "44d082edc87381d978e8588b148116b99fefdb43",
    "PYAV_COMMIT": "7e3d950a8b72062502c1a60d672f8ca565313af5",
    "AMF_COMMIT": "c35f613aea2e5057a688c979e75b1cf24253297e",
}
EXPECTED_AV_VERSION = "18.1.0"
EXPECTED_AV_LIBRARY_VERSIONS: Mapping[str, tuple[int, int, int]] = {
    "libavutil": (60, 33, 100),
    "libavcodec": (62, 36, 101),
    "libavformat": (62, 19, 101),
    "libavdevice": (62, 4, 100),
    "libavfilter": (11, 17, 100),
    "libswscale": (9, 8, 100),
    "libswresample": (6, 4, 100),
}
AMF_INTEROP_BRIDGE_PREFIX = "_jasna_amf_surface_probe."
WINDOWS_D3D11_HIP_RESIDENT_BRIDGE_PREFIX = "_jasna_amf_d3d11_hip_resident."
WINDOWS_D3D11_HIP_RESIDENT_API_VERSION = 1
_RUNTIME_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


@dataclass(frozen=True)
class RuntimePolicy:
    """Immutable expected contents for one accepted runtime build."""

    wheel_sha256: str
    executables: Mapping[str, str]
    libraries: Mapping[str, str]
    library_directory: str
    extra_source_pins: Mapping[str, str] = field(default_factory=dict)
    required_ffmpeg_help: Mapping[str, str] = field(default_factory=dict)


RUNTIME_POLICIES: dict[str, RuntimePolicy] = {
    "linux-amd": RuntimePolicy(
        wheel_sha256=(
            "7f918c588d41aea971b0965230609094a34e2af7aaafe9a54f91089a06ddb265"
        ),
        executables={
            "ffmpeg": "767f87cbd29cfdc7885b88a6baabbeb3094d02bd32f0544402ab4f02e3f1ef83",
            "ffprobe": "eca67c1264ac91644c475808d122b6df6db5a5a34df437e4e2fb155d37ecb3f1",
        },
        libraries={
            "libavcodec.so.62.36.101": (
                "63ace691082ef3ce7efaa6529a4117cf4a5fff705d0da4ba85681a654ff97058"
            ),
            "libavdevice.so.62.4.100": (
                "3d3a94854e5e749773178ae76353191f44a6ed16c973a4aae7b143065789d7b3"
            ),
            "libavfilter.so.11.17.100": (
                "457b9f3aed0e41a4d257057ca41eb3df21e8feaf1c2b5bf01cebe16157937bb3"
            ),
            "libavformat.so.62.19.101": (
                "7c6b782c8b7197be7c90b17f384924f78eeeb654a10ecc66349d7da334dc7e06"
            ),
            "libavutil.so.60.33.100": (
                "e735a2a1eadf91768fe8f6e77c39e3d4f68d8ff3ecefbe0c73ab44865b62bcbb"
            ),
            "libswresample.so.6.4.100": (
                "f4b6739cce65f6a7cfd1655d68ecebb3385e01a9c4232a531eb5cc419898aeab"
            ),
            "libswscale.so.9.8.100": (
                "dc000d0874a8983f1a38c96bc4b6df2e10b7413ca2b2eaf5e73a5f6620386ffa"
            ),
        },
        library_directory="lib",
        required_ffmpeg_help={
            "muxer=mpegts": "Muxer mpegts [",
            "demuxer=mpegts": "Demuxer mpegts [",
            "demuxer=concat": "Demuxer concat [",
            "muxer=nut": "Muxer nut [",
            "demuxer=nut": "Demuxer nut [",
            "muxer=mp4": "Muxer mp4 [",
            "demuxer=mov": "Demuxer mov,mp4,",
            "muxer=matroska": "Muxer matroska [",
            "demuxer=matroska": "Demuxer matroska,webm [",
            "decoder=aac": "Decoder aac [AAC (Advanced Audio Coding)]",
            "decoder=hevc_amf": "reset_on_keyframe",
            "encoder=hevc_amf": "host_zero_copy",
            "bsf=h264_mp4toannexb": "Bit stream filter h264_mp4toannexb",
            "bsf=hevc_mp4toannexb": "Bit stream filter hevc_mp4toannexb",
            "protocol=file": "file AVOptions:",
            "protocol=pipe": "pipe AVOptions:",
        },
    ),
    "windows-amd": RuntimePolicy(
        wheel_sha256=(
            "19484a78e4bad2d19d8d1428738c71eced09d7500e4960c8a9c2547b23bf4d81"
        ),
        executables={
            "ffmpeg.exe": (
                "d046539abc5f4dbcea741058364c780161a195b331595bef25359d90af06b483"
            ),
            "ffprobe.exe": (
                "0499bb5264373bc4ff7638c77fa18ee75659e4d7b4915cc93b46f6d81773d28a"
            ),
            "dav1d.dll": (
                "53e03994ac6e1cb4980151ae8ec7cb75cea5b7b64a738f5685a479d5643ecaab"
            ),
        },
        libraries={
            "avcodec-62.dll": (
                "cfe03b1ec61c1444e53833d991b246f0e769a5eaca74ec7abd2b253f78e02d01"
            ),
            "avdevice-62.dll": (
                "c223e3c0852e0ebef1a066620b7d2821344fc5c3c8f5dc20ae67f0c7e4f707c6"
            ),
            "avfilter-11.dll": (
                "2dfea9e62fb5201b32879ea5cc72ffbcaed5b421b35e521efbca738b6e83304e"
            ),
            "avformat-62.dll": (
                "314d1fd8deca79ed7fa30293181dabcd5999b71e9bf63ba6b6ebea0a0df80f05"
            ),
            "avutil-60.dll": (
                "d9981a1903730403086763ecbdd639dc6bcbd45e802f17d6da7fae959ad07e7a"
            ),
            "swresample-6.dll": (
                "fa9a962ee22b6174a22d866d89c4ec546a1d3e68d79a24c8e404abaa91f41825"
            ),
            "swscale-9.dll": (
                "dca0936c7b3ab05ea6415537cce55fcfd2a4ce629f7f261fba3393056a3cacd9"
            ),
        },
        library_directory="bin",
        extra_source_pins={
            "DAV1D_COMMIT": "b546257f770768b2c88258c533da38b91a06f737",
        },
    ),
}


class RuntimeContractError(RuntimeError):
    """Raised when a selected native runtime is absent, mixed, or stale."""


def runtime_platform_key(platform: str | None = None) -> str:
    value = sys.platform if platform is None else str(platform)
    if value.startswith("linux"):
        return "linux-amd"
    if value == "win32":
        return "windows-amd"
    raise RuntimeContractError(f"unsupported unified runtime platform: {value}")


def default_runtime_root(platform: str | None = None) -> Path:
    key = runtime_platform_key(platform)
    override = os.environ.get("JASNA_UNIFIED_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    if key == "windows-amd":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData/Local"
        return base / "Jasna/unified-runtime/windows-amd"
    return Path.home() / ".local/share/jasna/unified-runtime/linux-amd"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_build_manifest(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise RuntimeContractError(f"invalid build manifest line: {raw_line!r}")
        values[key.strip()] = value.strip()
    return values


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeContractError(
            f"{label}: expected {expected!r}, observed {actual!r}"
        )


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeContractError(f"{label} is missing: {path}")
    _require_equal(f"{label} SHA256", sha256_file(path), expected)


def load_runtime_manifest(runtime_root: str | Path) -> dict[str, object]:
    root = Path(runtime_root).expanduser().resolve(strict=False)
    manifest_path = root / "runtime.json"
    if not manifest_path.is_file():
        raise RuntimeContractError(
            f"unified runtime is not installed at {root}; missing runtime.json"
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"cannot read unified runtime manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeContractError("unified runtime manifest root must be an object")
    return data


def validate_runtime_layout(
    runtime_root: str | Path,
    *,
    platform: str | None = None,
) -> dict[str, object]:
    """Validate immutable runtime files without importing native modules."""

    root = Path(runtime_root).expanduser().resolve(strict=False)
    key = runtime_platform_key(platform)
    policy = RUNTIME_POLICIES[key]
    data = load_runtime_manifest(root)

    _require_equal("runtime schema", data.get("schema_version"), RUNTIME_SCHEMA_VERSION)
    _require_equal("runtime platform", data.get("platform"), key)
    _require_equal("PyAV wheel SHA256", data.get("wheel_sha256"), policy.wheel_sha256)

    pins = data.get("source_pins")
    if not isinstance(pins, dict):
        raise RuntimeContractError("runtime source_pins must be an object")
    for name, expected in {
        **EXPECTED_SOURCE_PINS,
        **policy.extra_source_pins,
    }.items():
        _require_equal(name, pins.get(name), expected)

    site_packages = root / "site-packages"
    if not (site_packages / "av/__init__.py").is_file():
        raise RuntimeContractError(f"PyAV runtime is missing below {site_packages}")

    for name, expected in policy.executables.items():
        _require_file_hash(root / "bin" / name, expected, name)
    library_root = root / policy.library_directory
    for name, expected in policy.libraries.items():
        _require_file_hash(library_root / name, expected, name)
    if key == "linux-amd":
        bridge = data.get("amf_interop_bridge")
        if not isinstance(bridge, dict):
            raise RuntimeContractError(
                "Linux AMD runtime manifest is missing amf_interop_bridge"
            )
        filename = bridge.get("filename")
        digest = bridge.get("sha256")
        source_digest = bridge.get("source_sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.startswith(AMF_INTEROP_BRIDGE_PREFIX)
            or not filename.endswith(".so")
        ):
            raise RuntimeContractError(
                f"invalid Linux AMD AMF interop bridge filename: {filename!r}"
            )
        for label, value in (
            ("AMF interop bridge", digest),
            ("AMF interop bridge source", source_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeContractError(f"invalid {label} SHA256: {value!r}")
        _require_file_hash(root / "bridge" / filename, digest, "AMF interop bridge")
    elif "windows_d3d11_hip_resident_bridge" in data:
        bridge = data.get("windows_d3d11_hip_resident_bridge")
        if not isinstance(bridge, dict):
            raise RuntimeContractError(
                "Windows D3D11/HIP resident bridge metadata must be an object"
            )
        filename = bridge.get("filename")
        digest = bridge.get("sha256")
        source_digest = bridge.get("source_sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.startswith(WINDOWS_D3D11_HIP_RESIDENT_BRIDGE_PREFIX)
            or not filename.endswith(".pyd")
        ):
            raise RuntimeContractError(
                "invalid Windows D3D11/HIP resident bridge filename: "
                f"{filename!r}"
            )
        for label, value in (
            ("Windows D3D11/HIP resident bridge", digest),
            ("Windows D3D11/HIP resident bridge source", source_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeContractError(f"invalid {label} SHA256: {value!r}")
        _require_equal(
            "Windows D3D11/HIP resident bridge API version",
            bridge.get("api_version"),
            WINDOWS_D3D11_HIP_RESIDENT_API_VERSION,
        )
        _require_file_hash(
            root / "bridge" / filename,
            digest,
            "Windows D3D11/HIP resident bridge",
        )
    return data


def build_runtime_environment(
    runtime_root: str | Path,
    repo_root: str | Path,
    *,
    python_executable: str | Path,
    platform: str | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a child environment with only the selected native ABI first."""

    root = Path(runtime_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    key = runtime_platform_key(platform)
    manifest = validate_runtime_layout(root, platform=platform)

    environment = dict(os.environ if base_environment is None else base_environment)
    python_dir = str(Path(python_executable).expanduser().resolve().parent)
    python_paths = [str(root / "site-packages")]
    if key == "linux-amd" or (
        key == "windows-amd"
        and isinstance(manifest, Mapping)
        and "windows_d3d11_hip_resident_bridge" in manifest
    ):
        python_paths.append(str(root / "bridge"))
    python_paths.append(str(repo))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    path_prefix = [str(root / "bin"), python_dir]
    if key == "linux-amd":
        path_prefix.append("/opt/rocm/bin")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(root / "lib"), "/opt/amdgpu/lib/x86_64-linux-gnu", "/opt/rocm/lib")
        )
    environment["PATH"] = os.pathsep.join(
        (*path_prefix, environment.get("PATH", ""))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["JASNA_UNIFIED_RUNTIME"] = "1"
    environment["JASNA_UNIFIED_RUNTIME_ROOT"] = str(root)
    environment["JASNA_REPO_ROOT"] = str(repo)
    return environment


def activate_runtime_dll_directories(
    runtime_root: str | Path,
    *,
    platform: str | None = None,
) -> tuple[Path, ...]:
    """Register the validated native DLL root for a Windows child process.

    Python 3.8 and newer no longer use ``PATH`` alone when resolving extension
    module dependencies on Windows. Keep the ``os.add_dll_directory`` handle
    alive so every PyAV extension continues to resolve the selected FFmpeg DLLs.
    """

    key = runtime_platform_key(platform)
    if key != "windows-amd":
        return ()

    root = Path(runtime_root).expanduser().resolve()
    directory = (root / RUNTIME_POLICIES[key].library_directory).resolve()
    handle_key = os.path.normcase(str(directory))
    if handle_key in _RUNTIME_DLL_DIRECTORY_HANDLES:
        return (directory,)

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if not callable(add_dll_directory):
        raise RuntimeContractError(
            "os.add_dll_directory is unavailable for the Windows unified runtime"
        )

    try:
        handle = add_dll_directory(str(directory))
    except OSError as exc:
        raise RuntimeContractError(
            f"cannot register unified runtime DLL directory {directory}: {exc}"
        ) from exc

    _RUNTIME_DLL_DIRECTORY_HANDLES[handle_key] = handle
    return (directory,)


def validate_ffmpeg_capabilities(
    runtime_root: str | Path,
    *,
    platform: str | None = None,
) -> dict[str, str]:
    """Fail closed when a pinned CLI lacks a required product capability."""

    root = Path(runtime_root).expanduser().resolve()
    key = runtime_platform_key(platform)
    policy = RUNTIME_POLICIES[key]
    if not policy.required_ffmpeg_help:
        return {}

    executable_name = "ffmpeg.exe" if key == "windows-amd" else "ffmpeg"
    executable = root / "bin" / executable_name
    environment = dict(os.environ)
    if key == "linux-amd":
        environment["LD_LIBRARY_PATH"] = str(root / policy.library_directory)

    observed: dict[str, str] = {}
    for topic, marker in policy.required_ffmpeg_help.items():
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-h",
                    topic,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeContractError(
                f"cannot inspect FFmpeg capability {topic}: {exc}"
            ) from exc
        detail = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or marker not in detail:
            summary = detail.splitlines()[0] if detail else "no output"
            raise RuntimeContractError(
                f"unified FFmpeg is missing required capability {topic}: {summary}"
            )
        observed[topic] = marker
    return observed


def validate_loaded_runtime(
    runtime_root: str | Path,
    repo_root: str | Path,
    *,
    platform: str | None = None,
) -> dict[str, object]:
    """Validate native imports from inside the prepared child process."""

    root = Path(runtime_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    key = runtime_platform_key(platform)
    manifest = validate_runtime_layout(root, platform=platform)
    activate_runtime_dll_directories(root, platform=platform)
    ffmpeg_capabilities = validate_ffmpeg_capabilities(root, platform=platform)

    import av
    import jasna

    _require_equal("PyAV version", av.__version__, EXPECTED_AV_VERSION)
    observed_versions = {
        name: tuple(value) for name, value in av.library_versions.items()
    }
    _require_equal("FFmpeg ABI", observed_versions, EXPECTED_AV_LIBRARY_VERSIONS)

    av_file = Path(av.__file__).resolve()
    jasna_file = Path(jasna.__file__).resolve()
    if not av_file.is_relative_to(root / "site-packages"):
        raise RuntimeContractError(f"PyAV loaded outside unified runtime: {av_file}")
    if not jasna_file.is_relative_to(repo):
        raise RuntimeContractError(f"Jasna loaded outside selected repository: {jasna_file}")

    bridge_file: Path | None = None
    if key == "linux-amd":
        import _jasna_amf_surface_probe as amf_bridge

        bridge_file = Path(amf_bridge.__file__).resolve()
        if not bridge_file.is_relative_to(root / "bridge"):
            raise RuntimeContractError(
                f"AMF interop bridge loaded outside unified runtime: {bridge_file}"
            )
        for name in (
            "inspect_amf_surface",
            "verify_private_deferred_stream_dependency",
            "AmfVulkanHipInteropSession",
        ):
            if not callable(getattr(amf_bridge, name, None)):
                raise RuntimeContractError(
                    f"AMF interop bridge is missing required entry point: {name}"
                )
        session_type = amf_bridge.AmfVulkanHipInteropSession
        for name in (
            "copy_amf_surface_to_hip_resource_cache",
            "close",
            "stats",
        ):
            if not callable(getattr(session_type, name, None)):
                raise RuntimeContractError(
                    "AMF interop bridge session is missing required entry point: "
                    f"{name}"
                )
    elif (
        isinstance(manifest, Mapping)
        and "windows_d3d11_hip_resident_bridge" in manifest
    ):
        import _jasna_amf_d3d11_hip_resident as resident_bridge

        bridge_file = Path(resident_bridge.__file__).resolve()
        if not bridge_file.is_relative_to(root / "bridge"):
            raise RuntimeContractError(
                "Windows D3D11/HIP resident bridge loaded outside unified "
                f"runtime: {bridge_file}"
            )
        api_version = getattr(resident_bridge, "api_version", None)
        create_root = getattr(
            resident_bridge,
            "create_or_get_process_root",
            None,
        )
        if not callable(api_version) or not callable(create_root):
            raise RuntimeContractError(
                "Windows D3D11/HIP resident bridge is missing required entry points"
            )
        _require_equal(
            "Windows D3D11/HIP resident bridge loaded API version",
            int(api_version()),
            WINDOWS_D3D11_HIP_RESIDENT_API_VERSION,
        )

    loaded_ffmpeg_libraries: list[str] = []
    maps = Path("/proc/self/maps")
    if key == "linux-amd" and maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            mapped = line.rsplit(maxsplit=1)[-1]
            if "/libav" not in mapped and "/libsw" not in mapped:
                continue
            mapped_path = Path(mapped).resolve(strict=False)
            if not mapped_path.is_relative_to(root / "lib"):
                raise RuntimeContractError(
                    f"FFmpeg shared library loaded outside unified runtime: {mapped_path}"
                )
            loaded_ffmpeg_libraries.append(str(mapped_path))

    return {
        "status": "PASSED",
        "platform": key,
        "runtime_root": str(root),
        "repo_root": str(repo),
        "python": sys.executable,
        "pyav_version": av.__version__,
        "pyav_file": str(av_file),
        "amf_interop_bridge_file": str(bridge_file) if bridge_file is not None else None,
        "ffmpeg_abi": {name: list(value) for name, value in observed_versions.items()},
        "ffmpeg_capabilities": ffmpeg_capabilities,
        "loaded_ffmpeg_libraries": sorted(set(loaded_ffmpeg_libraries)),
    }
