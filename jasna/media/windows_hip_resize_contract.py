"""CPU-only admission checks for the exact Windows HIP resize bundle.

This module deliberately does not import Torch, PyAV, or the HIP loader.  It
only validates the immutable bundle and the small geometry envelope that the
native launch path is allowed to hand to the shared resize ABI.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ENV = "JASNA_WINDOWS_HIP_RESIZE"
CODE_OBJECT = "resize_normalize.gfx1100.windows.co"
MANIFEST = "hip_resize_normalize.gfx1100.windows.json"

_SOURCE_FILE = "resize_normalize.cu"
_SCHEMA = "jasna.hip-resize.windows.v1"
_ARCHITECTURE = "gfx1100"
_TORCH_HIP = "7.16.26354"
_RUNTIME_VERSION = 71626354
_RUNTIME_DLL_SHA256 = "37c40daa884de68bb7b5ee1b0575ab97b96866adf304fd76fdbccba772830f5f"
_PARAMETER_ABI = "jasna.resize-normalize.params.v1"
_SOURCE_SHA256 = "206391fa046d3a96b725eaa14e66b9bbfe8e95b903421fe4be74ac270fef63ab"
_ARTIFACT_SHA256 = "3c93e066930ad74bfa90f50ffdc059b230bc3ccfcc1ead3b1aa40034d29d0a10"

_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1
_ELF_HEADER_BYTES = 64
_ELFOSABI_AMDGPU_HSA = 64
_ELF_AMDGPU_MACHINE = 224

_EXPECTED_MANIFEST: dict[str, object] = {
    "schema": _SCHEMA,
    "platform": "win32",
    "architecture": _ARCHITECTURE,
    "torch_hip": _TORCH_HIP,
    "runtime_version": _RUNTIME_VERSION,
    "runtime_dll_sha256": _RUNTIME_DLL_SHA256,
    "parameter_abi": _PARAMETER_ABI,
    "source": {"file": _SOURCE_FILE, "sha256": _SOURCE_SHA256},
    "artifact": {"file": CODE_OBJECT, "sha256": _ARTIFACT_SHA256},
}


def accepted_manifest() -> dict[str, object]:
    """Return an independent copy of the only accepted Windows resize manifest."""

    return deepcopy(_EXPECTED_MANIFEST)


def requested(environ: Mapping[str, str]) -> bool:
    """Return the explicit Windows HIP resize request without reading ``os.environ``."""

    raw = environ.get(ENV, "")
    if not isinstance(raw, str):
        raise ValueError(f"{ENV} must be a string switch")
    value = raw.strip().casefold()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{ENV} must be 0/1, false/true, no/yes, or off/on")


def _strict_ints(value: object, count: int) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(values) != count or any(type(item) is not int for item in values):
        return None
    return values


def supported_geometry(
    shape: object,
    strides: object,
    out_hw: object,
    content: object,
) -> bool:
    """Return whether a source view fits the exact shared resize ABI envelope.

    Positive strided views are intentionally allowed: the accepted VR path uses
    a non-contiguous eye view.  The nested-span checks reject aliases between
    source rows, channels, and batches while keeping ordinary pitched views.
    """

    source_shape = _strict_ints(shape, 4)
    source_strides = _strict_ints(strides, 4)
    output_shape = _strict_ints(out_hw, 2)
    placement = _strict_ints(content, 4)
    if (
        source_shape is None
        or source_strides is None
        or output_shape is None
        or placement is None
    ):
        return False

    batch, channels, height, width = source_shape
    batch_stride, channel_stride, row_stride, final_stride = source_strides
    out_height, out_width = output_shape
    left, top, content_width, content_height = placement

    # The shared kernel marshals shape and placement through signed C ints.
    if any(
        value < 0 or value > _INT32_MAX
        for value in (*source_shape, *output_shape, *placement)
    ):
        return False
    if not (1 <= batch <= 4 and channels == 3 and 1 <= height <= 8192 and 1 <= width <= 8192):
        return False
    if not (1 <= out_height <= 640 and 1 <= out_width <= 640):
        return False
    if not (content_width > 0 and content_height > 0):
        return False
    if left + content_width > out_width or top + content_height > out_height:
        return False

    # Source strides are passed as signed C int64 values.  The last dimension
    # is byte-contiguous for the uint8 source; other dimensions may be pitched.
    if (
        any(stride <= 0 or stride > _INT64_MAX for stride in source_strides)
        or final_stride != 1
    ):
        return False

    row_span = (height - 1) * row_stride + width
    if row_span > _INT64_MAX or (height > 1 and row_stride < width):
        return False
    channel_span = (channels - 1) * channel_stride + row_span
    if channel_span > _INT64_MAX or (channels > 1 and channel_stride < row_span):
        return False
    batch_span = (batch - 1) * batch_stride + channel_span
    if batch_span > _INT64_MAX or (batch > 1 and batch_stride < channel_span):
        return False
    return True


def _sha256_file(path: Path, *, maximum_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            while True:
                if maximum_bytes is None:
                    request_size = 64 * 1024
                else:
                    request_size = min(64 * 1024, maximum_bytes - total + 1)
                chunk = handle.read(request_size)
                if not chunk:
                    break
                total += len(chunk)
                if maximum_bytes is not None and total > maximum_bytes:
                    raise RuntimeError(
                        f"Windows HIP resize file exceeds {maximum_bytes} bytes: {path}"
                    )
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot read Windows HIP resize file {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise RuntimeError(f"cannot read Windows HIP resize manifest {path}: {exc}") from exc
    if len(data) > _MAX_MANIFEST_BYTES:
        raise RuntimeError(
            f"Windows HIP resize manifest exceeds {_MAX_MANIFEST_BYTES} bytes: {path}"
        )
    try:
        manifest = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid Windows HIP resize manifest {path}: {exc}") from exc
    if type(manifest) is not dict:
        raise RuntimeError(f"Windows HIP resize manifest must be an object: {path}")
    return manifest


def _same_contract_value(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return observed.keys() == expected.keys() and all(  # type: ignore[union-attr]
            _same_contract_value(observed[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if type(expected) is list:
        return len(observed) == len(expected) and all(  # type: ignore[arg-type]
            _same_contract_value(item, reference)
            for item, reference in zip(observed, expected, strict=True)  # type: ignore[arg-type]
        )
    return observed == expected


def _validate_runtime_identity(
    runtime_identity: dict[str, object], torch_hip: str, architecture: str
) -> None:
    if not isinstance(runtime_identity, Mapping):
        raise RuntimeError("Windows HIP resize runtime identity must be a mapping")
    expected_identity = {
        "torch_hip": _TORCH_HIP,
        "runtime_version": _RUNTIME_VERSION,
        "runtime_dll_sha256": _RUNTIME_DLL_SHA256,
    }
    for key, expected in expected_identity.items():
        observed = runtime_identity.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeError(
                f"Windows HIP resize runtime identity {key} mismatch: "
                f"expected {expected!r}, observed {observed!r}"
            )
    if type(torch_hip) is not str or torch_hip != _TORCH_HIP:
        raise RuntimeError(
            f"Windows HIP resize Torch HIP mismatch: expected {_TORCH_HIP!r}, "
            f"observed {torch_hip!r}"
        )
    if type(architecture) is not str or architecture != _ARCHITECTURE:
        raise RuntimeError(
            f"Windows HIP resize architecture mismatch: expected {_ARCHITECTURE!r}, "
            f"observed {architecture!r}"
        )


def _validate_elf_header(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(_ELF_HEADER_BYTES)
    except OSError as exc:
        raise RuntimeError(f"cannot read Windows HIP resize code object {path}: {exc}") from exc
    if len(header) != _ELF_HEADER_BYTES:
        raise RuntimeError(f"Windows HIP resize code object has a short ELF header: {path}")
    if header[:4] != b"\x7fELF" or header[4:7] != bytes((2, 1, 1)):
        raise RuntimeError(f"Windows HIP resize code object is not ELF64 little-endian: {path}")
    if header[7] != _ELFOSABI_AMDGPU_HSA or header[8] != 4:
        raise RuntimeError(f"Windows HIP resize code object is not AMDGPU HSA ABI 4: {path}")
    if header[16:18] != (3).to_bytes(2, "little"):
        raise RuntimeError(f"Windows HIP resize code object is not a shared ELF object: {path}")
    if header[18:20] != _ELF_AMDGPU_MACHINE.to_bytes(2, "little"):
        raise RuntimeError(f"Windows HIP resize code object is not for AMDGPU: {path}")
    if header[20:24] != (1).to_bytes(4, "little"):
        raise RuntimeError(f"Windows HIP resize code object has an invalid ELF version: {path}")


def validate_bundle(
    directory: Path,
    runtime_identity: dict[str, object],
    torch_hip: str,
    architecture: str,
) -> Path:
    """Validate and return the fixed, hash-pinned Windows HIP resize code object.

    The manifest cannot select paths.  A frozen package may omit the rebuild-only
    CUDA source, but a shipped source file must match the accepted source hash.
    """

    bundle = Path(directory)
    manifest = _read_manifest(bundle / MANIFEST)
    if not _same_contract_value(manifest, _EXPECTED_MANIFEST):
        raise RuntimeError("Windows HIP resize manifest does not match the accepted contract")
    _validate_runtime_identity(runtime_identity, torch_hip, architecture)

    artifact = bundle / CODE_OBJECT
    if not artifact.is_file():
        raise RuntimeError(f"missing Windows HIP resize code object: {artifact}")
    try:
        artifact_size = artifact.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"cannot stat Windows HIP resize code object {artifact}: {exc}") from exc
    if artifact_size > _MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            f"Windows HIP resize file exceeds {_MAX_ARTIFACT_BYTES} bytes: {artifact}"
        )
    artifact_hash = _sha256_file(artifact, maximum_bytes=_MAX_ARTIFACT_BYTES)
    if artifact_hash != _ARTIFACT_SHA256:
        raise RuntimeError(
            "Windows HIP resize code-object SHA256 mismatch: "
            f"expected {_ARTIFACT_SHA256}, observed {artifact_hash}"
        )
    _validate_elf_header(artifact)

    source = bundle / _SOURCE_FILE
    if source.exists():
        if not source.is_file():
            raise RuntimeError(f"Windows HIP resize source is not a file: {source}")
        source_hash = _sha256_file(source)
        if source_hash != _SOURCE_SHA256:
            raise RuntimeError(
                "Windows HIP resize source SHA256 mismatch: "
                f"expected {_SOURCE_SHA256}, observed {source_hash}"
            )
    return artifact
