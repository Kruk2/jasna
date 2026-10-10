from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DTYPE_NAMES = {
    "float16": torch.float16,
    "float32": torch.float32,
}
_MIGRAPHX_TYPE_NAMES = {
    "float16": "half_type",
    "float32": "float_type",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migraphx_manifest_sha256(path: str | Path) -> str:
    """Return a sidecar digest for callers that pin an explicit artifact."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MIGraphX artifact manifest not found: {manifest_path}")
    return _file_sha256(manifest_path)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"MIGraphX manifest {label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"MIGraphX manifest {label} keys mismatch: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"MIGraphX manifest {label} must be a non-empty string")
    return value


def _sha256_text(value: object, label: str) -> str:
    result = _text(value, label)
    if not _SHA256_RE.fullmatch(result):
        raise RuntimeError(f"MIGraphX manifest {label} must be lowercase SHA256")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"MIGraphX manifest {label} must be a positive integer")
    return value


def _integer_tuple(value: object, label: str, *, positive: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"MIGraphX manifest {label} must be a non-empty integer list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise RuntimeError(f"MIGraphX manifest {label} must contain only integers")
        if positive and item <= 0:
            raise RuntimeError(f"MIGraphX manifest {label} must contain only positive integers")
        if not positive and item < 0:
            raise RuntimeError(f"MIGraphX manifest {label} cannot contain negative integers")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class MigraphxTensorContract:
    logical_name: str
    parameter_name: str
    dtype_name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPE_NAMES[self.dtype_name]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_json(cls, value: object, label: str) -> "MigraphxTensorContract":
        data = _mapping(value, label)
        _exact_keys(
            data,
            {"logical_name", "parameter_name", "dtype", "shape", "strides"},
            label,
        )
        dtype_name = _text(data["dtype"], f"{label}.dtype")
        if dtype_name not in _DTYPE_NAMES:
            raise RuntimeError(
                f"MIGraphX manifest {label}.dtype is unsupported: {dtype_name!r}"
            )
        shape = _integer_tuple(data["shape"], f"{label}.shape", positive=True)
        strides = _integer_tuple(data["strides"], f"{label}.strides", positive=False)
        if len(shape) != len(strides):
            raise RuntimeError(
                f"MIGraphX manifest {label} shape/stride rank mismatch: "
                f"{len(shape)} != {len(strides)}"
            )
        return cls(
            logical_name=_text(data["logical_name"], f"{label}.logical_name"),
            parameter_name=_text(data["parameter_name"], f"{label}.parameter_name"),
            dtype_name=dtype_name,
            shape=shape,
            strides=strides,
        )


@dataclass(frozen=True)
class MigraphxArtifactManifest:
    """Immutable sidecar contract for one already-compiled MXR program."""

    manifest_path: Path
    purpose: str
    artifact_path: Path
    artifact_sha256: str
    artifact_size_bytes: int
    source_sha256: str
    platform: str
    device_arch: str
    torch_version: str
    torch_hip_version: str
    migraphx_version: str
    torch_migraphx_version: str
    internal_precision: str
    stream_type: str
    inputs: tuple[MigraphxTensorContract, ...]
    outputs: tuple[MigraphxTensorContract, ...]

    def validate_files(self, source_path: str | Path) -> None:
        artifact = self.artifact_path
        if not artifact.is_file():
            raise FileNotFoundError(f"MIGraphX artifact not found: {artifact}")
        if artifact.stat().st_size != self.artifact_size_bytes:
            raise RuntimeError(
                f"MIGraphX artifact size mismatch: {artifact.stat().st_size} "
                f"!= {self.artifact_size_bytes}"
            )
        actual_artifact_sha = _file_sha256(artifact)
        if actual_artifact_sha != self.artifact_sha256:
            raise RuntimeError(
                f"MIGraphX artifact SHA256 mismatch: {actual_artifact_sha} "
                f"!= {self.artifact_sha256}"
            )
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"MIGraphX source model not found: {source}")
        actual_source_sha = _file_sha256(source)
        if actual_source_sha != self.source_sha256:
            raise RuntimeError(
                f"MIGraphX source model SHA256 mismatch: {actual_source_sha} "
                f"!= {self.source_sha256}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "MigraphxArtifactManifest":
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MIGraphX artifact manifest not found: {manifest_path}")
        try:
            root = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read MIGraphX artifact manifest: {manifest_path}"
            ) from exc
        data = _mapping(root, "root")
        _exact_keys(
            data,
            {"schema_version", "backend", "purpose", "artifact", "source", "target", "program"},
            "root",
        )
        if data["schema_version"] != 1:
            raise RuntimeError(
                f"Unsupported MIGraphX manifest schema_version: {data['schema_version']!r}"
            )
        if data["backend"] != "migraphx":
            raise RuntimeError(
                f"MIGraphX manifest backend must be 'migraphx', got {data['backend']!r}"
            )

        artifact = _mapping(data["artifact"], "artifact")
        _exact_keys(artifact, {"path", "sha256", "size_bytes"}, "artifact")
        artifact_path = Path(_text(artifact["path"], "artifact.path"))
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path

        source = _mapping(data["source"], "source")
        _exact_keys(source, {"sha256"}, "source")
        target = _mapping(data["target"], "target")
        _exact_keys(
            target,
            {
                "platform",
                "device_arch",
                "torch_version",
                "torch_hip_version",
                "migraphx_version",
                "torch_migraphx_version",
            },
            "target",
        )
        program = _mapping(data["program"], "program")
        _exact_keys(
            program,
            {"internal_precision", "stream_type", "inputs", "outputs"},
            "program",
        )
        raw_inputs = program["inputs"]
        raw_outputs = program["outputs"]
        if not isinstance(raw_inputs, list) or not isinstance(raw_outputs, list):
            raise RuntimeError("MIGraphX manifest program inputs/outputs must be lists")
        inputs = tuple(
            MigraphxTensorContract.from_json(value, f"program.inputs[{index}]")
            for index, value in enumerate(raw_inputs)
        )
        outputs = tuple(
            MigraphxTensorContract.from_json(value, f"program.outputs[{index}]")
            for index, value in enumerate(raw_outputs)
        )
        if len(inputs) != 1:
            raise RuntimeError(
                f"MIGraphX pointer runner requires exactly one input, got {len(inputs)}"
            )
        if not outputs:
            raise RuntimeError("MIGraphX pointer runner requires at least one output")
        logical_names = [item.logical_name for item in (*inputs, *outputs)]
        parameter_names = [item.parameter_name for item in (*inputs, *outputs)]
        if len(logical_names) != len(set(logical_names)):
            raise RuntimeError("MIGraphX manifest logical tensor names must be unique")
        if len(parameter_names) != len(set(parameter_names)):
            raise RuntimeError("MIGraphX manifest parameter tensor names must be unique")

        return cls(
            manifest_path=manifest_path,
            purpose=_text(data["purpose"], "purpose"),
            artifact_path=artifact_path,
            artifact_sha256=_sha256_text(artifact["sha256"], "artifact.sha256"),
            artifact_size_bytes=_positive_int(artifact["size_bytes"], "artifact.size_bytes"),
            source_sha256=_sha256_text(source["sha256"], "source.sha256"),
            platform=_text(target["platform"], "target.platform"),
            device_arch=_text(target["device_arch"], "target.device_arch"),
            torch_version=_text(target["torch_version"], "target.torch_version"),
            torch_hip_version=_text(target["torch_hip_version"], "target.torch_hip_version"),
            migraphx_version=_text(target["migraphx_version"], "target.migraphx_version"),
            torch_migraphx_version=_text(
                target["torch_migraphx_version"], "target.torch_migraphx_version"
            ),
            internal_precision=_text(program["internal_precision"], "program.internal_precision"),
            stream_type=_text(program["stream_type"], "program.stream_type"),
            inputs=inputs,
            outputs=outputs,
        )


@dataclass(frozen=True)
class _RuntimeBindings:
    migraphx: Any
    mgx_argument_from_ptr: Any
    tensors_from_mgx_arguments: Any
    migraphx_version: str
    torch_migraphx_version: str


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required MIGraphX runtime package is missing: {name}") from exc


def _migraphx_python_search_dirs() -> tuple[Path, ...]:
    """Find only official ROCm libraries built for this Python ABI."""

    roots: list[Path] = []
    configured_root = os.environ.get("ROCM_PATH", "").strip()
    if configured_root:
        roots.append(Path(configured_root))
    roots.append(Path("/opt/rocm"))
    opt = Path("/opt")
    if opt.is_dir():
        roots.extend(sorted(opt.glob("rocm-*"), reverse=True))

    extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "")
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        library_dir = root / "lib"
        key = str(library_dir.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if extension_suffix and (library_dir / f"migraphx{extension_suffix}").is_file():
            candidates.append(library_dir)
    return tuple(candidates)


def _import_migraphx() -> Any:
    try:
        return importlib.import_module("migraphx")
    except ModuleNotFoundError as exc:
        if exc.name != "migraphx":
            raise

    searched: list[str] = []
    for library_dir in _migraphx_python_search_dirs():
        searched.append(str(library_dir))
        sys.path.insert(0, str(library_dir))
        try:
            importlib.invalidate_caches()
            return importlib.import_module("migraphx")
        except ModuleNotFoundError as exc:
            if exc.name != "migraphx":
                raise
        finally:
            try:
                sys.path.remove(str(library_dir))
            except ValueError:
                pass
    raise RuntimeError(
        "The AMD MIGraphX artifact backend requires the official MIGraphX "
        "Python bindings for this interpreter ABI. Searched ROCm library "
        f"directories: {searched!r}"
    )


def _load_runtime_bindings() -> _RuntimeBindings:
    migraphx = _import_migraphx()
    original_path = os.environ.get("PATH")
    interpreter_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = os.pathsep.join(
        part for part in (interpreter_bin, original_path or "") if part
    )
    try:
        # The supported upstream package owns the pointer bridge/JIT lifecycle.
        # Only the active interpreter's bin is temporarily prepended so a shell
        # PATH or external research environment cannot change that dependency.
        importlib.import_module("torch_migraphx")
        utils = importlib.import_module("torch_migraphx.fx.utils")
        mgx_argument_from_ptr = utils.mgx_argument_from_ptr
        tensors_from_mgx_arguments = utils.tensors_from_mgx_arguments
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "The AMD MIGraphX artifact backend requires the product-provisioned "
            "MIGraphX and torch-migraphx runtimes"
        ) from exc
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path
    migraphx_version = getattr(migraphx, "__version__", None)
    if not isinstance(migraphx_version, str) or not migraphx_version:
        migraphx_version = _package_version("migraphx")
    return _RuntimeBindings(
        migraphx=migraphx,
        mgx_argument_from_ptr=mgx_argument_from_ptr,
        tensors_from_mgx_arguments=tensors_from_mgx_arguments,
        migraphx_version=migraphx_version,
        torch_migraphx_version=_package_version("torch-migraphx"),
    )


def _shape_contract(shape: object) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (
        str(shape.type_string()),
        tuple(int(value) for value in shape.lens()),
        tuple(int(value) for value in shape.strides()),
    )


class MigraphxArtifactRunner:
    """Fail-closed GPU-pointer runner for an explicitly supplied MXR artifact.

    The sidecar is the compatibility boundary. A missing dependency, changed
    source/artifact digest, platform/runtime/architecture mismatch, tensor ABI
    mismatch, or invalid pointer result raises immediately. This class never
    copies the input or falls back to PyTorch/another backend.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        source_path: str | Path,
        device: torch.device,
        expected_purpose: str,
    ) -> None:
        self._program = None
        self._bindings: _RuntimeBindings | None = None
        self._arguments: dict[str, object] = {}
        self._output_buffers: dict[str, torch.Tensor] = {}
        self.manifest = MigraphxArtifactManifest.load(manifest_path)
        if self.manifest.purpose != expected_purpose:
            raise RuntimeError(
                f"MIGraphX artifact purpose mismatch: {self.manifest.purpose!r} "
                f"!= {expected_purpose!r}"
            )
        self.device = torch.device(device)
        self._validate_files(Path(source_path))
        bindings = _load_runtime_bindings()
        self._validate_runtime(bindings)
        program = bindings.migraphx.load(str(self.manifest.artifact_path))
        if not program.is_compiled():
            raise RuntimeError("MIGraphX artifact is not a compiled program")
        self._validate_program(program)

        self.input_names = [item.logical_name for item in self.manifest.inputs]
        self.input_dtypes = {item.logical_name: item.dtype for item in self.manifest.inputs}
        self.output_names = [item.logical_name for item in self.manifest.outputs]
        self.outputs = {item.logical_name: item for item in self.manifest.outputs}
        self.artifact_path = self.manifest.artifact_path
        self.batch_size = int(self.manifest.inputs[0].shape[0])
        self._output_shapes = tuple(program.get_output_shapes())
        for contract in self.manifest.outputs:
            buffer = torch.empty_strided(
                contract.shape,
                contract.strides,
                dtype=contract.dtype,
                device=self.device,
            )
            self._output_buffers[contract.parameter_name] = buffer
            self._arguments[contract.parameter_name] = bindings.mgx_argument_from_ptr(
                buffer.data_ptr(), program.get_parameter_shapes()[contract.parameter_name]
            )
        self._bindings = bindings
        self._program = program

    def _validate_files(self, source_path: Path) -> None:
        self.manifest.validate_files(source_path)

    def _validate_runtime(self, bindings: _RuntimeBindings) -> None:
        if self.device.type != "cuda" or torch.version.hip is None:
            raise RuntimeError("MIGraphX artifact backend requires a ROCm CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("MIGraphX artifact backend requires an available ROCm GPU")
        if self.manifest.platform != sys.platform:
            raise RuntimeError(
                f"MIGraphX platform mismatch: {sys.platform!r} != {self.manifest.platform!r}"
            )
        versions = {
            "torch": str(torch.__version__),
            "torch_hip": str(torch.version.hip),
            "migraphx": bindings.migraphx_version,
            "torch_migraphx": bindings.torch_migraphx_version,
        }
        expected = {
            "torch": self.manifest.torch_version,
            "torch_hip": self.manifest.torch_hip_version,
            "migraphx": self.manifest.migraphx_version,
            "torch_migraphx": self.manifest.torch_migraphx_version,
        }
        if versions != expected:
            raise RuntimeError(
                f"MIGraphX runtime version mismatch: actual={versions!r}, expected={expected!r}"
            )
        properties = torch.cuda.get_device_properties(self.device)
        actual_arch = getattr(properties, "gcnArchName", None)
        if actual_arch != self.manifest.device_arch:
            raise RuntimeError(
                f"MIGraphX device architecture mismatch: {actual_arch!r} "
                f"!= {self.manifest.device_arch!r}"
            )

    def _validate_program(self, program: object) -> None:
        parameters = program.get_parameter_shapes()
        expected_names = {
            item.parameter_name for item in (*self.manifest.inputs, *self.manifest.outputs)
        }
        if set(parameters) != expected_names:
            raise RuntimeError(
                "MIGraphX program parameter names mismatch: "
                f"{sorted(parameters)} != {sorted(expected_names)}"
            )
        for contract in (*self.manifest.inputs, *self.manifest.outputs):
            actual = _shape_contract(parameters[contract.parameter_name])
            expected = (
                _MIGRAPHX_TYPE_NAMES[contract.dtype_name],
                contract.shape,
                contract.strides,
            )
            if actual != expected:
                raise RuntimeError(
                    f"MIGraphX tensor ABI mismatch for {contract.parameter_name!r}: "
                    f"{actual!r} != {expected!r}"
                )
        output_shapes = tuple(program.get_output_shapes())
        if len(output_shapes) != len(self.manifest.outputs):
            raise RuntimeError(
                f"MIGraphX output count mismatch: {len(output_shapes)} "
                f"!= {len(self.manifest.outputs)}"
            )
        for index, (shape, contract) in enumerate(zip(output_shapes, self.manifest.outputs)):
            actual = _shape_contract(shape)
            expected = (
                _MIGRAPHX_TYPE_NAMES[contract.dtype_name],
                contract.shape,
                contract.strides,
            )
            if actual != expected:
                raise RuntimeError(
                    f"MIGraphX output ABI mismatch at index {index}: "
                    f"{actual!r} != {expected!r}"
                )

    def close(self) -> None:
        self._arguments.clear()
        self._output_buffers.clear()
        self._output_shapes = ()
        self._program = None
        self._bindings = None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._program is None or self._bindings is None:
            raise RuntimeError("MIGraphX artifact runner is closed")
        if set(inputs) != set(self.input_names):
            raise RuntimeError(
                f"MIGraphX input names mismatch: {sorted(inputs)} != {sorted(self.input_names)}"
            )
        contract = self.manifest.inputs[0]
        value = inputs[contract.logical_name]
        if value.device != self.device:
            raise RuntimeError(
                f"MIGraphX input device mismatch: {value.device} != {self.device}"
            )
        if value.dtype != contract.dtype:
            raise RuntimeError(
                f"MIGraphX input dtype mismatch: {value.dtype} != {contract.dtype}"
            )
        if tuple(value.shape) != contract.shape or tuple(value.stride()) != contract.strides:
            raise RuntimeError(
                f"MIGraphX input ABI mismatch: shape={tuple(value.shape)}, "
                f"strides={tuple(value.stride())}; expected shape={contract.shape}, "
                f"strides={contract.strides}"
            )
        parameter_shape = self._program.get_parameter_shapes()[contract.parameter_name]
        self._arguments[contract.parameter_name] = self._bindings.mgx_argument_from_ptr(
            value.data_ptr(), parameter_shape
        )
        stream = torch.cuda.current_stream(self.device)
        raw_outputs = self._program.run_async(
            self._arguments,
            stream.cuda_stream,
            self.manifest.stream_type,
        )
        output_tensors = self._bindings.tensors_from_mgx_arguments(
            raw_outputs, self._output_shapes
        )
        if len(output_tensors) != len(self.manifest.outputs):
            raise RuntimeError(
                f"MIGraphX runtime output count mismatch: {len(output_tensors)} "
                f"!= {len(self.manifest.outputs)}"
            )
        result: dict[str, torch.Tensor] = {}
        for value, output_contract in zip(output_tensors, self.manifest.outputs):
            if (
                value.device != self.device
                or value.dtype != output_contract.dtype
                or tuple(value.shape) != output_contract.shape
                or tuple(value.stride()) != output_contract.strides
            ):
                raise RuntimeError(
                    f"MIGraphX runtime output ABI mismatch for "
                    f"{output_contract.logical_name!r}"
                )
            owned_buffer = self._output_buffers[output_contract.parameter_name]
            if value.data_ptr() != owned_buffer.data_ptr():
                raise RuntimeError(
                    "MIGraphX runtime output pointer mismatch for "
                    f"{output_contract.logical_name!r}: {value.data_ptr()} != "
                    f"owned {owned_buffer.data_ptr()}"
                )
            result[output_contract.logical_name] = value
        return result
