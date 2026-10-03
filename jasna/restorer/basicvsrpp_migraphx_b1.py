"""Strict Linux AMD loader for the accepted BasicVSR++ B1 artifacts.

Only the repeated ``i > 0`` body of each propagation direction is dispatched
to four static-B1 Torch-MIGraphX GraphModules.  Model loading, optical flow,
first-frame propagation, reconstruction, and every other operation remain on
the established PyTorch path.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from jasna.accelerator import is_amd_device
from jasna.migraphx_artifact import _import_migraphx
from jasna.models.basicvsrpp.mmagic.flow_warp import flow_warp

BASICVSRPP_MIGRAPHX_B1_ENV = "JASNA_BASICVSRPP_MIGRAPHX_B1"
BASICVSRPP_MIGRAPHX_B1_DIR_ENV = "JASNA_BASICVSRPP_MIGRAPHX_B1_DIR"
TORCH_MIGRAPHX_EXTENSION_ENV = "JASNA_TORCH_MIGRAPHX_EXTENSION"
PRODUCT_ARTIFACT_DIRECTORY = "basicvsrpp-b1-migraphx-gfx1100"

_DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")
_INPUT_NAMES = (
    "feat_prop",
    "grid_n1",
    "feat_n2",
    "grid_n2",
    "feat_current",
    "flow_n1",
    "flow_n2",
    "backbone_prefix",
)
_PREFIX_CHANNELS = {
    "backward_1": 64,
    "forward_1": 128,
    "backward_2": 192,
    "forward_2": 256,
}
_SEMANTIC_SOURCE_HASHES = {
    "jasna/models/basicvsrpp/inference.py": "6b6f12e62eac8e06be338347ea6e02955df77e8137ed9bfca7f43ef5a5f82fe9",
    "jasna/models/basicvsrpp/mmagic/base_edit_model.py": "7cbd31895ff4a31d57277b8c9d0f9b5cdf6871a05d33e48419deac0932634616",
    "jasna/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py": "ecb93a88abb92a46958321f33186f25b7e5a4608b526ebe20b4f8b54aec297f9",
    "jasna/models/basicvsrpp/mmagic/flow_warp.py": "e297e303453be80f44edc91e8a5cac6efa84dd66a88f420902c0f8a054ca0462",
    "jasna/models/basicvsrpp/mmagic/real_basicvsr.py": "1fd2686a5267f39f8add38ebd1b88e3a26427fc163003dd127d4379a5ac78e01",
    "jasna/restorer/basicvsrpp_sub_engines.py": "8e2f7b67ce72b68a44cb1b440bd5cb80f34bf805a088b348cf8b61e6d36749fb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _override_from_environment(environ: Mapping[str, str]) -> bool | None:
    raw = environ.get(BASICVSRPP_MIGRAPHX_B1_ENV, "auto").strip().casefold()
    if raw in {"", "auto"}:
        return None
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{BASICVSRPP_MIGRAPHX_B1_ENV} must be auto, 0/1, false/true, no/yes, or off/on"
    )


def basicvsrpp_migraphx_b1_enabled(
    device: torch.device | str,
    *,
    fp16: bool,
    checkpoint_path: str | Path,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Auto-select the installed artifact or fail closed when forced."""

    selected = os.environ if environ is None else environ
    override = _override_from_environment(selected)
    if override is False:
        return False
    resolved = torch.device(device)
    eligible = (
        sys.platform == "linux"
        and bool(fp16)
        and resolved.type == "cuda"
        and is_amd_device(resolved)
        and getattr(torch.version, "hip", None) is not None
        and torch.cuda.is_available()
    )
    if not eligible:
        if override is True:
            raise RuntimeError(
                f"{BASICVSRPP_MIGRAPHX_B1_ENV}=1 requires FP16 on an "
                "available Linux AMD/ROCm device"
            )
        return False
    architecture = str(
        getattr(torch.cuda.get_device_properties(resolved), "gcnArchName", "")
    ).split(":", 1)[0]
    if architecture != "gfx1100":
        if override is True:
            raise RuntimeError(
                f"{BASICVSRPP_MIGRAPHX_B1_ENV}=1 requires gfx1100, got {architecture!r}"
            )
        return False
    directory = artifact_directory(checkpoint_path, environ=selected)
    required = (
        directory / "B1_COLD_MANIFEST.json",
        directory / "B1_COLD_MANIFEST.sha256",
        *(directory / f"b1_{direction}.torch" for direction in _DIRECTIONS),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        if override is True:
            raise RuntimeError(
                f"{BASICVSRPP_MIGRAPHX_B1_ENV}=1 is missing artifact files: {missing}"
            )
        return False
    return True


def artifact_directory(
    checkpoint_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    selected = os.environ if environ is None else environ
    configured = selected.get(BASICVSRPP_MIGRAPHX_B1_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(checkpoint_path).resolve().parent / PRODUCT_ARTIFACT_DIRECTORY


def _canonical_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for size in reversed(shape):
        reversed_strides.append(stride)
        stride *= size
    return tuple(reversed(reversed_strides))


def _canonicalize(value: torch.Tensor) -> torch.Tensor:
    shape = tuple(int(size) for size in value.shape)
    expected = _canonical_strides(shape)
    if tuple(int(stride) for stride in value.stride()) != expected:
        value = value.clone(memory_format=torch.contiguous_format)
    if tuple(int(stride) for stride in value.stride()) != expected:
        raise RuntimeError(
            f"could not canonicalize tensor strides: {value.stride()} != {expected}"
        )
    return value


def _artifact_shapes(direction: str) -> tuple[tuple[int, ...], ...]:
    prefix = _PREFIX_CHANNELS[direction]
    return (
        (1, 64, 64, 64),
        (1, 64, 64, 2),
        (1, 64, 64, 64),
        (1, 64, 64, 2),
        (1, 64, 64, 64),
        (1, 2, 64, 64),
        (1, 2, 64, 64),
        (1, prefix, 64, 64),
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _extension_candidates(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get(TORCH_MIGRAPHX_EXTENSION_ENV, "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(sorted(directory.glob("_torch_migraphx*.so")))
    extensions_root = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    roots = [Path(extensions_root)] if extensions_root else []
    roots.append(Path.home() / ".cache" / "torch_extensions")
    for root in roots:
        if root.is_dir():
            candidates.extend(
                sorted(root.glob("*/_torch_migraphx/_torch_migraphx*.so"))
            )
    return candidates


def _preload_torch_migraphx_extension(
    directory: Path, expected_sha256: str
) -> Path:
    loaded = sys.modules.get("_torch_migraphx")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if not loaded_path.is_file() or _sha256(loaded_path) != expected_sha256:
            raise RuntimeError(
                "the loaded Torch-MIGraphX extension differs from the artifact manifest"
            )
        return loaded_path

    # torch_migraphx._C loads its native module through
    # torch.utils.cpp_extension.load().  That API returns the extension object
    # without necessarily retaining it in sys.modules under
    # ``_torch_migraphx``.  RF-DETR initializes this route before the restorer,
    # so inspect the object owned by _C before trying a second import.
    torch_migraphx_c = sys.modules.get("torch_migraphx._C")
    jit_loaded = getattr(torch_migraphx_c, "_mod", None)
    if jit_loaded is not None:
        loaded_path = Path(str(getattr(jit_loaded, "__file__", ""))).resolve()
        if not loaded_path.is_file() or _sha256(loaded_path) != expected_sha256:
            raise RuntimeError(
                "the loaded Torch-MIGraphX extension differs from the artifact manifest"
            )
        return loaded_path

    for extension in _extension_candidates(directory):
        if extension.is_file() and _sha256(extension) == expected_sha256:
            parent = str(extension.resolve().parent)
            sys.path.insert(0, parent)
            try:
                importlib.invalidate_caches()
                imported = importlib.import_module("_torch_migraphx")
            finally:
                try:
                    sys.path.remove(parent)
                except ValueError:
                    pass
            loaded_path = Path(str(imported.__file__)).resolve()
            # CPython may reuse an already-loaded extension object from an
            # equal-bytes cache copy while updating the new module spec to the
            # selected artifact path.  The manifest's content digest is the
            # compatibility/security boundary; physical path identity is not.
            if not loaded_path.is_file() or _sha256(loaded_path) != expected_sha256:
                raise RuntimeError(
                    "loaded Torch-MIGraphX extension differs from the selected binary"
                )
            return loaded_path
    raise RuntimeError(
        "no already-built Torch-MIGraphX extension matches the B1 manifest; "
        "loader-side JIT compilation is forbidden"
    )


def _get_inference_generator(model: torch.nn.Module) -> torch.nn.Module:
    generator_ema = getattr(model, "generator_ema", None)
    if generator_ema is not None:
        return generator_ema
    return model.generator


def _validate_manifest(
    directory: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    manifest_path = directory / "B1_COLD_MANIFEST.json"
    manifest_sha_path = directory / "B1_COLD_MANIFEST.sha256"
    if not manifest_path.is_file() or not manifest_sha_path.is_file():
        raise FileNotFoundError(
            f"B1 artifact manifest or SHA256 sidecar is missing from {directory}"
        )
    expected_manifest_sha = manifest_sha_path.read_text(encoding="utf-8").strip()
    if _sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError("B1 artifact manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "WRITTEN_NOT_LOADED"
        or manifest.get("static_batch") != 1
        or manifest.get("dtype") != "torch.float16"
        or manifest.get("direction_order") != list(_DIRECTIONS)
        or manifest.get("input_order") != list(_INPUT_NAMES)
    ):
        raise RuntimeError("B1 artifact manifest contract is not accepted")

    frozen_files = manifest.get("frozen_files", {})
    expected_checkpoint = next(
        (
            value
            for name, value in frozen_files.items()
            if Path(name).name == checkpoint_path.name
        ),
        None,
    )
    if expected_checkpoint is None or _sha256(checkpoint_path) != expected_checkpoint:
        raise RuntimeError(
            "current restoration checkpoint differs from the B1 artifact manifest"
        )

    repository_root = Path(__file__).resolve().parents[2]
    for relative, expected in _SEMANTIC_SOURCE_HASHES.items():
        path = repository_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(
                f"current semantic source differs from the B1 artifact: {relative}"
            )

    artifact_paths: dict[str, Path] = {}
    directions = manifest.get("directions")
    if not isinstance(directions, dict):
        raise RuntimeError("B1 artifact direction manifest is missing")
    for direction in _DIRECTIONS:
        direction_row = directions.get(direction)
        if not isinstance(direction_row, dict):
            raise RuntimeError(f"B1 artifact direction is missing: {direction}")
        expected_contract = []
        for position, (name, shape) in enumerate(
            zip(_INPUT_NAMES, _artifact_shapes(direction), strict=True)
        ):
            expected_contract.append(
                {
                    "position": position,
                    "name": name,
                    "shape": list(shape),
                    "dtype": "torch.float16",
                    "device": "cuda:0",
                    "strides": list(_canonical_strides(shape)),
                    "storage_offset": 0,
                }
            )
        if direction_row.get("input_contract") != expected_contract:
            raise RuntimeError(f"B1 artifact ABI manifest mismatch: {direction}")
        artifact = direction_row.get("artifact")
        if not isinstance(artifact, dict):
            raise RuntimeError(f"B1 artifact identity is missing: {direction}")
        path = directory / str(artifact.get("name", ""))
        if (
            not path.is_file()
            or path.stat().st_size != artifact.get("size_bytes")
            or _sha256(path) != artifact.get("sha256")
        ):
            raise RuntimeError(f"B1 artifact identity mismatch: {direction}")
        artifact_paths[direction] = path
    return manifest, artifact_paths


def _validate_runtime(
    manifest: dict[str, object],
    directory: Path,
    device: torch.device,
) -> Path:
    expected = manifest.get("runtime")
    expected_gpu = manifest.get("gpu")
    if not isinstance(expected, dict) or not isinstance(expected_gpu, dict):
        raise RuntimeError("B1 artifact runtime/GPU manifest is missing")
    actual = {
        "torch": torch.__version__,
        "hip": str(torch.version.hip),
        "torch_migraphx": _package_version("torch-migraphx"),
        "migraphx": _package_version("migraphx"),
        "gpu_name": torch.cuda.get_device_name(device),
        "architecture": str(
            getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        ),
    }
    for name in ("torch", "hip", "torch_migraphx", "migraphx"):
        if actual[name] != expected.get(name):
            raise RuntimeError(
                f"B1 artifact runtime mismatch for {name}: "
                f"actual={actual[name]!r}, expected={expected.get(name)!r}"
            )
    if actual["gpu_name"] != expected_gpu.get("name"):
        raise RuntimeError("B1 artifact GPU name mismatch")
    if actual["architecture"] != expected_gpu.get("architecture"):
        raise RuntimeError("B1 artifact GPU architecture mismatch")

    expected_extension_sha = str(expected.get("extension_sha256", ""))
    extension = _preload_torch_migraphx_extension(
        directory, expected_extension_sha
    )
    _import_migraphx()
    importlib.import_module("torch_migraphx")
    if _sha256(extension) != expected_extension_sha:
        raise RuntimeError("B1 artifact Torch-MIGraphX extension SHA256 mismatch")
    return extension


class _StrictArtifact:
    """Validate the static B1 ABI and own every reused artifact output."""

    def __init__(
        self,
        direction: str,
        target: Callable[..., object],
        device: torch.device,
    ) -> None:
        self.direction = direction
        self.target = target
        self.device = device

    def __call__(self, *values: torch.Tensor) -> torch.Tensor:
        expected_shapes = _artifact_shapes(self.direction)
        if len(values) != len(expected_shapes):
            raise RuntimeError(f"{self.direction}: expected eight inputs")
        for name, value, shape in zip(
            _INPUT_NAMES, values, expected_shapes, strict=True
        ):
            if (
                value.device != self.device
                or value.dtype != torch.float16
                or tuple(value.shape) != shape
                or tuple(value.stride()) != _canonical_strides(shape)
                or value.storage_offset() != 0
            ):
                raise RuntimeError(
                    f"{self.direction}/{name}: incompatible artifact ABI: "
                    f"shape={tuple(value.shape)} stride={tuple(value.stride())} "
                    f"dtype={value.dtype} device={value.device} "
                    f"offset={value.storage_offset()}"
                )
        raw = self.target(*values)
        if isinstance(raw, torch.Tensor):
            output = raw
        elif type(raw) is tuple and len(raw) == 1 and isinstance(raw[0], torch.Tensor):
            output = raw[0]
        else:
            raise RuntimeError(
                f"{self.direction}: unsupported artifact output {type(raw)!r}"
            )
        if (
            output.shape != (1, 64, 64, 64)
            or output.dtype != torch.float16
            or output.device != self.device
        ):
            raise RuntimeError(f"{self.direction}: incompatible artifact output")
        return output.clone()


def _load_artifacts(
    paths: dict[str, Path], device: torch.device
) -> tuple[dict[str, _StrictArtifact], dict[str, object]]:
    from torch_migraphx.fx.mgx_module import MGXModule

    wrappers: dict[str, _StrictArtifact] = {}
    topology: dict[str, object] = {}
    original_initialize = MGXModule._initialize

    def reject_uncompiled_restore(
        instance: MGXModule, *args: object, **kwargs: object
    ):
        program = getattr(instance, "program", None)
        if program is None or not program.is_compiled():
            raise RuntimeError(
                "refusing a B1 artifact that needs loader-side compilation"
            )
        return original_initialize(instance, *args, **kwargs)

    MGXModule._initialize = reject_uncompiled_restore
    try:
        for direction in _DIRECTIONS:
            loaded = torch.load(paths[direction], weights_only=False)
            if not isinstance(loaded, torch.fx.GraphModule):
                raise RuntimeError(f"{direction}: artifact is not a GraphModule")
            tracer = repr(getattr(loaded, "_tracer_cls", None))
            if "PythonKeyTracer" in tracer:
                raise RuntimeError(
                    f"{direction}: unsafe tracer survived artifact loading"
                )
            mgx_modules = []
            for name, child in loaded.named_modules():
                if type(child).__name__ == "MGXModule":
                    program = getattr(child, "program", None)
                    if program is None or not program.is_compiled():
                        raise RuntimeError(
                            f"{direction}/{name}: MIGraphX program is not compiled"
                        )
                    mgx_modules.append(name)
            graph_nodes = len(list(loaded.graph.nodes))
            if graph_nodes != 28:
                raise RuntimeError(
                    f"{direction}: unexpected root graph node count {graph_nodes}"
                )
            if mgx_modules != ["fused_0", "fused_1", "fused_2"]:
                raise RuntimeError(
                    f"{direction}: unexpected MIGraphX topology {mgx_modules}"
                )
            topology[direction] = {
                "graph_nodes": graph_nodes,
                "migraphx_modules": tuple(mgx_modules),
            }
            wrappers[direction] = _StrictArtifact(direction, loaded, device)
    finally:
        MGXModule._initialize = original_initialize
    return wrappers, topology


class BasicvsrppB1MigraphxPropagation:
    """Source-equivalent propagation with only the repeated body replaced."""

    def __init__(
        self,
        generator: torch.nn.Module,
        artifacts: dict[str, _StrictArtifact],
        *,
        directory: Path,
        extension_path: Path,
        topology: dict[str, object],
    ) -> None:
        if set(artifacts) != set(_DIRECTIONS):
            raise RuntimeError("all four B1 propagation artifacts are required")
        self.generator = generator
        self.artifacts = artifacts
        self.directory = directory
        self.extension_path = extension_path
        self.topology = topology

    @staticmethod
    def _flow_grid(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        n, channels, height, width = feature.shape
        theta = (
            torch.eye(2, 3, device=feature.device, dtype=feature.dtype)
            .unsqueeze(0)
            .expand(n, -1, -1)
        )
        grid = F.affine_grid(
            theta, (n, channels, height, width), align_corners=True
        )
        flow_nhwc = flow.permute(0, 2, 3, 1)
        flow_x = flow_nhwc[..., 0] * (2.0 / max(width - 1, 1))
        flow_y = flow_nhwc[..., 1] * (2.0 / max(height - 1, 1))
        return grid + torch.stack((flow_x, flow_y), dim=-1)

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        if module_name not in _DIRECTIONS:
            raise RuntimeError(f"unexpected propagation direction: {module_name}")
        _n, temporal_minus_one, _channels, height, width = flows.size()
        frame_idx = list(range(temporal_minus_one + 1))
        flow_idx = list(range(-1, temporal_minus_one))
        mapping_idx = list(range(len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(
            1, self.generator.mid_channels, height, width
        )
        for index, frame_position in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[frame_position]]
            if index > 0:
                flow_n1 = flows[:, flow_idx[index], :, :, :]
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                if index > 1:
                    feat_n2 = feats[module_name][-2]
                    prior_flow = flows[:, flow_idx[index - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(
                        prior_flow, flow_n1.permute(0, 2, 3, 1)
                    )
                prefix = torch.cat(
                    [feat_current]
                    + [
                        feats[key][frame_position]
                        for key in feats
                        if key not in ["spatial", module_name]
                    ],
                    dim=1,
                )
                artifact_inputs = tuple(
                    _canonicalize(value.contiguous())
                    for value in (
                        feat_prop,
                        self._flow_grid(feat_prop, flow_n1),
                        feat_n2,
                        self._flow_grid(feat_n2, flow_n2),
                        feat_current,
                        flow_n1,
                        flow_n2,
                        prefix,
                    )
                )
                # _StrictArtifact already clones the reused MIGraphX output
                # immediately; keep that single ownership boundary here.
                feat_prop = self.artifacts[module_name](*artifact_inputs)
            else:
                feat = [feat_current] + [
                    feats[key][frame_position]
                    for key in feats
                    if key not in ["spatial", module_name]
                ] + [feat_prop]
                feat_prop = feat_prop + self.generator.backbone[module_name](
                    torch.cat(feat, dim=1)
                )
            feats[module_name].append(feat_prop)

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    @contextmanager
    def dispatch(self) -> Iterator[None]:
        original = self.generator.propagate

        def dispatched(
            _self: torch.nn.Module,
            feats: dict[str, list[torch.Tensor]],
            flows: torch.Tensor,
            module_name: str,
        ) -> dict[str, list[torch.Tensor]]:
            return self.propagate(feats, flows, module_name)

        self.generator.propagate = types.MethodType(dispatched, self.generator)
        try:
            yield
        finally:
            self.generator.propagate = original

    def close(self) -> None:
        self.artifacts.clear()
        self.topology.clear()


def load_basicvsrpp_b1_migraphx(
    model: torch.nn.Module,
    *,
    checkpoint_path: str | Path,
    device: torch.device,
) -> BasicvsrppB1MigraphxPropagation:
    directory = artifact_directory(checkpoint_path)
    checkpoint = Path(checkpoint_path).resolve()
    manifest, paths = _validate_manifest(directory, checkpoint)
    extension = _validate_runtime(manifest, directory, device)
    artifacts, topology = _load_artifacts(paths, device)
    generator = _get_inference_generator(model).eval()
    if not getattr(model, "is_use_ema", False) or generator is not getattr(
        model, "generator_ema", None
    ):
        raise RuntimeError("current BasicVSR++ model did not select the expected EMA generator")
    return BasicvsrppB1MigraphxPropagation(
        generator,
        artifacts,
        directory=directory,
        extension_path=extension,
        topology=topology,
    )
