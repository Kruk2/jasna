#!/usr/bin/env python3
"""Isolated current-tree probe for accepted BasicVSR++ B1 MIGraphX artifacts.

The probe never changes product routing.  It loads the current checkpoint and
model sources, dispatches only the four ``i > 0`` propagation loop bodies to
precompiled B1 artifacts, and compares that path with the current eager model on
the same real 256x256 ROI sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jasna.accelerator import is_amd_device
from jasna.migraphx_artifact import _import_migraphx
from jasna.models.basicvsrpp.inference import load_model
from jasna.models.basicvsrpp.mmagic.flow_warp import flow_warp
DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")
INPUT_NAMES = (
    "feat_prop",
    "grid_n1",
    "feat_n2",
    "grid_n2",
    "feat_current",
    "flow_n1",
    "flow_n2",
    "backbone_prefix",
)
PREFIX_CHANNELS = {
    "backward_1": 64,
    "forward_1": 128,
    "backward_2": 192,
    "forward_2": 256,
}
SEMANTIC_SOURCE_HASHES = {
    "jasna/models/basicvsrpp/inference.py": "99b7234dab1a986393ff71658835cb0bad967221fc9aaa0887a82ef2d3ad5f37",
    "jasna/models/basicvsrpp/mmagic/base_edit_model.py": "7cbd31895ff4a31d57277b8c9d0f9b5cdf6871a05d33e48419deac0932634616",
    "jasna/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py": "ecb93a88abb92a46958321f33186f25b7e5a4608b526ebe20b4f8b54aec297f9",
    "jasna/models/basicvsrpp/mmagic/flow_warp.py": "e297e303453be80f44edc91e8a5cac6efa84dd66a88f420902c0f8a054ca0462",
    "jasna/models/basicvsrpp/mmagic/real_basicvsr.py": "1fd2686a5267f39f8add38ebd1b88e3a26427fc163003dd127d4379a5ac78e01",
    "jasna/restorer/basicvsrpp_sub_engines.py": "1c413287ce50e67d8ee81f434dad6fb77ced7ca8e96b2e36b913305e23502962",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_inference_generator(model: torch.nn.Module) -> torch.nn.Module:
    """Mirror the tiny product helper without importing TensorRT on AMD."""

    generator_ema = getattr(model, "generator_ema", None)
    if generator_ema is not None:
        return generator_ema
    return model.generator


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _preload_torch_migraphx_extension(expected_sha256: str) -> Path:
    """Load only the already-built, manifest-matched native pointer bridge."""

    candidates: list[Path] = []
    configured = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    roots = [Path(configured)] if configured else []
    roots.append(Path.home() / ".cache" / "torch_extensions")
    for root in roots:
        if root.is_dir():
            candidates.extend(
                sorted(root.glob("*/_torch_migraphx/_torch_migraphx*.so"))
            )
    for extension in candidates:
        if extension.is_file() and _sha256(extension) == expected_sha256:
            sys.path.insert(0, str(extension.parent))
            importlib.invalidate_caches()
            loaded = importlib.import_module("_torch_migraphx")
            loaded_path = Path(loaded.__file__).resolve()
            if loaded_path != extension.resolve():
                raise RuntimeError(
                    "loaded Torch-MIGraphX extension differs from the selected binary"
                )
            return loaded_path
    raise RuntimeError(
        "no already-built Torch-MIGraphX extension matches the artifact manifest; "
        "JIT compilation is intentionally forbidden"
    )


def _canonical_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for size in reversed(shape):
        reversed_strides.append(stride)
        stride *= size
    return tuple(reversed(reversed_strides))


def _canonicalize(value: torch.Tensor) -> torch.Tensor:
    expected = _canonical_strides(tuple(int(size) for size in value.shape))
    if tuple(int(stride) for stride in value.stride()) != expected:
        value = value.clone(memory_format=torch.contiguous_format)
    if tuple(int(stride) for stride in value.stride()) != expected:
        raise RuntimeError(f"could not canonicalize tensor strides: {value.stride()} != {expected}")
    return value


def _artifact_shapes(direction: str) -> tuple[tuple[int, ...], ...]:
    prefix = PREFIX_CHANNELS[direction]
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


class StrictArtifact:
    """Validate the static B1 ABI and never expose a reused artifact output."""

    def __init__(self, direction: str, target: Callable[..., object], device: torch.device):
        self.direction = direction
        self.target = target
        self.device = device
        self.calls = 0

    def __call__(self, *values: torch.Tensor) -> torch.Tensor:
        expected_shapes = _artifact_shapes(self.direction)
        if len(values) != len(expected_shapes):
            raise RuntimeError(f"{self.direction}: expected eight inputs")
        for name, value, shape in zip(INPUT_NAMES, values, expected_shapes, strict=True):
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
                    f"dtype={value.dtype} device={value.device} offset={value.storage_offset()}"
                )
        raw = self.target(*values)
        if isinstance(raw, torch.Tensor):
            output = raw
        elif type(raw) is tuple and len(raw) == 1 and isinstance(raw[0], torch.Tensor):
            output = raw[0]
        else:
            raise RuntimeError(f"{self.direction}: unsupported artifact output {type(raw)!r}")
        if (
            output.shape != (1, 64, 64, 64)
            or output.dtype != torch.float16
            or output.device != self.device
        ):
            raise RuntimeError(f"{self.direction}: incompatible artifact output")
        self.calls += 1
        return output.clone()


class B1MigraphxPropagation:
    """Source-equivalent propagation with only the repeated body replaced."""

    def __init__(self, generator: torch.nn.Module, artifacts: dict[str, StrictArtifact]):
        if set(artifacts) != set(DIRECTIONS):
            raise RuntimeError("all four B1 propagation artifacts are required")
        self.generator = generator
        self.artifacts = artifacts
        self.propagate_calls = {direction: 0 for direction in DIRECTIONS}

    @staticmethod
    def _flow_grid(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        n, channels, height, width = feature.shape
        theta = (
            torch.eye(2, 3, device=feature.device, dtype=feature.dtype)
            .unsqueeze(0)
            .expand(n, -1, -1)
        )
        grid = F.affine_grid(theta, (n, channels, height, width), align_corners=True)
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
        if module_name not in DIRECTIONS:
            raise RuntimeError(f"unexpected propagation direction: {module_name}")
        _n, temporal_minus_one, _channels, height, width = flows.size()
        frame_idx = list(range(temporal_minus_one + 1))
        flow_idx = list(range(-1, temporal_minus_one))
        mapping_idx = list(range(len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(1, self.generator.mid_channels, height, width)
        self.propagate_calls[module_name] += 1
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
                # StrictArtifact already takes ownership of the reused output.
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
def _dispatch(generator: torch.nn.Module, replacement: B1MigraphxPropagation) -> Iterator[None]:
    original = generator.propagate

    def dispatched(
        _self: torch.nn.Module,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        return replacement.propagate(feats, flows, module_name)

    generator.propagate = types.MethodType(dispatched, generator)
    try:
        yield
    finally:
        generator.propagate = original


def _decode_roi(path: Path, frame_count: int) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        "crop=256:256:(iw-256)/2:(ih-256)/2,setpts=PTS-STARTPTS",
        "-frames:v",
        str(frame_count),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    expected = frame_count * 256 * 256 * 3
    if len(completed.stdout) != expected:
        raise RuntimeError(f"decoded {len(completed.stdout)} bytes, expected {expected}")
    array = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        frame_count, 256, 256, 3
    )
    return torch.from_numpy(array.copy()).permute(0, 3, 1, 2).contiguous()


def _validate_manifest(
    directory: Path,
    weights: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    manifest_path = directory / "B1_COLD_MANIFEST.json"
    manifest_sha_path = directory / "B1_COLD_MANIFEST.sha256"
    if not manifest_path.is_file() or not manifest_sha_path.is_file():
        raise FileNotFoundError("B1 artifact manifest or SHA256 sidecar is missing")
    expected_manifest_sha = manifest_sha_path.read_text(encoding="utf-8").strip()
    if _sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError("B1 artifact manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "WRITTEN_NOT_LOADED"
        or manifest.get("static_batch") != 1
        or manifest.get("dtype") != "torch.float16"
        or manifest.get("direction_order") != list(DIRECTIONS)
        or manifest.get("input_order") != list(INPUT_NAMES)
    ):
        raise RuntimeError("B1 artifact manifest contract is not accepted")
    frozen = manifest.get("frozen_files", {})
    expected_weights = next(
        (
            value
            for name, value in frozen.items()
            if Path(name).name == "lada_mosaic_restoration_model_generic_v1.2.pth"
        ),
        None,
    )
    if _sha256(weights) != expected_weights:
        raise RuntimeError("current restoration checkpoint differs from the artifact manifest")
    for relative, expected in SEMANTIC_SOURCE_HASHES.items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"current semantic source differs from the artifact: {relative}")
    artifact_paths: dict[str, Path] = {}
    for direction in DIRECTIONS:
        direction_row = manifest["directions"][direction]
        expected_contract = []
        for position, (name, shape) in enumerate(
            zip(INPUT_NAMES, _artifact_shapes(direction), strict=True)
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
            raise RuntimeError(f"artifact ABI manifest mismatch: {direction}")
        row = direction_row["artifact"]
        path = directory / row["name"]
        if path.stat().st_size != row["size_bytes"] or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"artifact identity mismatch: {direction}")
        artifact_paths[direction] = path
    return manifest, artifact_paths


def _runtime_gate(manifest: dict[str, object], device: torch.device) -> dict[str, str]:
    expected = manifest["runtime"]
    expected_gpu = manifest["gpu"]
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
        if actual[name] != expected[name]:
            raise RuntimeError(
                f"B1 artifact runtime mismatch for {name}: "
                f"actual={actual[name]!r}, expected={expected[name]!r}"
            )
    if actual["gpu_name"] != expected_gpu["name"]:
        raise RuntimeError("B1 artifact GPU name mismatch")
    if actual["architecture"] != expected_gpu["architecture"]:
        raise RuntimeError("B1 artifact GPU architecture mismatch")

    # Bind the exact native pointer bridge before importing torch_migraphx.
    # Its upstream fallback JIT is intentionally bypassed: a benchmark must not
    # mutate the runtime or silently test a newly compiled extension.
    extension = _preload_torch_migraphx_extension(expected["extension_sha256"])
    # Match the product's official ROCm ABI-aware discovery: distro packages
    # install the binding in /opt/rocm/lib rather than this venv's site-packages.
    _import_migraphx()
    import torch_migraphx  # noqa: F401
    extension_sha = _sha256(extension)
    if extension_sha != expected["extension_sha256"]:
        raise RuntimeError(
            "B1 artifact Torch-MIGraphX extension SHA256 mismatch: "
            f"{extension_sha}"
        )
    actual["extension_path"] = str(extension)
    actual["extension_sha256"] = extension_sha
    return actual


def _load_artifacts(
    paths: dict[str, Path], device: torch.device
) -> tuple[dict[str, StrictArtifact], dict[str, object]]:
    from torch_migraphx.fx.mgx_module import MGXModule

    wrappers: dict[str, StrictArtifact] = {}
    topology: dict[str, object] = {}
    original_initialize = MGXModule._initialize

    def reject_uncompiled_restore(instance: MGXModule, *args: object, **kwargs: object):
        program = getattr(instance, "program", None)
        if program is None or not program.is_compiled():
            raise RuntimeError("refusing a B1 artifact that needs loader-side compilation")
        return original_initialize(instance, *args, **kwargs)

    MGXModule._initialize = reject_uncompiled_restore
    try:
        for direction in DIRECTIONS:
            loaded = torch.load(paths[direction], weights_only=False)
            if not isinstance(loaded, torch.fx.GraphModule):
                raise RuntimeError(f"{direction}: artifact is not a GraphModule")
            tracer = repr(getattr(loaded, "_tracer_cls", None))
            if "PythonKeyTracer" in tracer:
                raise RuntimeError(f"{direction}: unsafe tracer survived artifact loading")
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
                "mgx_modules": mgx_modules,
            }
            wrappers[direction] = StrictArtifact(direction, loaded, device)
    finally:
        MGXModule._initialize = original_initialize
    return wrappers, topology


def _timed_call(
    model: torch.nn.Module,
    value: torch.Tensor,
    dispatch: B1MigraphxPropagation | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    torch.cuda.synchronize()
    before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    with torch.inference_mode():
        if dispatch is None:
            output = model(inputs=value)
        else:
            with _dispatch(_get_inference_generator(model), dispatch):
                output = model(inputs=value)
    end.record()
    end.synchronize()
    return output, {
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "device_ms": float(start.elapsed_time(end)),
        "peak_allocated_delta_bytes": float(
            max(0, int(torch.cuda.max_memory_allocated()) - before)
        ),
    }


def _summary(values: list[dict[str, float]]) -> dict[str, object]:
    return {
        "samples": values,
        "wall_ms_median": statistics.median(row["wall_ms"] for row in values),
        "device_ms_median": statistics.median(row["device_ms"] for row in values),
        "peak_allocated_delta_max": max(
            row["peak_allocated_delta_bytes"] for row in values
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "model_weights/lada_mosaic_restoration_model_generic_v1.2.pth",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 2 or args.rounds <= 0:
        raise ValueError("--frames must be >= 2 and --rounds must be positive")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not is_amd_device(device):
        raise RuntimeError("this probe requires a ROCm-backed AMD GPU")
    torch.cuda.set_device(device)
    manifest, artifact_paths = _validate_manifest(args.artifact_dir, args.weights)
    runtime = _runtime_gate(manifest, device)
    decoded = _decode_roi(args.input, args.frames)
    value = decoded.unsqueeze(0).to(device=device, dtype=torch.float16).div_(255.0)
    del decoded

    model = load_model(None, str(args.weights), device, fp16=True).eval()
    generator = _get_inference_generator(model).eval()
    if not getattr(model, "is_use_ema", False) or generator is not getattr(
        model, "generator_ema", None
    ):
        raise RuntimeError("current model did not select the expected EMA generator")

    resident_before_artifacts = int(torch.cuda.memory_allocated(device))
    artifacts, topology = _load_artifacts(artifact_paths, device)
    resident_after_artifacts = int(torch.cuda.memory_allocated(device))
    candidate = B1MigraphxPropagation(generator, artifacts)

    eager_output, _ = _timed_call(model, value, None)
    candidate_output, _ = _timed_call(model, value, candidate)
    difference = (eager_output.float() - candidate_output.float()).abs()
    output_gate = {
        "shape_equal": tuple(eager_output.shape) == tuple(candidate_output.shape),
        "finite": bool(torch.isfinite(candidate_output).all().item()),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "allclose_atol_0_01_rtol_0_01": bool(
            torch.allclose(eager_output, candidate_output, atol=0.01, rtol=0.01)
        ),
    }
    del eager_output, candidate_output, difference
    if not all(
        output_gate[name]
        for name in ("shape_equal", "finite", "allclose_atol_0_01_rtol_0_01")
    ):
        raise RuntimeError(f"B1 output parity gate failed: {output_gate}")

    eager_samples: list[dict[str, float]] = []
    candidate_samples: list[dict[str, float]] = []
    for round_index in range(args.rounds):
        order = (None, candidate) if round_index % 2 == 0 else (candidate, None)
        for selected in order:
            output, timing = _timed_call(model, value, selected)
            del output
            (candidate_samples if selected is candidate else eager_samples).append(timing)

    expected_calls = (args.frames - 1) * (args.rounds + 1)
    artifact_calls = {name: artifact.calls for name, artifact in artifacts.items()}
    if any(calls != expected_calls for calls in artifact_calls.values()):
        raise RuntimeError(
            f"B1 artifact call counts differ from {expected_calls}: {artifact_calls}"
        )
    eager = _summary(eager_samples)
    migrated = _summary(candidate_samples)
    report = {
        "schema": "jasna.basicvsrpp-b1-migraphx-current-tree-probe.v1",
        "device": torch.cuda.get_device_name(device),
        "architecture": str(
            getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        ),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "manifest_runtime": manifest["runtime"],
        "actual_runtime": runtime,
        "frames": args.frames,
        "rounds": args.rounds,
        "output_gate": output_gate,
        "topology": topology,
        "artifact_calls": artifact_calls,
        "artifact_resident_torch_bytes": resident_after_artifacts - resident_before_artifacts,
        "eager": eager,
        "migraphx": migrated,
        "wall_speedup": eager["wall_ms_median"] / migrated["wall_ms_median"],
        "device_speedup": eager["device_ms_median"] / migrated["device_ms_median"],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
