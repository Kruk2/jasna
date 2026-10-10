#!/usr/bin/env python3
"""Explicit offline B1 compiler; never installs artifacts or changes defaults.

The product cold-loader must validate these artifacts in a separate process.
Compilation/zero-input parity alone is not real-video acceptance.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory only; existing directories are never overwritten")
    return parser


def snapshot_sources(checkpoint: Path, expected: dict[str, str]) -> dict[str, str]:
    sources = {name: sha256_file(ROOT / name) for name in expected}
    if sources != expected:
        raise RuntimeError("semantic source differs from the reviewed product contract")
    sources[str(checkpoint)] = sha256_file(checkpoint)
    return sources


def main() -> None:
    args = build_parser().parse_args()
    # Torch must initialize its dependency graph before MIGraphX. Require an
    # already-built native extension before importing Torch-MIGraphX; this
    # writer may compile graphs, but must not silently JIT the pointer bridge.
    import torch
    import _torch_migraphx
    import migraphx  # noqa: F401
    import torch_migraphx  # noqa: F401
    import torch_migraphx.dynamo.backends as backends

    from jasna.models.basicvsrpp.inference import load_model
    from jasna.restorer.basicvsrpp_sub_engines import _PropagateBodyWrapper
    from jasna.restorer import basicvsrpp_migraphx_b1 as contract

    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.name != "lada_mosaic_restoration_model_generic_v1.2.pth":
        raise RuntimeError("only the product generic v1.2 checkpoint is supported")
    sources = snapshot_sources(checkpoint, contract._SEMANTIC_SOURCE_HASHES)
    device = torch.device("cuda:0")
    if sys.platform != "linux" or not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("B1 compiler requires an available Linux AMD/ROCm GPU")
    architecture = torch.cuda.get_device_properties(device).gcnArchName
    if architecture.split(":")[0] != "gfx1100":
        raise RuntimeError(f"unproven compiler device: {architecture}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = load_model(None, str(checkpoint), device, fp16=True).eval()
    generator = contract._get_inference_generator(model).eval()
    if not model.is_use_ema or generator is not model.generator_ema:
        raise RuntimeError("compiler did not select the EMA generator")
    rows = {}
    original = backends.lower_aten_to_mgx
    captured = []

    def capture(*values, **kwargs):
        lowered = original(*values, **kwargs)
        if not isinstance(lowered, torch.fx.GraphModule):
            raise RuntimeError("lowerer did not return GraphModule")
        if getattr(lowered, "_tracer_cls", None) is not None:
            raise RuntimeError("upstream tracer serialization fix is missing")
        captured.append(lowered)
        return lowered

    backends.lower_aten_to_mgx = capture
    try:
        for direction in contract._DIRECTIONS:
            torch._dynamo.reset()
            captured.clear()
            artifact = output / f"b1_{direction}.torch"
            wrapper = _PropagateBodyWrapper(
                generator.deform_align[direction], generator.backbone[direction]
            ).eval()
            started = time.perf_counter()
            print(f"compiling {direction}", flush=True)
            with torch.inference_mode():
                values = tuple(torch.zeros(shape, device=device, dtype=torch.float16)
                               for shape in contract._artifact_shapes(direction))
                compiled = torch.compile(
                    wrapper, backend="migraphx", fullgraph=True, dynamic=False,
                    options={"fp16": True, "exhaustive_tune": False,
                             "save_compiled": str(artifact)},
                )
                actual = compiled(*values).clone()
                torch.cuda.synchronize()
                expected = wrapper(*values).clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            if len(captured) != 1 or not artifact.is_file():
                raise RuntimeError("compile/save count mismatch")
            lowered = captured[0]
            children = [(name, child) for name, child in lowered.named_modules()
                        if type(child).__name__ == "MGXModule"]
            node_count = len(list(lowered.graph.nodes))
            if (node_count != 28
                    or [name for name, _ in children] != ["fused_0", "fused_1", "fused_2"]
                    or not all(child.program.is_compiled() for _, child in children)):
                raise RuntimeError("compiled graph violates the product topology contract")
            rows[direction] = {
                "artifact": {"name": artifact.name, "size_bytes": artifact.stat().st_size,
                             "sha256": sha256_file(artifact)},
                "input_contract": [
                    {"position": i, "name": name, "shape": list(shape),
                     "dtype": "torch.float16", "device": "cuda:0",
                     "strides": list(contract._canonical_strides(shape)), "storage_offset": 0}
                    for i, (name, shape) in enumerate(zip(
                        contract._INPUT_NAMES, contract._artifact_shapes(direction), strict=True
                    ))
                ],
                "topology": {"root_graph_node_count": node_count,
                             "mgx_module_names": [name for name, _ in children]},
                "compile_seconds": time.perf_counter() - started,
                "zero_input_max_abs": float((actual.float() - expected.float()).abs().max()),
            }
            print(json.dumps({"direction": direction, **rows[direction]}), flush=True)
            (output / "writer-progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            del wrapper, compiled, actual, expected, values, lowered, children
            captured.clear()
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        backends.lower_aten_to_mgx = original
    if snapshot_sources(checkpoint, contract._SEMANTIC_SOURCE_HASHES) != sources:
        raise RuntimeError("source/checkpoint changed during compilation")
    manifest = {
        "schema": 1, "status": "WRITTEN_NOT_LOADED", "static_batch": 1,
        "dtype": "torch.float16", "direction_order": list(contract._DIRECTIONS),
        "input_order": list(contract._INPUT_NAMES), "directions": rows, "frozen_files": sources,
        "runtime": {"torch": torch.__version__, "hip": torch.version.hip,
                    "torch_migraphx": importlib.metadata.version("torch-migraphx"),
                    "migraphx": importlib.metadata.version("migraphx"),
                    "extension_sha256": sha256_file(Path(_torch_migraphx.__file__))},
        "gpu": {"name": torch.cuda.get_device_name(device), "architecture": architecture},
    }
    path = output / "B1_COLD_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "B1_COLD_MANIFEST.sha256").write_text(sha256_file(path) + "\n")
    print("written, not promoted: separate cold-load and real-video acceptance required", flush=True)


if __name__ == "__main__":
    main()
