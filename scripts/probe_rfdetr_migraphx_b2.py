#!/usr/bin/env python3
"""Build and benchmark a transaction-only RF-DETR MIGraphX static-B2 artifact.

The product artifact and its manifest are never modified.  The probe recreates
the accepted static graph compatibility rewrites, applies the same
dot-projector mixed precision policy as the product B1 artifact, and compares
one B2 dispatch with two sequential B1 dispatches on identical inputs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable


PURPOSE = "jasna-rfdetr-v6-medium-segmentation"
INTERNAL_PRECISION = "mixed-dot-projector-convolution-fp16"
INPUT_NAME = "input"
INPUT_SHAPE = (2, 3, 576, 576)
INPUT_STRIDES = (995328, 331776, 576, 1)
OUTPUTS = (
    ("dets", "main:#output_0", (2, 200, 4), (800, 4, 1)),
    ("labels", "main:#output_1", (2, 200, 3), (600, 3, 1)),
    ("masks", "main:#output_2", (2, 200, 144, 144), (4147200, 20736, 144, 1)),
)
IDENTITY_CONCAT = "/transformer/Concat_8"
RESIZE_NAME = "/segmentation_head/Resize"
RESIZE_INPUT = "/backbone/backbone.0/projector/stages.0/stages.0.1/Transpose_1_output_0"
RESIZE_OUTPUT = "/segmentation_head/Resize_output_0"
NHWC_INPUT = "/segmentation_head/Resize__migraphx_nhwc_input"
NHWC_OUTPUT = "/segmentation_head/Resize__migraphx_nhwc_output"
NHWC_SIZES = "/segmentation_head/Resize__migraphx_nhwc_sizes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dims(value: Any) -> tuple[int | str | None, ...]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return tuple(result)


def assert_static_b2_contract(model: Any) -> None:
    if len(model.graph.input) != 1:
        raise RuntimeError("RF-DETR B2 export must have exactly one input")
    actual_input = (model.graph.input[0].name, _dims(model.graph.input[0]))
    if actual_input != (INPUT_NAME, INPUT_SHAPE):
        raise RuntimeError(f"unexpected B2 input contract: {actual_input!r}")
    actual_outputs = [(value.name, _dims(value)) for value in model.graph.output]
    expected_outputs = [(name, shape) for name, _parameter, shape, _strides in OUTPUTS]
    if actual_outputs != expected_outputs:
        raise RuntimeError(
            f"unexpected B2 output contract: {actual_outputs!r} != {expected_outputs!r}"
        )


def rewrite_identity_concat(model: Any) -> dict[str, Any]:
    matches = [node for node in model.graph.node if node.name == IDENTITY_CONCAT]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {IDENTITY_CONCAT}, found {len(matches)}")
    node = matches[0]
    if node.op_type != "Concat" or len(node.input) != 2 or len(node.output) != 1:
        raise RuntimeError(f"unexpected identity concat contract: {node}")
    kept_input = node.input[0]
    dropped_input = node.input[1]
    node.op_type = "Identity"
    node.ClearField("input")
    node.input.append(kept_input)
    node.ClearField("attribute")
    return {"node": IDENTITY_CONCAT, "kept_input": kept_input, "dropped_input": dropped_input}


def rewrite_resize_nhwc(model: Any) -> dict[str, Any]:
    import numpy as np
    import onnx

    matches = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.name == RESIZE_NAME and RESIZE_OUTPUT in node.output
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one segmentation resize, found {len(matches)}")
    index, resize = matches[0]
    if list(resize.input) != [
        RESIZE_INPUT,
        "",
        "",
        "/segmentation_head/Concat_output_0",
    ]:
        raise RuntimeError(f"unexpected segmentation resize inputs: {list(resize.input)!r}")

    to_nhwc = onnx.helper.make_node(
        "Transpose",
        [RESIZE_INPUT],
        [NHWC_INPUT],
        name="/segmentation_head/Resize__migraphx_TransposeToNHWC",
        perm=[0, 2, 3, 1],
    )
    nhwc_resize = copy.deepcopy(resize)
    nhwc_resize.name = "/segmentation_head/Resize__migraphx_NHWC"
    nhwc_resize.input[0] = NHWC_INPUT
    nhwc_resize.input[3] = NHWC_SIZES
    nhwc_resize.output[0] = NHWC_OUTPUT
    to_nchw = onnx.helper.make_node(
        "Transpose",
        [NHWC_OUTPUT],
        [RESIZE_OUTPUT],
        name="/segmentation_head/Resize__migraphx_TransposeToNCHW",
        perm=[0, 3, 1, 2],
    )
    nodes = list(model.graph.node)
    nodes[index : index + 1] = [to_nhwc, nhwc_resize, to_nchw]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.asarray([2, 144, 144, 256], dtype=np.int64), name=NHWC_SIZES
        )
    )
    model.graph.value_info.extend(
        [
            onnx.helper.make_tensor_value_info(
                NHWC_INPUT, onnx.TensorProto.FLOAT, [2, 48, 48, 256]
            ),
            onnx.helper.make_tensor_value_info(
                NHWC_OUTPUT, onnx.TensorProto.FLOAT, [2, 144, 144, 256]
            ),
        ]
    )
    return {
        "source_node": RESIZE_NAME,
        "rewrite": "NCHW transpose -> NHWC resize -> NCHW transpose",
        "sizes": [2, 144, 144, 256],
    }


def rewrite_selected_convolutions(
    model: Any,
    *,
    selector: Callable[[str], bool] = lambda name: "/projector/" in name,
    expected_selected: int = 8,
) -> list[str]:
    import onnx

    rewritten = []
    selected: list[str] = []
    for index, original in enumerate(model.graph.node):
        node = copy.deepcopy(original)
        if node.op_type != "Conv" or not selector(node.name):
            rewritten.append(node)
            continue
        selected.append(node.name)
        half_inputs = []
        for input_index, input_name in enumerate(node.input):
            cast_output = f"{input_name}__dot-projector_conv{index}_input{input_index}_fp16"
            rewritten.append(
                onnx.helper.make_node(
                    "Cast",
                    [input_name],
                    [cast_output],
                    name=f"dot-projector.conv{index}.input{input_index}.to_fp16",
                    to=onnx.TensorProto.FLOAT16,
                )
            )
            half_inputs.append(cast_output)
        if len(node.output) != 1:
            raise RuntimeError(f"selected Conv has unexpected outputs: {node.name}")
        public_output = node.output[0]
        half_output = f"{public_output}__dot-projector_fp16"
        del node.input[:]
        node.input.extend(half_inputs)
        node.output[0] = half_output
        rewritten.append(node)
        rewritten.append(
            onnx.helper.make_node(
                "Cast",
                [half_output],
                [public_output],
                name=f"dot-projector.conv{index}.output.to_fp32",
                to=onnx.TensorProto.FLOAT,
            )
        )
    if len(selected) != expected_selected:
        raise RuntimeError(
            f"selected {len(selected)} projector convolutions, expected {expected_selected}"
        )
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return selected


def optimize_onnx_basic(source: Path, destination: Path) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(destination)
    session = ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [item.name for item in session.get_outputs()] != [item[0] for item in OUTPUTS]:
        raise RuntimeError("ORT BASIC changed RF-DETR output names")


def export_static_b2(weights: Path, output_dir: Path) -> Path:
    import torch
    from rfdetr import RFDETRSegMedium

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    num_classes = int(checkpoint["model"]["class_embed.weight"].shape[0]) - 1
    del checkpoint
    wrapper = RFDETRSegMedium(
        num_classes=num_classes,
        resolution=576,
        pretrain_weights=str(weights),
        device="cpu",
    )
    try:
        exported = wrapper.export(
            output_dir=str(output_dir),
            shape=(576, 576),
            batch_size=2,
            dynamic_batch=False,
            opset_version=17,
            verbose=False,
            format="onnx",
            notes={
                "purpose": "Jasna transaction-only RF-DETR MIGraphX B2 probe",
                "batch_size": 2,
                "product_default": False,
            },
        )
    finally:
        del wrapper
        gc.collect()
    return Path(exported)


def cpu_resize_parity(before: Path, after: Path, input_value: Any) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    names = [item[0] for item in OUTPUTS]
    left = ort.InferenceSession(str(before), providers=["CPUExecutionProvider"])
    right = ort.InferenceSession(str(after), providers=["CPUExecutionProvider"])
    left_outputs = left.run(names, {INPUT_NAME: input_value})
    right_outputs = right.run(names, {INPUT_NAME: input_value})
    report: dict[str, Any] = {}
    for name, expected, actual in zip(names, left_outputs, right_outputs, strict=True):
        delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        report[name] = {
            "exact": bool(np.array_equal(expected, actual)),
            "max_abs": float(delta.max(initial=0.0)),
            "mean_abs": float(delta.mean()),
        }
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"NHWC resize compatibility rewrite changed CPU output {name}")
    return report


def shape_contract(shape: Any) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (
        str(shape.type_string()),
        tuple(int(value) for value in shape.lens()),
        tuple(int(value) for value in shape.strides()),
    )


def build_manifest(
    *, artifact: Path, weights: Path, migraphx_version: str, torch: Any
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "migraphx",
        "purpose": PURPOSE,
        "artifact": {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "source": {"sha256": sha256_file(weights)},
        "target": {
            "platform": "linux",
            "device_arch": "gfx1100",
            "torch_version": str(torch.__version__),
            "torch_hip_version": str(torch.version.hip),
            "migraphx_version": migraphx_version,
            "torch_migraphx_version": importlib.metadata.version("torch-migraphx"),
        },
        "program": {
            "internal_precision": INTERNAL_PRECISION,
            "stream_type": "ihipStream_t",
            "inputs": [
                {
                    "logical_name": INPUT_NAME,
                    "parameter_name": INPUT_NAME,
                    "dtype": "float32",
                    "shape": list(INPUT_SHAPE),
                    "strides": list(INPUT_STRIDES),
                }
            ],
            "outputs": [
                {
                    "logical_name": name,
                    "parameter_name": parameter,
                    "dtype": "float32",
                    "shape": list(shape),
                    "strides": list(strides),
                }
                for name, parameter, shape, strides in OUTPUTS
            ],
        },
    }


def clone_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    return {name: value.clone() for name, value in outputs.items()}


def infer_b1_pair(runner: Any, value: Any, torch: Any) -> dict[str, Any]:
    parts: dict[str, list[Any]] = {name: [] for name, *_rest in OUTPUTS}
    for frame in value.split(1):
        outputs = clone_outputs(runner.infer({INPUT_NAME: frame}))
        for name in parts:
            parts[name].append(outputs[name])
    return {name: torch.cat(values, dim=0) for name, values in parts.items()}


def infer_b2(runner: Any, value: Any) -> dict[str, Any]:
    return clone_outputs(runner.infer({INPUT_NAME: value}))


def benchmark_pair(label: str, function: Callable[[], Any], torch: Any, rounds: int) -> dict[str, Any]:
    for _ in range(4):
        function()
    torch.cuda.synchronize()
    wall_ms: list[float] = []
    device_ms: list[float] = []
    for _ in range(rounds):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        function()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        device_ms.append(float(begin.elapsed_time(end)))
    return {
        "label": label,
        "rounds": rounds,
        "wall_ms": wall_ms,
        "device_ms": device_ms,
        "wall_p50_ms_per_pair": statistics.median(wall_ms),
        "device_p50_ms_per_pair": statistics.median(device_ms),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import migraphx
    import numpy as np
    import onnx
    import torch
    import torch_migraphx  # noqa: F401

    from jasna.migraphx_artifact import MigraphxArtifactRunner

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    weights = args.weights.resolve()
    b1_manifest = args.b1_manifest.resolve()
    if not weights.is_file() or not b1_manifest.is_file():
        raise FileNotFoundError("weights or B1 product manifest is missing")

    stage_times: dict[str, float] = {}
    started = time.perf_counter()
    raw_onnx = export_static_b2(weights, output_dir)
    stage_times["export_seconds"] = time.perf_counter() - started
    raw = onnx.load(str(raw_onnx), load_external_data=False)
    onnx.checker.check_model(raw)
    assert_static_b2_contract(raw)
    concat = rewrite_identity_concat(raw)
    concat_onnx = output_dir / "rfdetr-seg-medium.static-b2.concat-identity.onnx"
    onnx.checker.check_model(raw)
    onnx.save(raw, concat_onnx)
    del raw
    gc.collect()

    basic_onnx = output_dir / "rfdetr-seg-medium.static-b2.concat-identity.ort-basic.onnx"
    started = time.perf_counter()
    optimize_onnx_basic(concat_onnx, basic_onnx)
    stage_times["ort_basic_seconds"] = time.perf_counter() - started
    basic = onnx.load(str(basic_onnx), load_external_data=False)
    onnx.checker.check_model(basic)
    assert_static_b2_contract(basic)

    compatible = copy.deepcopy(basic)
    resize = rewrite_resize_nhwc(compatible)
    compatible_onnx = output_dir / "rfdetr-seg-medium.static-b2.resize-nhwc.onnx"
    onnx.checker.check_model(compatible)
    onnx.save(compatible, compatible_onnx)
    rng = np.random.default_rng(20260905)
    input_host = rng.normal(0.0, 1.0, size=INPUT_SHAPE).astype(np.float32)
    cpu_parity = cpu_resize_parity(basic_onnx, compatible_onnx, input_host)
    del basic

    selected = rewrite_selected_convolutions(compatible)
    onnx.checker.check_model(compatible)
    candidate_onnx = output_dir / "rfdetr-seg-medium.static-b2.dot-projector.onnx"
    onnx.save(compatible, candidate_onnx)
    del compatible
    gc.collect()

    started = time.perf_counter()
    program = migraphx.parse_onnx(str(candidate_onnx))
    stage_times["parse_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    migraphx.quantize_fp16(program, ["dot"])
    stage_times["quantize_seconds"] = time.perf_counter() - started
    graph_text = str(program)
    if "half_type" not in graph_text or "float_type" not in graph_text:
        raise RuntimeError("B2 graph is not mixed precision after dot quantization")
    started = time.perf_counter()
    program.compile(migraphx.get_target("gpu"), offload_copy=False, exhaustive_tune=False)
    stage_times["compile_seconds"] = time.perf_counter() - started
    if not program.is_compiled():
        raise RuntimeError("B2 MIGraphX program did not compile")

    expected_abi = {
        INPUT_NAME: ("float_type", INPUT_SHAPE, INPUT_STRIDES),
        **{
            parameter: ("float_type", shape, strides)
            for _name, parameter, shape, strides in OUTPUTS
        },
    }
    actual_abi = {
        name: shape_contract(shape)
        for name, shape in program.get_parameter_shapes().items()
    }
    if actual_abi != expected_abi:
        raise RuntimeError(f"compiled B2 ABI mismatch: {actual_abi!r} != {expected_abi!r}")

    artifact = output_dir / "rfdetr-seg-medium.static-b2.dot-projector-fp16-gfx1100.mxr"
    migraphx.save(program, str(artifact))
    del program
    manifest = build_manifest(
        artifact=artifact,
        weights=weights,
        migraphx_version=str(migraphx.__version__),
        torch=torch,
    )
    manifest_path = output_dir / "rfdetr-v6.migraphx-b2-gfx1100.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    torch.cuda.set_device(0)
    value = torch.from_numpy(input_host).to("cuda:0")
    b1 = MigraphxArtifactRunner(
        b1_manifest,
        source_path=weights,
        device=torch.device("cuda:0"),
        expected_purpose=PURPOSE,
    )
    b2 = MigraphxArtifactRunner(
        manifest_path,
        source_path=weights,
        device=torch.device("cuda:0"),
        expected_purpose=PURPOSE,
    )
    try:
        with torch.inference_mode():
            b1_first = infer_b1_pair(b1, value, torch)
            b1_repeat = infer_b1_pair(b1, value, torch)
            b2_first = infer_b2(b2, value)
            b2_repeat = infer_b2(b2, value)
            order = (
                ("b1-r1", lambda: infer_b1_pair(b1, value, torch)),
                ("b2-r1", lambda: infer_b2(b2, value)),
                ("b2-r2", lambda: infer_b2(b2, value)),
                ("b1-r2", lambda: infer_b1_pair(b1, value, torch)),
            )
            timings = {
                label: benchmark_pair(label, function, torch, args.rounds)
                for label, function in order
            }
    finally:
        b2.close()
        b1.close()

    comparisons: dict[str, Any] = {}
    deterministic = True
    finite = True
    for name, *_rest in OUTPUTS:
        b1_delta = (b1_first[name] - b1_repeat[name]).abs()
        b2_delta = (b2_first[name] - b2_repeat[name]).abs()
        cross_delta = (b1_first[name] - b2_first[name]).abs()
        entry = {
            "b1_repeat_max_abs": float(b1_delta.max().item()),
            "b2_repeat_max_abs": float(b2_delta.max().item()),
            "b1_vs_b2_max_abs": float(cross_delta.max().item()),
            "b1_vs_b2_mean_abs": float(cross_delta.mean().item()),
            "b1_finite": bool(torch.isfinite(b1_first[name]).all().item()),
            "b2_finite": bool(torch.isfinite(b2_first[name]).all().item()),
        }
        comparisons[name] = entry
        deterministic &= entry["b1_repeat_max_abs"] == 0.0 and entry["b2_repeat_max_abs"] == 0.0
        finite &= entry["b1_finite"] and entry["b2_finite"]

    b1_median = statistics.median(
        [timings["b1-r1"]["wall_p50_ms_per_pair"], timings["b1-r2"]["wall_p50_ms_per_pair"]]
    )
    b2_median = statistics.median(
        [timings["b2-r1"]["wall_p50_ms_per_pair"], timings["b2-r2"]["wall_p50_ms_per_pair"]]
    )
    speedup = (b1_median / b2_median - 1.0) * 100.0
    report = {
        "status": "COMPLETE" if deterministic and finite else "FAILED",
        "scope": "transaction-only RF-DETR MIGraphX B1-vs-B2 paired inference canary",
        "product_changed": False,
        "weights": {"path": str(weights), "sha256": sha256_file(weights)},
        "b1_manifest": {"path": str(b1_manifest), "sha256": sha256_file(b1_manifest)},
        "b2_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "b2_artifact": {"path": str(artifact), "sha256": sha256_file(artifact), "size_bytes": artifact.stat().st_size},
        "rewrites": {"identity_concat": concat, "resize": resize, "projector_convolutions": selected},
        "cpu_resize_parity": cpu_parity,
        "compiled_abi": {name: [kind, list(shape), list(strides)] for name, (kind, shape, strides) in actual_abi.items()},
        "stage_times": stage_times,
        "timings": timings,
        "comparison": comparisons,
        "deterministic": deterministic,
        "finite": finite,
        "paired_median_ms": {"b1": b1_median, "b2": b2_median},
        "b2_throughput_speedup_percent": speedup,
        "microbenchmark_gate": {
            "minimum_speedup_percent": 3.0,
            "passed": bool(deterministic and finite and speedup >= 3.0),
            "note": "Passing authorizes only a real-video product A/B, not integration.",
        },
    }
    (output_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--b1-manifest", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=60)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("--rounds must be at least 5")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)
