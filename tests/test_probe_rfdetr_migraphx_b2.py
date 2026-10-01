from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_rfdetr_migraphx_b2.py"
SPEC = importlib.util.spec_from_file_location("probe_rfdetr_migraphx_b2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

PRODUCT_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "probe_rfdetr_migraphx_b2_product.py"
)
PRODUCT_SPEC = importlib.util.spec_from_file_location(
    "probe_rfdetr_migraphx_b2_product", PRODUCT_SCRIPT
)
assert PRODUCT_SPEC is not None and PRODUCT_SPEC.loader is not None
product_probe = importlib.util.module_from_spec(PRODUCT_SPEC)
PRODUCT_SPEC.loader.exec_module(product_probe)


def test_static_contract_constants_are_product_compatible() -> None:
    assert probe.INPUT_SHAPE == (2, 3, 576, 576)
    assert probe.INPUT_STRIDES == (995328, 331776, 576, 1)
    assert [item[0] for item in probe.OUTPUTS] == ["dets", "labels", "masks"]
    assert probe.INTERNAL_PRECISION == "mixed-dot-projector-convolution-fp16"


def test_identity_concat_rewrite_is_exact_and_narrow() -> None:
    onnx = pytest.importorskip("onnx")
    node = onnx.helper.make_node(
        "Concat", ["kept", "empty"], ["output"], name=probe.IDENTITY_CONCAT, axis=1
    )
    graph = onnx.helper.make_graph(
        [node],
        "identity-concat",
        [
            onnx.helper.make_tensor_value_info("kept", onnx.TensorProto.FLOAT, [2, 200, 4]),
            onnx.helper.make_tensor_value_info("empty", onnx.TensorProto.FLOAT, [2, 0, 4]),
        ],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 200, 4])],
    )
    model = onnx.helper.make_model(graph)

    evidence = probe.rewrite_identity_concat(model)

    assert model.graph.node[0].op_type == "Identity"
    assert list(model.graph.node[0].input) == ["kept"]
    assert evidence["dropped_input"] == "empty"


def test_projector_rewrite_only_casts_selected_convolutions() -> None:
    onnx = pytest.importorskip("onnx")
    selected = onnx.helper.make_node(
        "Conv", ["x", "w"], ["selected"], name="/backbone/projector/example/Conv"
    )
    untouched = onnx.helper.make_node(
        "Conv", ["selected", "w"], ["output"], name="/segmentation_head/example/Conv"
    )
    graph = onnx.helper.make_graph(
        [selected, untouched],
        "projector",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 1, 2, 2])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 1, 2, 2])],
        initializer=[onnx.numpy_helper.from_array(__import__("numpy").ones((1, 1, 1, 1), dtype="float32"), name="w")],
    )
    model = onnx.helper.make_model(graph)

    names = probe.rewrite_selected_convolutions(model, expected_selected=1)

    assert names == ["/backbone/projector/example/Conv"]
    assert [node.op_type for node in model.graph.node] == ["Cast", "Cast", "Conv", "Cast", "Conv"]
    assert model.graph.node[-1].name == "/segmentation_head/example/Conv"


def test_projector_rewrite_fails_closed_on_topology_change() -> None:
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [],
        "empty",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 1])],
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 1])],
    )
    model = onnx.helper.make_model(graph)
    with pytest.raises(RuntimeError, match="expected 8"):
        probe.rewrite_selected_convolutions(model)


def test_product_probe_normalizes_separator_and_requires_arguments() -> None:
    assert product_probe.normalize_jasna_args(["--", "--input", "in.mp4"]) == [
        "--input",
        "in.mp4",
    ]
    with pytest.raises(ValueError, match="missing Jasna"):
        product_probe.normalize_jasna_args(["--"])


def test_product_probe_requires_one_output_pair() -> None:
    assert product_probe.value_after(["--output", "out.mp4"], "--output") == "out.mp4"
    with pytest.raises(ValueError, match="exactly one"):
        product_probe.value_after([], "--output")
    with pytest.raises(ValueError, match="exactly one"):
        product_probe.value_after(
            ["--output", "one.mp4", "--output", "two.mp4"], "--output"
        )
