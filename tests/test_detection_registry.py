from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.mosaic.detection_registry import (
    DEFAULT_DETECTION_MODEL_NAME,
    RFDETR_MODEL_NAMES,
    YOLO_MODEL_FILES,
    coerce_detection_model_name,
    detection_model_weights_path,
    discover_available_detection_models,
    is_rfdetr_model,
    is_yolo_model,
    precompile_detection_engine,
)


def test_default_detection_model_is_rfdetr_v5() -> None:
    assert DEFAULT_DETECTION_MODEL_NAME == "rfdetr-v5"
    assert "rfdetr-v5" in RFDETR_MODEL_NAMES


def test_rfdetr_v5_weights_path() -> None:
    assert detection_model_weights_path("rfdetr-v5") == Path("model_weights/rfdetr-v5.onnx")
    assert coerce_detection_model_name("rfdetr-v5") == "rfdetr-v5"


def test_lada_yolo_v4_weights_path() -> None:
    assert detection_model_weights_path("lada-yolo-v4") == Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
    assert coerce_detection_model_name("lada-yolo-v4") == "lada-yolo-v4"


# --- is_rfdetr_model ---

def test_is_rfdetr_model_known() -> None:
    assert is_rfdetr_model("rfdetr-v5")
    assert is_rfdetr_model("rfdetr-v2")


def test_is_rfdetr_model_unknown_version() -> None:
    assert is_rfdetr_model("rfdetr-v99")
    assert is_rfdetr_model("rfdetr-custom")


def test_is_rfdetr_model_rejects_non_rfdetr() -> None:
    assert not is_rfdetr_model("lada-yolo-v4")
    assert not is_rfdetr_model("garbage")
    assert not is_rfdetr_model("")


# --- is_yolo_model ---

def test_is_yolo_model() -> None:
    assert is_yolo_model("lada-yolo-v2")
    assert is_yolo_model("lada-yolo-v4")
    assert not is_yolo_model("rfdetr-v5")
    assert not is_yolo_model("lada-yolo-v99")


# --- coerce_detection_model_name ---

def test_coerce_known_rfdetr() -> None:
    assert coerce_detection_model_name("rfdetr-v5") == "rfdetr-v5"


def test_coerce_dynamic_rfdetr() -> None:
    assert coerce_detection_model_name("rfdetr-v99") == "rfdetr-v99"


def test_coerce_yolo() -> None:
    assert coerce_detection_model_name("lada-yolo-v4") == "lada-yolo-v4"


def test_coerce_garbage_falls_back_to_default() -> None:
    assert coerce_detection_model_name("nonsense") == DEFAULT_DETECTION_MODEL_NAME
    assert coerce_detection_model_name("") == DEFAULT_DETECTION_MODEL_NAME


# --- detection_model_weights_path for dynamic rfdetr ---

def test_dynamic_rfdetr_weights_path() -> None:
    assert detection_model_weights_path("rfdetr-v99") == Path("model_weights/rfdetr-v99.onnx")


# --- discover_available_detection_models ---

def test_discover_empty_dir(tmp_path: Path) -> None:
    assert discover_available_detection_models(tmp_path) == []


def test_discover_nonexistent_dir(tmp_path: Path) -> None:
    assert discover_available_detection_models(tmp_path / "nope") == []


def test_discover_rfdetr_only(tmp_path: Path) -> None:
    (tmp_path / "rfdetr-v5.onnx").touch()
    (tmp_path / "rfdetr-v3.onnx").touch()
    result = discover_available_detection_models(tmp_path)
    assert result == ["rfdetr-v5", "rfdetr-v3"]


def test_discover_unknown_rfdetr_version(tmp_path: Path) -> None:
    (tmp_path / "rfdetr-v99.onnx").touch()
    (tmp_path / "rfdetr-v5.onnx").touch()
    result = discover_available_detection_models(tmp_path)
    assert result == ["rfdetr-v99", "rfdetr-v5"]


def test_discover_yolo_only_when_pt_exists(tmp_path: Path) -> None:
    (tmp_path / "lada_mosaic_detection_model_v4_fast.pt").touch()
    result = discover_available_detection_models(tmp_path)
    assert result == ["lada-yolo-v4"]


def test_discover_yolo_absent_when_pt_missing(tmp_path: Path) -> None:
    result = discover_available_detection_models(tmp_path)
    assert "lada-yolo-v4" not in result


def test_discover_mixed(tmp_path: Path) -> None:
    (tmp_path / "rfdetr-v5.onnx").touch()
    (tmp_path / "rfdetr-v3.onnx").touch()
    (tmp_path / "lada_mosaic_detection_model_v2.pt").touch()
    (tmp_path / "lada_mosaic_detection_model_v4_fast.pt").touch()
    result = discover_available_detection_models(tmp_path)
    assert result == ["rfdetr-v5", "rfdetr-v3", "lada-yolo-v4", "lada-yolo-v2"]


def test_discover_ignores_non_matching_files(tmp_path: Path) -> None:
    (tmp_path / "rfdetr-v5.onnx").touch()
    (tmp_path / "some_random_model.onnx").touch()
    (tmp_path / "random.pt").touch()
    result = discover_available_detection_models(tmp_path)
    assert result == ["rfdetr-v5"]


# --- precompile_detection_engine ---

def test_precompile_noop_on_cpu() -> None:
    precompile_detection_engine("rfdetr-v5", Path("m.onnx"), 1, torch.device("cpu"), True)


def test_precompile_rfdetr_on_cuda() -> None:
    with patch("jasna.trt.compile_onnx_to_tensorrt_engine") as mock_compile:
        precompile_detection_engine("rfdetr-v5", Path("m.onnx"), 2, torch.device("cuda:0"), True)
        mock_compile.assert_called_once_with(Path("m.onnx"), torch.device("cuda:0"), batch_size=2, fp16=True, workspace_gb=20)


def test_precompile_yolo_on_cuda() -> None:
    with (
        patch("jasna.mosaic.yolo_tensorrt_compilation.compile_yolo_to_tensorrt_engine") as mock_compile,
    ):
        precompile_detection_engine("lada-yolo-v4", Path("m.pt"), 4, torch.device("cuda:0"), True)
        mock_compile.assert_called_once()
