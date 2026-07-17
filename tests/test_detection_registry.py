from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.mosaic.detection_registry import (
    DEFAULT_DETECTION_MODEL_NAME,
    RFDETR_MODEL_NAMES,
    YOLO_MODEL_FILES,
    build_detection_model,
    coerce_detection_model_name,
    detection_model_weights_path,
    detection_model_choices,
    discover_available_detection_models,
    is_rfdetr_model,
    is_yolo_model,
    precompile_detection_engine,
    require_detection_model_weights,
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
    assert is_yolo_model("zelefans-vr-yolo-v2")
    assert not is_yolo_model("rfdetr-v5")
    assert not is_yolo_model("lada-yolo-v99")


# --- coerce_detection_model_name ---

def test_coerce_known_rfdetr() -> None:
    assert coerce_detection_model_name("rfdetr-v5") == "rfdetr-v5"


def test_coerce_dynamic_rfdetr() -> None:
    assert coerce_detection_model_name("rfdetr-v99") == "rfdetr-v99"


def test_coerce_yolo() -> None:
    assert coerce_detection_model_name("lada-yolo-v4") == "lada-yolo-v4"


def test_coerce_garbage_raises() -> None:
    with pytest.raises(ValueError, match="Unknown detection model 'nonsense'"):
        coerce_detection_model_name("nonsense")
    with pytest.raises(ValueError):
        coerce_detection_model_name("")


def test_coerce_yolo_typo_raises_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="lada-yolo-v4"):
        coerce_detection_model_name("yolo-v4")


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


def test_bundled_vr_model_is_not_offered_when_weights_are_missing(tmp_path: Path) -> None:
    result = detection_model_choices(tmp_path)
    assert "zelefans-vr-yolo-v2" not in result


def test_discover_bundled_vr_model(tmp_path: Path) -> None:
    (tmp_path / "lada_vr_mosaic_detection_model_v2_accurate.pt").touch()
    assert discover_available_detection_models(tmp_path) == [
        "zelefans-vr-yolo-v2"
    ]


def test_discover_mixed(tmp_path: Path) -> None:
    (tmp_path / "rfdetr-v5.onnx").touch()
    (tmp_path / "rfdetr-v3.onnx").touch()
    (tmp_path / "lada_mosaic_detection_model_v2.pt").touch()
    (tmp_path / "lada_mosaic_detection_model_v4_fast.pt").touch()
    result = discover_available_detection_models(tmp_path)
    assert result == ["rfdetr-v5", "rfdetr-v3", "lada-yolo-v4", "lada-yolo-v2"]


def test_zelefans_vr_model_weights_path() -> None:
    assert detection_model_weights_path("zelefans-vr-yolo-v2") == Path(
        "model_weights/lada_vr_mosaic_detection_model_v2_accurate.pt"
    )


def test_require_detection_model_weights_returns_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "vr.pt"
    path.touch()
    with patch(
        "jasna.mosaic.detection_registry.detection_model_weights_path",
        return_value=path,
    ):
        assert require_detection_model_weights("zelefans-vr-yolo-v2") == path


def test_require_detection_model_weights_rejects_missing_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.pt"
    with (
        patch(
            "jasna.mosaic.detection_registry.detection_model_weights_path",
            return_value=path,
        ),
        pytest.raises(FileNotFoundError, match="Detection model weights not found"),
    ):
        require_detection_model_weights("zelefans-vr-yolo-v2")


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


def test_precompile_zelefans_yolo_on_cuda() -> None:
    with patch(
        "jasna.mosaic.yolo_tensorrt_compilation.compile_yolo_to_tensorrt_engine"
    ) as mock_compile:
        precompile_detection_engine(
            "zelefans-vr-yolo-v2",
            Path("vr.pt"),
            4,
            torch.device("cuda:0"),
            True,
        )

    mock_compile.assert_called_once()


# --- build_detection_model ---

def test_build_detection_model_rfdetr() -> None:
    with (
        patch("jasna.mosaic.rfdetr.RfDetrMosaicDetectionModel") as mock_rf,
        patch("jasna.mosaic.yolo.YoloMosaicDetectionModel") as mock_yolo,
    ):
        build_detection_model(
            "rfdetr-v5", Path("rfdetr-v5.onnx"),
            batch_size=4, device=torch.device("cpu"), score_threshold=0.25, fp16=True,
        )
        mock_rf.assert_called_once()
        mock_yolo.assert_not_called()
        assert mock_rf.call_args.kwargs["onnx_path"] == Path("rfdetr-v5.onnx")


def test_build_detection_model_yolo() -> None:
    with (
        patch("jasna.mosaic.rfdetr.RfDetrMosaicDetectionModel") as mock_rf,
        patch("jasna.mosaic.yolo.YoloMosaicDetectionModel") as mock_yolo,
    ):
        build_detection_model(
            "lada-yolo-v4", Path("lada_mosaic_detection_model_v4_fast.pt"),
            batch_size=4, device=torch.device("cpu"), score_threshold=0.25, fp16=True,
        )
        mock_yolo.assert_called_once()
        mock_rf.assert_not_called()
        assert mock_yolo.call_args.kwargs["model_path"] == Path("lada_mosaic_detection_model_v4_fast.pt")


def test_build_detection_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown detection model 'nonsense'"):
        build_detection_model(
            "nonsense", Path("x"),
            batch_size=1, device=torch.device("cpu"), score_threshold=0.25, fp16=False,
        )
