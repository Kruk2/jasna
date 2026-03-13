from pathlib import Path

from jasna.mosaic.detection_registry import (
    DEFAULT_DETECTION_MODEL_NAME,
    RFDETR_MODEL_NAMES,
    coerce_detection_model_name,
    detection_model_weights_path,
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
