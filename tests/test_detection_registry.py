from pathlib import Path

from jasna.mosaic.detection_registry import (
    DEFAULT_DETECTION_MODEL_NAME,
    RFDETR_MODEL_NAMES,
    coerce_detection_model_name,
    detection_model_weights_path,
)


def test_default_detection_model_is_rfdetr_v4() -> None:
    assert DEFAULT_DETECTION_MODEL_NAME == "rfdetr-v4"
    assert "rfdetr-v4" in RFDETR_MODEL_NAMES


def test_rfdetr_v4_weights_path() -> None:
    assert detection_model_weights_path("rfdetr-v4") == Path("model_weights/rfdetr-v4.onnx")
    assert coerce_detection_model_name("rfdetr-v4") == "rfdetr-v4"
