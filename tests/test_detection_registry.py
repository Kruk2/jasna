from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import jasna.mosaic.detection_registry as registry
from jasna.mosaic.detection_registry import (
    build_detection_model,
    coerce_detection_model_name,
    detection_model_weights_path,
    detection_model_choices,
    discover_available_detection_models,
    is_rfdetr_model,
    is_yolo_model,
    recommended_score_threshold,
    require_detection_model_weights,
    rfdetr_model_config,
)

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def test_rfdetr_model_config_unknown_falls_back_to_dynamic_batch() -> None:
    config = rfdetr_model_config("rfdetr-custom")
    assert (
        config.resolution,
        config.score_threshold,
        config.dynamic_batch,
        config.fixed_batch_size,
    ) == (768, 0.25, True, None)


@pytest.mark.parametrize(
    ("name", "requested", "engine_batch_size", "dynamic"),
    [
        ("rfdetr-v5", 2, 4, False),
        ("rfdetr-v5", 8, 4, False),
        ("rfdetr-v6", 8, 8, True),
        ("rfdetr-v6-large", 1, 1, True),
        ("rfdetr-custom", 3, 3, True),
    ],
)
def test_an_rfdetr_engine_takes_its_models_fixed_batch_or_the_requested_one(name, requested, engine_batch_size, dynamic):
    config = rfdetr_model_config(name)

    assert config.engine_batch_size(requested) == engine_batch_size
    assert config.dynamic_batch is dynamic


@pytest.mark.parametrize("requested", [0, -1])
def test_an_rfdetr_engine_batch_must_be_positive(requested):
    with pytest.raises(ValueError, match="requested_batch_size must be > 0"):
        rfdetr_model_config("rfdetr-v6").engine_batch_size(requested)


def test_a_yolo_model_gets_the_default_score_threshold() -> None:
    assert recommended_score_threshold("lada-yolo-v4") == 0.25


@pytest.mark.parametrize(
    ("name", "rfdetr", "yolo"),
    [
        ("rfdetr-v5", True, False),
        ("rfdetr-v2", True, False),
        ("rfdetr-v99", True, False),
        ("rfdetr-custom", True, False),
        ("lada-yolo-v2", False, True),
        ("lada-yolo-v4", False, True),
        ("zelefans-vr-yolo-v2", False, True),
        ("lada-yolo-v99", False, False),
        ("garbage", False, False),
        ("", False, False),
    ],
)
def test_model_family_follows_the_name(name, rfdetr, yolo) -> None:
    assert (is_rfdetr_model(name), is_yolo_model(name)) == (rfdetr, yolo)


@pytest.mark.parametrize("name", ["rfdetr-v5", "rfdetr-v6", "rfdetr-v99", "lada-yolo-v4"])
def test_coerce_keeps_a_known_or_dynamic_model_name(name) -> None:
    assert coerce_detection_model_name(name) == name


@pytest.mark.parametrize(
    ("name", "message"),
    [("nonsense", "Unknown detection model 'nonsense'"), ("", None), ("yolo-v4", "lada-yolo-v4")],
    ids=["garbage", "empty", "yolo-typo-lists-valid-names"],
)
def test_coerce_rejects_an_unknown_name(name, message) -> None:
    with pytest.raises(ValueError, match=message):
        coerce_detection_model_name(name)


def test_dynamic_rfdetr_weights_path() -> None:
    assert detection_model_weights_path("rfdetr-v99") == Path("model_weights/rfdetr-v99.onnx")


def test_discover_nonexistent_dir(tmp_path: Path) -> None:
    assert discover_available_detection_models(tmp_path / "nope") == []


@pytest.mark.parametrize(
    ("files", "amd", "expected"),
    [
        ((), False, []),
        (("rfdetr-v5.onnx", "rfdetr-v3.onnx"), False, ["rfdetr-v5", "rfdetr-v3"]),
        (("rfdetr-v99.onnx", "rfdetr-v5.onnx"), False, ["rfdetr-v99", "rfdetr-v5"]),
        (("lada_mosaic_detection_model_v4_fast.pt",), False, ["lada-yolo-v4"]),
        (("lada_vr_mosaic_detection_model_v2_accurate.pt",), False, ["zelefans-vr-yolo-v2"]),
        (
            ("rfdetr-v5.onnx", "rfdetr-v3.onnx", "lada_mosaic_detection_model_v2.pt", "lada_mosaic_detection_model_v4_fast.pt"),
            False,
            ["rfdetr-v5", "rfdetr-v3", "lada-yolo-v4", "lada-yolo-v2"],
        ),
        (("rfdetr-v5.onnx", "some_random_model.onnx", "random.pt"), False, ["rfdetr-v5"]),
        (("rfdetr-v6.pt", "rfdetr-v6-large.onnx"), True, ["rfdetr-v6"]),
        (("rfdetr-v6.pt", "rfdetr-v6-large.onnx"), False, ["rfdetr-v6-large"]),
    ],
    ids=[
        "empty", "rfdetr", "unknown-rfdetr-version", "yolo", "bundled-vr", "mixed", "ignores-other-files",
        "amd-torch-weights", "nvidia-onnx-weights",
    ],
)
def test_discover_lists_the_models_whose_weights_are_present(monkeypatch, tmp_path, files, amd, expected) -> None:
    monkeypatch.setattr(registry, "is_amd_device", lambda _device=None: amd)
    for name in files:
        (tmp_path / name).touch()

    assert discover_available_detection_models(tmp_path) == expected


def test_bundled_vr_model_is_not_offered_when_weights_are_missing(tmp_path: Path) -> None:
    result = detection_model_choices(tmp_path)
    assert "zelefans-vr-yolo-v2" not in result


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


def test_build_detection_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown detection model 'nonsense'"):
        build_detection_model(
            "nonsense", Path("x"),
            batch_size=1, device=torch.device("cpu"), score_threshold=0.25, fp16=False,
        )


def test_rfdetr_weights_suffix_is_vendor_specific(monkeypatch) -> None:
    monkeypatch.setattr(registry, "is_amd_device", lambda: True)
    assert registry.rfdetr_weights_suffix() == ".pt"
    assert registry.detection_model_spec("rfdetr-v6").filename == "rfdetr-v6.pt"

    monkeypatch.setattr(registry, "is_amd_device", lambda: False)
    assert registry.rfdetr_weights_suffix() == ".onnx"
    assert registry.detection_model_spec("rfdetr-v6").filename == "rfdetr-v6.onnx"


def test_resolve_detection_model_uses_explicit_path_and_recommended_threshold(tmp_path) -> None:
    from jasna.mosaic.detection_registry import resolve_detection_model

    weights = tmp_path / "custom.onnx"
    weights.write_bytes(b"x")

    name, path, threshold = resolve_detection_model("RFDETR-V6", str(weights), None)

    assert name == "rfdetr-v6"
    assert path == weights
    assert threshold == recommended_score_threshold("rfdetr-v6")
    assert resolve_detection_model("rfdetr-v6", str(weights), 0.5)[2] == 0.5


def test_resolve_detection_model_rejects_a_missing_explicit_path(tmp_path) -> None:
    import pytest

    from jasna.mosaic.detection_registry import resolve_detection_model

    with pytest.raises(FileNotFoundError):
        resolve_detection_model("rfdetr-v6", str(tmp_path / "missing.onnx"), None)


@pytest.mark.parametrize(("amd", "expected"), [(True, 8), (False, 4)])
def test_default_detection_batch_size_is_eight_on_amd(monkeypatch, amd, expected):
    monkeypatch.setattr(registry, "is_amd_device", lambda _device: amd)

    assert registry.default_detection_batch_size("cuda:0") == expected
