import pytest

import torch

from jasna.mosaic import detection_registry as registry


def test_detection_engine_roundtrip():
    registry.set_detection_engine("torch")
    assert registry.get_detection_engine() == "torch"
    registry.set_detection_engine("migraphx")
    assert registry.get_detection_engine() == "migraphx"
    registry.set_detection_engine("torch")
    with pytest.raises(ValueError):
        registry.set_detection_engine("tensorrt")


def test_detection_engine_env_fallback(monkeypatch):
    # no explicit pin: the environment decides, else the vendor default (non-AMD here)
    monkeypatch.setattr(registry, "_detection_engine", None)
    monkeypatch.setattr(registry, "is_amd_device", lambda *a, **k: False)
    monkeypatch.setenv("JASNA_DETECTION_ENGINE", "migraphx")
    assert registry.get_detection_engine() == "migraphx"
    monkeypatch.setenv("JASNA_DETECTION_ENGINE", "nonsense")
    assert registry.get_detection_engine() == "torch"
    monkeypatch.delenv("JASNA_DETECTION_ENGINE")
    assert registry.get_detection_engine() == "torch"
    # an explicit pin beats the environment
    registry.set_detection_engine("torch")
    monkeypatch.setenv("JASNA_DETECTION_ENGINE", "migraphx")
    assert registry.get_detection_engine() == "torch"
    registry.set_detection_engine("torch")


def test_migraphx_onnx_path_naming():
    path = registry.Path("model_weights/rfdetr-v6.pt")
    onnx = registry.Path(str(path))
    from jasna.mosaic.rfdetr_migraphx_runner import migraphx_onnx_path

    result = migraphx_onnx_path(path, 480, 8)
    assert result.name == "rfdetr-v6.migraphx.r480.b8.fp16.onnx"
    assert result.parent == onnx.parent


def test_migraphx_engine_rejected_off_amd():
    # a CPU device is neither AMD nor NVIDIA: the model must refuse the engine
    with pytest.raises(RuntimeError, match="only available on AMD"):
        registry.RfDetrModelConfig  # import sanity
        from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

        RfDetrMosaicDetectionModel(
            weights_path=registry.Path("model_weights/rfdetr-v6.pt"),
            batch_size=8,
            device=torch.device("cpu"),
            resolution=480,
            dynamic_batch=True,
            torch_variant="medium",
            engine="migraphx",
        )


def test_detection_engine_default_is_vendor_aware(monkeypatch):
    # no pin, no env: AMD defaults to migraphx, anything else to torch
    monkeypatch.setattr(registry, "_detection_engine", None)
    monkeypatch.delenv("JASNA_DETECTION_ENGINE", raising=False)
    monkeypatch.setattr(registry, "is_amd_device", lambda *a, **k: True)
    assert registry.get_detection_engine() == "migraphx"
    monkeypatch.setattr(registry, "is_amd_device", lambda *a, **k: False)
    assert registry.get_detection_engine() == "torch"
    registry.set_detection_engine("torch")


def test_build_detection_model_ignores_engine_off_amd(monkeypatch):
    # on non-AMD devices the engine setting is inert: NVIDIA keeps its TRT path
    recorded = {}

    class FakeModel:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(registry, "get_detection_engine", lambda: "migraphx")
    monkeypatch.setattr(registry, "is_amd_device", lambda *a, **k: False)
    monkeypatch.setattr("jasna.mosaic.rfdetr.RfDetrMosaicDetectionModel", FakeModel)
    registry.build_detection_model(
        "rfdetr-v6",
        registry.Path("model_weights/rfdetr-v6.pt"),
        batch_size=8,
        device=torch.device("cuda"),
        score_threshold=0.35,
        fp16=True,
    )
    assert recorded["engine"] == "torch"
