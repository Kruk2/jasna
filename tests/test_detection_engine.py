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


# ---------------------------------------------------------------------------
# MIGraphX runtime location: a self-contained copy vs the Windows ML catalogue
# ---------------------------------------------------------------------------


def _touch_provider(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "onnxruntime_providers_migraphx.dll").write_bytes(b"")
    return directory


def test_migraphx_runtime_covers_both_amd_generations():
    # One EP package carries RDNA 3 and RDNA 4 kernels: nothing per-generation to pick.
    from jasna.mosaic.rfdetr_migraphx_runner import MIGRAPHX_SUPPORTED_ARCHS

    assert {"gfx1100", "gfx1101", "gfx1102"} <= MIGRAPHX_SUPPORTED_ARCHS   # RX 7000
    assert {"gfx1200", "gfx1201"} <= MIGRAPHX_SUPPORTED_ARCHS              # RX 9000


def test_migraphx_ep_dir_env_override_wins(monkeypatch, tmp_path):
    from jasna.mosaic import rfdetr_migraphx_runner as mod

    env_dir = _touch_provider(tmp_path / "custom")
    _touch_provider(tmp_path / "model_weights" / "migraphx-ep")
    monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path / "model_weights")
    monkeypatch.setenv("JASNA_MIGRAPHX_EP_DIR", str(env_dir))

    assert mod.migraphx_ep_dir_candidates()[0] == env_dir
    assert mod.find_local_ep_dir() == env_dir


def test_migraphx_ep_dir_finds_bundled_weights_copy(monkeypatch, tmp_path):
    from jasna.mosaic import rfdetr_migraphx_runner as mod

    bundled = _touch_provider(tmp_path / "model_weights" / "migraphx-ep")
    monkeypatch.delenv("JASNA_MIGRAPHX_EP_DIR", raising=False)
    monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path / "model_weights")

    assert mod.find_local_ep_dir() == bundled


def test_migraphx_ep_dir_finds_app_relative_copy(monkeypatch, tmp_path):
    from jasna.mosaic import rfdetr_migraphx_runner as mod

    monkeypatch.delenv("JASNA_MIGRAPHX_EP_DIR", raising=False)
    monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path / "model_weights")
    monkeypatch.chdir(tmp_path)
    _touch_provider(tmp_path / "ep" / "amd")

    assert mod.find_local_ep_dir() == registry.Path("ep/amd")


def test_migraphx_ep_package_returns_absolute_paths(monkeypatch, tmp_path):
    # os.add_dll_directory() rejects relative paths (WinError 87), and the bundled
    # candidates are relative exactly like model_weights in a source checkout.
    from jasna.mosaic import rfdetr_migraphx_runner as mod

    monkeypatch.delenv("JASNA_MIGRAPHX_EP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "model_weights_dir", lambda: registry.Path("model_weights"))
    _touch_provider(tmp_path / "model_weights" / "migraphx-ep")

    ep_dir, provider = mod._ep_package()

    assert ep_dir.is_absolute()
    assert ep_dir.name == "migraphx-ep"
    assert provider.is_absolute()
    assert provider.parent == ep_dir
    assert provider.name == "onnxruntime_providers_migraphx.dll"


def test_migraphx_ep_dir_absent_falls_back_to_catalogue(monkeypatch, tmp_path):
    from jasna.mosaic import rfdetr_migraphx_runner as mod

    monkeypatch.delenv("JASNA_MIGRAPHX_EP_DIR", raising=False)
    monkeypatch.setattr(mod, "model_weights_dir", lambda: tmp_path / "model_weights")
    monkeypatch.chdir(tmp_path)

    assert mod.find_local_ep_dir() is None


def test_migraphx_skips_the_windows_ml_plugin_rail():
    # Preloading the plugin rail is what used to crash first runs; the classic provider
    # never imports it, so it must stay out of the preload set.
    from jasna.mosaic.rfdetr_migraphx_runner import _SKIP_PRELOAD

    assert {"amdgpu-ep.dll", "directml-backend.dll"} <= _SKIP_PRELOAD


def test_hip_arch_is_read_or_none():
    from jasna.mosaic.rfdetr_migraphx_runner import hip_arch

    arch = hip_arch()
    assert arch is None or arch.startswith("gfx")
