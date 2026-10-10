from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _eligible_host(monkeypatch: pytest.MonkeyPatch) -> None:
    import jasna.mosaic.rfdetr_migraphx_runner as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(module.torch.version, "hip", "test-hip")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1100"),
    )


def test_product_discovery_returns_installed_gfx1100_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from jasna.mosaic.rfdetr_migraphx_runner import (
        PRODUCT_MANIFEST_FILENAME,
        discover_product_rfdetr_migraphx_manifest,
    )

    _eligible_host(monkeypatch)
    weights = tmp_path / "rfdetr-v6.pt"
    manifest = tmp_path / PRODUCT_MANIFEST_FILENAME
    weights.write_bytes(b"weights")
    manifest.write_text("{}", encoding="utf-8")

    assert discover_product_rfdetr_migraphx_manifest(
        detection_model_name="rfdetr-v6",
        weights_path=weights,
        device=torch.device("cuda:0"),
        fp16=True,
    ) == manifest


@pytest.mark.parametrize(
    ("model_name", "fp16", "arch", "manifest_exists"),
    [
        ("rfdetr-v6-large", True, "gfx1100", True),
        ("rfdetr-v6", False, "gfx1100", True),
        ("rfdetr-v6", True, "gfx1151", True),
        ("rfdetr-v6", True, "gfx1100", False),
    ],
)
def test_product_discovery_preserves_pytorch_when_ineligible_or_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    fp16: bool,
    arch: str,
    manifest_exists: bool,
) -> None:
    import jasna.mosaic.rfdetr_migraphx_runner as module

    _eligible_host(monkeypatch)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName=arch),
    )
    weights = tmp_path / "rfdetr-v6.pt"
    if manifest_exists:
        (tmp_path / module.PRODUCT_MANIFEST_FILENAME).write_text(
            "{}", encoding="utf-8"
        )

    assert module.discover_product_rfdetr_migraphx_manifest(
        detection_model_name=model_name,
        weights_path=weights,
        device=torch.device("cuda:0"),
        fp16=fp16,
    ) is None


def test_registry_auto_selects_product_manifest_without_changing_shared_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jasna.mosaic.detection_registry as registry
    import jasna.mosaic.rfdetr as rfdetr_module
    import jasna.mosaic.rfdetr_migraphx_runner as migraphx_module

    manifest = Path("models/rfdetr-v6.migraphx-gfx1100.json")
    model = MagicMock()
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr(
        registry, "is_amd_device", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        migraphx_module,
        "discover_product_rfdetr_migraphx_manifest",
        lambda **_kwargs: manifest,
    )
    monkeypatch.setattr(rfdetr_module, "RfDetrMosaicDetectionModel", constructor)

    result = registry.build_detection_model(
        "rfdetr-v6",
        Path("models/rfdetr-v6.pt"),
        batch_size=4,
        device=torch.device("cpu"),
        score_threshold=0.35,
        fp16=True,
    )

    assert result is model
    kwargs = constructor.call_args.kwargs
    assert kwargs["batch_size"] == 4
    assert kwargs["score_threshold"] == 0.35
    assert kwargs["amd_migraphx_manifest_path"] == manifest


def test_selected_product_manifest_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jasna.mosaic.detection_registry as registry
    import jasna.mosaic.rfdetr as rfdetr_module
    import jasna.mosaic.rfdetr_migraphx_runner as migraphx_module

    monkeypatch.setattr(
        registry, "is_amd_device", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        migraphx_module,
        "discover_product_rfdetr_migraphx_manifest",
        lambda **_kwargs: Path("installed.json"),
    )
    monkeypatch.setattr(
        rfdetr_module,
        "RfDetrMosaicDetectionModel",
        MagicMock(side_effect=RuntimeError("invalid installed artifact")),
    )

    with pytest.raises(RuntimeError, match="invalid installed artifact"):
        registry.build_detection_model(
            "rfdetr-v6",
            Path("rfdetr-v6.pt"),
            batch_size=4,
            device=torch.device("cpu"),
            score_threshold=0.35,
            fp16=True,
        )


def test_explicit_manifest_rejects_non_amd_before_model_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jasna.mosaic.detection_registry as registry

    monkeypatch.setattr(
        registry, "is_amd_device", lambda *_args, **_kwargs: False
    )
    with pytest.raises(RuntimeError, match="AMD/ROCm"):
        registry.build_detection_model(
            "rfdetr-v6",
            Path("rfdetr-v6.onnx"),
            batch_size=4,
            device=torch.device("cpu"),
            score_threshold=0.35,
            fp16=True,
            amd_migraphx_manifest_path=Path("explicit.json"),
        )
