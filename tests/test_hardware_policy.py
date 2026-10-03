from __future__ import annotations

from types import SimpleNamespace

import pytest

from jasna.gui.hardware_policy import (
    DEFAULT_DETECTION_BATCH_SIZE,
    gui_batch_size_from_custom_args,
    recommended_detection_batch_size,
    split_batch_size_custom_arg,
)
from jasna.gui.models import AppSettings
from jasna.gui.settings_panel import SettingsPanel


def test_hardware_telemetry_never_changes_default_batch_size() -> None:
    assert recommended_detection_batch_size("rfdetr-v6", 24 * 1024**3) == 4
    assert recommended_detection_batch_size("rfdetr-v6", None) == 4
    assert recommended_detection_batch_size("rfdetr-v6-large", 48 * 1024**3) == 4


@pytest.mark.parametrize(
    ("custom_args", "expected_batch", "expected_encoder_args"),
    [
        ("", None, ""),
        ("--batch-size 1", 1, ""),
        ("--batch-size=1", 1, ""),
        ("--batch-size 4", 4, ""),
        ("--batch-size 8", 8, ""),
        ("--batch-size=8", 8, ""),
        ("--batch-size 1,rc-lookahead=32", 1, "rc-lookahead=32"),
        ("--batch-size 8,rc-lookahead=32", 8, "rc-lookahead=32"),
        ("rc-lookahead=32,--batch-size 4", 4, "rc-lookahead=32"),
    ],
)
def test_batch_flag_is_extracted_before_encoder_validation(
    custom_args: str,
    expected_batch: int | None,
    expected_encoder_args: str,
) -> None:
    assert split_batch_size_custom_arg(custom_args) == (
        expected_batch,
        expected_encoder_args,
    )
    assert gui_batch_size_from_custom_args(custom_args) == (
        DEFAULT_DETECTION_BATCH_SIZE
        if expected_batch is None
        else expected_batch
    )


@pytest.mark.parametrize(
    "custom_args",
    [
        "--batch-size",
        "--batch-size 0",
        "--batch-size 2",
        "--batch-size 6",
        "--batch-size -1",
        "--batch-size 1.0",
        "--batch-size 8 rc-lookahead=32",
        "--batch-size 1,--batch-size 4",
        "--batch-size 4,--batch-size 8",
        "--batch-size-eight=8",
    ],
)
def test_batch_flag_rejects_unsupported_or_ambiguous_forms(custom_args: str) -> None:
    with pytest.raises(ValueError):
        split_batch_size_custom_arg(custom_args)


@pytest.mark.parametrize(
    ("custom_args", "expected_batch"),
    [
        ("", 4),
        ("--batch-size 1", 1),
        ("--batch-size 4", 4),
        ("--batch-size 8", 8),
    ],
)
def test_settings_panel_uses_explicit_custom_flag(
    custom_args: str,
    expected_batch: int,
) -> None:
    panel = SimpleNamespace(
        _sections=[
            SimpleNamespace(
                collect=lambda: {
                    "detection_model": "rfdetr-v6",
                    "encoder_custom_args": custom_args,
                }
            )
        ]
    )

    settings = SettingsPanel.get_settings(panel)

    assert isinstance(settings, AppSettings)
    assert settings.batch_size == expected_batch


@pytest.mark.parametrize("batch_size", [1, 8])
def test_processor_does_not_forward_batch_flag_to_encoder(
    monkeypatch,
    batch_size: int,
) -> None:
    from jasna.accelerator import AcceleratorVendor
    from jasna.gui.processor import Processor

    processor = Processor.__new__(Processor)
    processor._settings = AppSettings(
        encoder_cq=28,
        encoder_custom_args=f"--batch-size {batch_size},preanalysis=1",
    )
    monkeypatch.setattr(
        "jasna.accelerator.vendor_for_device",
        lambda: AcceleratorVendor.AMD,
    )

    assert processor._build_encoder_settings("hevc") == {
        "cq": 28,
        "preanalysis": 1,
    }


@pytest.mark.parametrize("batch_size", [1, 8])
def test_preset_migration_preserves_explicit_batch_flag(batch_size: int) -> None:
    from jasna.gui.models import _migrate_preset_dict

    migrated = _migrate_preset_dict(
        {
            "encoder_custom_args": f"--batch-size {batch_size},cq=22,lookahead=16",
        }
    )

    assert migrated["encoder_cq"] == 22
    assert migrated["encoder_custom_args"] == (
        f"--batch-size {batch_size},rc-lookahead=16"
    )


def test_gui_validation_reports_invalid_batch_flag(monkeypatch) -> None:
    from jasna.gui import validation

    monkeypatch.setattr(validation, "t", lambda key, **_kwargs: key)
    errors = validation.validate_gui_start(
        AppSettings(encoder_custom_args="--batch-size 6"),
        [],
        ltx_available=True,
    )

    assert "error_batch_size_custom_args" in errors
