from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem


_STAGE_ROOT = Path(
    os.environ.get("JASNA_SOURCE_UNDER_TEST", Path(__file__).resolve().parents[1])
)
_IMPORT_ROOT = Path(os.environ.get("JASNA_PRODUCT_IMPORT_ROOT", _STAGE_ROOT))


def _load_staged_module(name: str, relative_path: str):
    path = _STAGE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _settings(**overrides):
    values = {
        "encoder_custom_args": "",
        "post_export_action": "none",
        "post_export_command": "",
        "secondary_restoration": "none",
        "detection_model": "rfdetr-v6",
    }
    values.update(overrides)
    return AppSettings(**values)


def _install_detection_resolver(monkeypatch, resolver) -> None:
    import jasna.mosaic.detection_registry as registry
    monkeypatch.setattr(registry, "require_detection_model_weights", resolver)


def _patch_weights_dir(monkeypatch, weights_dir: Path) -> None:
    import jasna.engine_paths as engine_paths

    monkeypatch.setattr(engine_paths, "model_weights_dir", lambda: weights_dir)


def test_start_validation_reports_the_selected_missing_detector(
    monkeypatch, tmp_path: Path
) -> None:
    validation = _load_staged_module(
        "staged_gui_validation_missing_detector", "jasna/gui/validation.py"
    )
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    (weights_dir / "lada_mosaic_restoration_model_generic_v1.2.pth").touch()
    _patch_weights_dir(monkeypatch, weights_dir)

    observed: list[str] = []

    def require_weights(name: str) -> Path:
        observed.append(name)
        raise FileNotFoundError("Detection model weights not found: missing-selected.pt")

    _install_detection_resolver(monkeypatch, require_weights)

    assert validation.validate_gui_start(
        _settings(detection_model="rfdetr-v6-large")
    , [JobItem(path=Path("video.mp4"))], ltx_available=True) == ["Detection model weights not found: missing-selected.pt"]
    assert observed == ["rfdetr-v6-large"]


def test_start_validation_reports_an_unknown_detector_choice(
    monkeypatch, tmp_path: Path
) -> None:
    validation = _load_staged_module(
        "staged_gui_validation_unknown_detector", "jasna/gui/validation.py"
    )
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    (weights_dir / "lada_mosaic_restoration_model_generic_v1.2.pth").touch()
    _patch_weights_dir(monkeypatch, weights_dir)

    observed: list[str] = []

    def require_weights(name: str) -> Path:
        observed.append(name)
        raise ValueError("Unknown detection model 'retired-detector'")

    _install_detection_resolver(monkeypatch, require_weights)

    assert validation.validate_gui_start(
        _settings(detection_model="retired-detector")
    , [JobItem(path=Path("video.mp4"))], ltx_available=True) == ["Unknown detection model 'retired-detector'"]
    assert observed == ["retired-detector"]


def test_start_validation_reports_missing_primary_restoration_weights(
    monkeypatch, tmp_path: Path
) -> None:
    validation = _load_staged_module(
        "staged_gui_validation_missing_restoration", "jasna/gui/validation.py"
    )
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    detector = weights_dir / "rfdetr-v6.pt"
    detector.touch()
    _patch_weights_dir(monkeypatch, weights_dir)
    _install_detection_resolver(monkeypatch, lambda _name: detector)

    assert validation.validate_gui_start(_settings(), [JobItem(path=Path("video.mp4"))], ltx_available=True) == [
        f"Restoration model weights not found: "
        f"{weights_dir / 'lada_mosaic_restoration_model_generic_v1.2.pth'}"
    ]


def test_start_validation_rejects_a_restoration_checkpoint_directory(
    monkeypatch, tmp_path: Path
) -> None:
    validation = _load_staged_module(
        "staged_gui_validation_restoration_directory", "jasna/gui/validation.py"
    )
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    detector = weights_dir / "rfdetr-v6.pt"
    detector.touch()
    checkpoint_directory = (
        weights_dir / "lada_mosaic_restoration_model_generic_v1.2.pth"
    )
    checkpoint_directory.mkdir()
    _patch_weights_dir(monkeypatch, weights_dir)
    _install_detection_resolver(monkeypatch, lambda _name: detector)

    assert validation.validate_gui_start(_settings(), [JobItem(path=Path("video.mp4"))], ltx_available=True) == [
        f"Restoration model weights not found: {checkpoint_directory}"
    ]


def test_start_validation_keeps_valid_core_models_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    validation = _load_staged_module(
        "staged_gui_validation_valid_models", "jasna/gui/validation.py"
    )
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    detector = weights_dir / "selected.pt"
    detector.touch()
    (weights_dir / "lada_mosaic_restoration_model_generic_v1.2.pth").touch()
    _patch_weights_dir(monkeypatch, weights_dir)
    _install_detection_resolver(monkeypatch, lambda _name: detector)

    assert validation.validate_gui_start(_settings(detection_model="selected"), [JobItem(path=Path("video.mp4"))], ltx_available=True) == []


def test_start_validation_error_order_keeps_existing_and_tvai_errors_around_model_errors(
    monkeypatch, tmp_path: Path
) -> None:
    validation = _load_staged_module(
        "staged_gui_validation_error_order", "jasna/gui/validation.py"
    )
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    _patch_weights_dir(monkeypatch, weights_dir)

    def require_weights(_name: str) -> Path:
        raise FileNotFoundError("Detection model weights not found: selected.pt")

    _install_detection_resolver(monkeypatch, require_weights)
    monkeypatch.delenv("TVAI_MODEL_DATA_DIR", raising=False)
    monkeypatch.delenv("TVAI_MODEL_DIR", raising=False)
    missing_ffmpeg = tmp_path / "missing_ffmpeg.exe"

    assert validation.validate_gui_start(
        _settings(
            encoder_custom_args="--batch-size 3",
            post_export_action="command",
            secondary_restoration="tvai",
            tvai_ffmpeg_path=str(missing_ffmpeg),
        )
    , [JobItem(path=Path("video.mp4"))], ltx_available=True) == [
        t("error_batch_size_custom_args"),
        t("error_post_export_command_required"),
        "Detection model weights not found: selected.pt",
        f"Restoration model weights not found: "
        f"{weights_dir / 'lada_mosaic_restoration_model_generic_v1.2.pth'}",
        t("error_tvai_data_dir_not_set"),
        t("error_tvai_model_dir_not_set"),
        t("error_tvai_ffmpeg_not_found", path=str(missing_ffmpeg)),
    ]


def test_font_backend_startup_error_exits_nonzero_without_real_torch() -> None:
    app_path = _STAGE_ROOT / "jasna/gui/app.py"
    script = textwrap.dedent(
        f"""
        import importlib.util
        import os
        import sys
        import types

        sys.path.insert(0, {str(_IMPORT_ROOT)!r})
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(hip=None, cuda=None)
        sys.modules["torch"] = torch

        source = {str(app_path)!r}
        spec = importlib.util.spec_from_file_location("staged_gui_app", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        import jasna._frozen as frozen
        frozen.patch_frozen_torch = lambda: None
        module.scaling.activate_static_dpi = lambda _minimum: None
        os.environ.pop("JASNA_GUI_FONT_PROBE", None)

        class BrokenApp:
            def __init__(self):
                raise module.GuiFontBackendError("font backend unavailable")

        module.JasnaApp = BrokenApp
        try:
            module.run_gui()
        except SystemExit as error:
            if error.code != 1:
                raise AssertionError(f"unexpected exit code: {{error.code!r}}")
        else:
            raise AssertionError("font backend startup failure returned successfully")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_IMPORT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "font backend unavailable" in completed.stderr
