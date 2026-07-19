from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jasna.gui.models import AppSettings
from jasna.gui.video_session import video_session_config
from jasna.main import _session_config_from_args, build_parser

_DETECTION_PATH = Path("mw") / "rfdetr-v5.onnx"
_RESTORATION_PATH = Path("mw") / "lada_mosaic_restoration_model_generic_v1.2.pth"


def _cli_config(extra_args: list[str] | None = None):
    args = build_parser().parse_args(
        ["--input", "a.mp4", "--output", "b.mp4", *(extra_args or [])]
    )
    return _session_config_from_args(
        args,
        codec="hevc",
        encoder_settings={},
        detection_model_name="rfdetr-v5",
        detection_model_path=_DETECTION_PATH,
        restoration_model_path=_RESTORATION_PATH,
        lut_path=None,
    )


def _gui_config(settings: AppSettings):
    with (
        patch(
            "jasna.mosaic.detection_registry.coerce_detection_model_name",
            side_effect=lambda name: name,
        ),
        patch(
            "jasna.mosaic.detection_registry.require_detection_model_weights",
            return_value=_DETECTION_PATH,
        ),
        patch("jasna.engine_paths.model_weights_dir", return_value=Path("mw")),
    ):
        return video_session_config(settings, codec="hevc", encoder_settings={})


def test_cli_defaults_map_to_expected_config() -> None:
    config = _cli_config()

    assert config.device == "cuda:0"
    assert config.fp16 is True
    assert config.batch_size == 4
    assert config.detection_model_name == "rfdetr-v5"
    assert config.detection_model_path == _DETECTION_PATH
    assert config.detection_score_threshold == 0.25
    assert config.max_detection_gap == 2
    assert config.min_detection_duration == 2
    assert config.restoration_model_path == _RESTORATION_PATH
    assert config.compile_basicvsrpp is True
    assert config.max_clip_size == 90
    assert config.temporal_overlap == 8
    assert config.enable_crossfade is True
    assert config.denoise_strength == "none"
    assert config.denoise_step == "after_primary"
    assert config.secondary_restoration == "none"
    assert config.tvai_model == "iris-2"
    assert config.tvai_scale == 4
    assert config.tvai_workers == 2
    assert config.rtx_scale == 4
    assert config.rtx_quality == "high"
    assert config.rtx_denoise == "medium"
    assert config.rtx_deblur == "none"
    assert config.vr_mode == "auto"
    assert config.codec == "hevc"
    assert config.encoder_settings == {}
    assert config.lut_path is None
    assert config.retarget_high_fps is False
    assert config.disable_progress is False
    assert config.working_dir is None


def test_cli_non_default_args_are_mapped() -> None:
    config = _cli_config(
        [
            "--no-fp16",
            "--batch-size", "8",
            "--denoise", "high",
            "--denoise-step", "after_secondary",
            "--secondary-restoration", "rtx-super-res",
            "--rtx-quality", "ultra",
            "--vr-mode", "sbs",
            "--no-progress",
            "--working-directory", "/fast/scratch",
            "--retarget-high-fps",
        ]
    )

    assert config.fp16 is False
    assert config.batch_size == 8
    assert config.denoise_strength == "high"
    assert config.denoise_step == "after_secondary"
    assert config.secondary_restoration == "rtx-super-res"
    assert config.rtx_quality == "ultra"
    assert config.vr_mode == "sbs"
    assert config.disable_progress is True
    assert config.working_dir == Path("/fast/scratch")
    assert config.retarget_high_fps is True


def test_gui_defaults_match_cli_defaults() -> None:
    cli = _cli_config()
    gui = _gui_config(AppSettings())

    assert replace(cli, disable_progress=True) == gui


def test_gui_config_maps_settings_fields() -> None:
    settings = AppSettings(
        fp16_mode=False,
        lut_path="  lut.cube  ",
        working_directory="/tmp/work",
        secondary_restoration="tvai",
        tvai_scale=2,
        denoise_strength="low",
    )
    config = _gui_config(settings)

    assert config.fp16 is False
    assert config.lut_path == "lut.cube"
    assert config.working_dir == Path("/tmp/work")
    assert config.secondary_restoration == "tvai"
    assert config.tvai_scale == 2
    assert config.denoise_strength == "low"
    assert config.disable_progress is True
