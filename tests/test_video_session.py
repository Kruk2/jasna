from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from jasna.gui.models import AppSettings
from jasna.gui.video_session import (
    build_video_session,
    video_session_config,
    video_session_key,
)


def test_video_session_key_stable_for_identical_settings() -> None:
    assert video_session_key(AppSettings()) == video_session_key(AppSettings())


def test_video_session_key_changes_on_session_fields() -> None:
    base = video_session_key(AppSettings())
    assert video_session_key(replace(AppSettings(), detection_model="lada-yolo-v2")) != base
    assert video_session_key(replace(AppSettings(), secondary_restoration="unet-4x")) != base
    assert video_session_key(replace(AppSettings(), max_clip_size=60)) != base
    assert video_session_key(replace(AppSettings(), fp16_mode=False)) != base


def test_video_session_key_ignores_encoder_fields() -> None:
    base = video_session_key(AppSettings())
    assert video_session_key(replace(AppSettings(), encoder_cq=30, codec="h264")) == base


def test_video_session_key_ignores_vr_routing() -> None:
    base = video_session_key(AppSettings())
    assert video_session_key(replace(AppSettings(), vr_mode="off")) == base
    assert video_session_key(replace(AppSettings(), vr_mode="sbs-fisheye")) == base
    assert video_session_key(replace(AppSettings(), vr_projection="fisheye")) == base


def test_video_session_config_forwards_vr_projection() -> None:
    settings = replace(AppSettings(), vr_projection="gnomonic")

    with (
        patch("jasna.engine_paths.model_weights_dir"),
        patch(
            "jasna.mosaic.detection_registry.coerce_detection_model_name",
            side_effect=lambda name: name,
        ),
        patch(
            "jasna.mosaic.detection_registry.require_detection_model_weights",
            return_value=Path("det.engine"),
        ),
    ):
        config = video_session_config(
            settings,
            codec="hevc",
            encoder_settings={},
        )

    assert config.vr_projection == "gnomonic"


def test_video_session_key_includes_active_secondary_knobs() -> None:
    tvai = replace(AppSettings(), secondary_restoration="tvai")
    assert video_session_key(replace(tvai, tvai_scale=2)) != video_session_key(tvai)
    assert video_session_key(replace(tvai, rtx_scale=2)) == video_session_key(tvai)

    rtx = replace(AppSettings(), secondary_restoration="rtx-super-res")
    assert video_session_key(replace(rtx, rtx_quality="low")) != video_session_key(rtx)
    assert video_session_key(replace(rtx, tvai_scale=2)) == video_session_key(rtx)


def _build(settings: AppSettings):
    compile_result = MagicMock(use_basicvsrpp_tensorrt=True)
    with (
        patch("jasna._suppress_noise.install"),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=compile_result) as compiled,
        patch("jasna.engine_paths.model_weights_dir"),
        patch("jasna.mosaic.detection_registry.coerce_detection_model_name", side_effect=lambda n: n),
        patch("jasna.mosaic.detection_registry.require_detection_model_weights") as det_path,
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_cls,
        patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as pipeline_cls,
        patch("jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer") as unet_cls,
    ):
        det_path.return_value = "det.engine"
        session = build_video_session(settings, log=lambda _msg: None)
    return session, compiled, restorer_cls, pipeline_cls, unet_cls


def test_build_video_session_without_secondary() -> None:
    session, compiled, restorer_cls, pipeline_cls, _unet_cls = _build(AppSettings())

    assert session.restoration_pipeline is pipeline_cls.return_value
    assert restorer_cls.call_args.kwargs["use_tensorrt"] is True
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is None
    assert compiled.call_args.args[0].unet4x is False


def test_build_video_session_selects_unet_secondary() -> None:
    settings = replace(AppSettings(), secondary_restoration="unet-4x")
    session, compiled, _restorer_cls, pipeline_cls, unet_cls = _build(settings)

    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is unet_cls.return_value
    assert compiled.call_args.args[0].unet4x is True


def test_video_session_close_closes_restorers() -> None:
    session, *_ = _build(replace(AppSettings(), secondary_restoration="unet-4x"))

    session.close()

    session.restoration_pipeline.restorer.close.assert_called_once_with()
    session.restoration_pipeline.secondary_restorer.close.assert_called_once_with()


def _config(settings: AppSettings):
    with (
        patch("jasna.engine_paths.model_weights_dir", return_value=Path("weights")),
        patch("jasna.mosaic.detection_registry.coerce_detection_model_name", side_effect=lambda name: name),
        patch("jasna.mosaic.detection_registry.require_detection_model_weights", return_value=Path("det.engine")),
    ):
        return video_session_config(settings, codec="hevc", encoder_settings={})


def test_video_session_config_maps_ltx_settings() -> None:
    config = _config(
        replace(AppSettings(), restoration_model="ltx", ltx_seed=7, ltx_fast=True, ltx_large_canvas=True)
    )

    assert config.restoration_model_name == "ltx"
    assert config.restoration_model_path == Path("weights") / "ltx-restore"
    assert (config.ltx_seed, config.ltx_fast, config.ltx_large_canvas) == (7, True, True)


def test_video_session_config_ignores_standard_only_extras_for_ltx() -> None:
    settings = replace(
        AppSettings(),
        restoration_model="ltx",
        secondary_restoration="tvai",
        tvai_denoise=True,
        denoise_strength="high",
    )

    config = _config(settings)

    assert (config.secondary_restoration, config.tvai_denoise, config.denoise_strength) == ("none", False, "none")
    standard = _config(replace(settings, restoration_model="basicvsrpp"))
    assert (standard.secondary_restoration, standard.tvai_denoise, standard.denoise_strength) == ("tvai", True, "high")


def test_video_session_key_tracks_model_and_ltx_variant() -> None:
    ltx = replace(AppSettings(), restoration_model="ltx")

    assert video_session_key(ltx) != video_session_key(AppSettings())
    assert video_session_key(replace(ltx, ltx_fast=True)) != video_session_key(ltx)
    assert video_session_key(replace(ltx, ltx_seed=1, ltx_large_canvas=True)) == video_session_key(ltx)
