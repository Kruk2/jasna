from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.gui.models import AppSettings
from jasna.gui.video_session import (
    build_video_session,
    video_session_config,
    video_session_key,
)

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


@pytest.mark.parametrize(
    ("changes", "same_session"),
    [
        ({}, True),
        ({"detection_model": "lada-yolo-v2"}, False),
        ({"secondary_restoration": "unet-4x"}, False),
        ({"max_clip_size": 60}, False),
        ({"fp16_mode": False}, False),
        ({"encoder_cq": 30, "codec": "h264"}, True),
        ({"vr_mode": "off"}, True),
        ({"vr_mode": "sbs-fisheye"}, True),
        ({"vr_projection": "fisheye"}, True),
        ({"ltx_vae_encoder": "full"}, False),
        ({"ltx_vae_decoder": "half"}, False),
        ({"decode_backend": "pyav-sw"}, True),
    ],
)
def test_video_session_key_tracks_only_session_fields(changes, same_session) -> None:
    assert (video_session_key(replace(AppSettings(), **changes)) == video_session_key(AppSettings())) is same_session


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
        patch("jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer")
        if settings.secondary_restoration == "unet-4x"
        else nullcontext() as unet_cls,
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
    pytest.importorskip("tensorrt")
    settings = replace(AppSettings(), secondary_restoration="unet-4x")
    session, compiled, _restorer_cls, pipeline_cls, unet_cls = _build(settings)

    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is unet_cls.return_value
    assert compiled.call_args.args[0].unet4x is True


def test_video_session_close_closes_restorers() -> None:
    pytest.importorskip("tensorrt")
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
    assert config.ltx_window_hop == 104


def test_video_session_config_maps_more_ltx_overlap_to_a_shorter_window_hop() -> None:
    config = _config(replace(AppSettings(), restoration_model="ltx", ltx_overlap=33))

    assert config.ltx_window_hop == 88


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
    auto_split = _config(replace(settings, ltx_auto_split=True))
    assert (auto_split.secondary_restoration, auto_split.tvai_denoise, auto_split.denoise_strength) == (
        "tvai", True, "none"
    )


def test_video_session_key_tracks_model_and_ltx_variant() -> None:
    ltx = replace(AppSettings(), restoration_model="ltx")

    assert video_session_key(ltx) != video_session_key(AppSettings())
    assert video_session_key(replace(ltx, ltx_fast=True)) != video_session_key(ltx)
    assert video_session_key(replace(ltx, ltx_seed=1, ltx_large_canvas=True)) == video_session_key(ltx)


def test_video_session_key_tracks_the_ltx_variant_for_ltx_ranges_of_standard_jobs() -> None:
    assert video_session_key(replace(AppSettings(), ltx_fast=True)) != video_session_key(AppSettings())


def test_video_session_config_and_key_follow_the_ltx_model() -> None:
    undistilled = replace(AppSettings(), restoration_model="ltx", ltx_model="undistilled")

    assert _config(undistilled).ltx_model == "undistilled"
    assert _config(replace(undistilled, ltx_model="distilled")).ltx_model == "distilled"
    assert video_session_key(undistilled) != video_session_key(replace(undistilled, ltx_model="distilled"))
    standard = replace(AppSettings(), ltx_model="undistilled")
    assert video_session_key(standard) != video_session_key(AppSettings())
