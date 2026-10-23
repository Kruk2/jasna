from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.engine_paths import default_restoration_model_path
from jasna.segments import SegmentRange, SegmentRestoration
from factories import session_config
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline, build_restoration_session

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


_SECONDARY_RESTORER_CLASSES = {
    "tvai": "jasna.restorer.tvai_secondary_restorer.TvaiSecondaryRestorer",
    "unet-4x": "jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer",
    "rtx-super-res": "jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer",
}


def _build_session(
    config: SessionConfig,
    *,
    amd: bool = False,
):
    """Mocks the configured secondary restorer only: the TensorRT ones import only on NVIDIA builds."""
    compile_result = MagicMock(use_basicvsrpp_tensorrt=True)
    secondary_class = _SECONDARY_RESTORER_CLASSES.get(config.secondary_restoration)
    with (
        patch("jasna.accelerator.is_amd_device", return_value=amd),
        patch(
            "jasna.engine_compiler.ensure_engines_compiled",
            return_value=compile_result,
        ) as compiled,
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_cls,
        patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as pipeline_cls,
        patch(secondary_class) if secondary_class else nullcontext() as secondary_cls,
    ):
        session = build_restoration_session(
            config,
            log_callback=None,
        )
    return session, compiled, restorer_cls, pipeline_cls, secondary_cls


def test_session_without_secondary() -> None:
    session, compiled, restorer_cls, pipeline_cls, *_ = _build_session(session_config())

    assert session.restoration_pipeline is pipeline_cls.return_value
    request = compiled.call_args.args[0]
    assert request.basicvsrpp is True
    assert request.basicvsrpp_model_path == "restore.pth"
    assert request.detection is True
    assert request.detection_model_name == "rfdetr-v5"
    assert request.detection_batch_size == 4
    assert request.unet4x is False
    assert restorer_cls.call_args.kwargs["use_tensorrt"] is True
    assert restorer_cls.call_args.kwargs["max_clip_size"] == 90
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is None


def test_session_selects_tvai_secondary() -> None:
    session, _, _, pipeline_cls, tvai_cls = _build_session(
        session_config(secondary_restoration="tvai", tvai_scale=2, tvai_workers=1, tvai_denoise=True)
    )

    kwargs = tvai_cls.call_args.kwargs
    assert kwargs["ffmpeg_path"] == "ffmpeg.exe"
    assert kwargs["tvai_args"] == "model=iris-2:scale=2:noise=0"
    assert kwargs["scale"] == 2
    assert kwargs["num_workers"] == 1
    assert kwargs["tvai_denoise"] is True
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is tvai_cls.return_value


def test_tvai_denoise_requires_tvai_secondary() -> None:
    with pytest.raises(ValueError, match="requires secondary restoration 'tvai'"):
        _build_session(session_config(tvai_denoise=True))


def test_session_selects_unet_secondary() -> None:
    pytest.importorskip("tensorrt")
    session, compiled, _, pipeline_cls, unet_cls = _build_session(
        session_config(secondary_restoration="unet-4x")
    )

    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is unet_cls.return_value
    assert compiled.call_args.args[0].unet4x is True
    assert unet_cls.call_args.kwargs["fp16"] is True


def test_session_selects_rtx_secondary_and_maps_none_levels() -> None:
    pytest.importorskip("tensorrt")
    session, _, _, pipeline_cls, rtx_cls = _build_session(
        session_config(secondary_restoration="rtx-super-res", rtx_denoise="none", rtx_deblur="low")
    )

    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is rtx_cls.return_value
    kwargs = rtx_cls.call_args.kwargs
    assert kwargs["scale"] == 4
    assert kwargs["quality"] == "high"
    assert kwargs["denoise"] is None
    assert kwargs["deblur"] == "low"


def test_amd_rejects_secondary_restoration() -> None:
    with pytest.raises(ValueError, match="not available in the AMD build"):
        _build_session(session_config(secondary_restoration="tvai"), amd=True)


def test_amd_disables_basicvsrpp_compilation() -> None:
    _, compiled, *_ = _build_session(session_config(), amd=True)

    assert compiled.call_args.args[0].basicvsrpp is False


def test_session_close_closes_restorers() -> None:
    pytest.importorskip("tensorrt")
    session, *_ = _build_session(session_config(secondary_restoration="unet-4x"))

    session.close()

    session.restoration_pipeline.restorer.close.assert_called_once_with()
    session.restoration_pipeline.secondary_restorer.close.assert_called_once_with()


def test_build_pipeline_passes_through_config_and_session() -> None:
    config = session_config()
    session = RestorationSession(device=MagicMock(), restoration_pipeline=MagicMock())
    segments = (SegmentRange(1, 2),)
    splice_plan = MagicMock()
    progress_callback = MagicMock()

    with patch("jasna.pipeline.Pipeline") as pipeline_cls:
        pipeline = build_pipeline(
            config,
            session,
            Path("in.mp4"),
            Path("out.mp4"),
            progress_callback=progress_callback,
            segments=segments,
            splice_plan=splice_plan,
        )

    assert pipeline is pipeline_cls.return_value
    assert pipeline_cls.call_args.kwargs == dict(
        config=config,
        session=session,
        input_video=Path("in.mp4"),
        output_video=Path("out.mp4"),
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
    )


def test_session_reuses_its_detector_until_detection_settings_change() -> None:
    session = RestorationSession(device=MagicMock(), restoration_pipeline=MagicMock())

    with patch("jasna.mosaic.detection_registry.build_detection_model") as build:
        build.side_effect = lambda *args, **kwargs: MagicMock()
        first = session.detection_model_for(session_config())
        again = session.detection_model_for(session_config(max_clip_size=30))
        changed = session.detection_model_for(session_config(detection_score_threshold=0.5))

    assert again is first
    assert changed is not first
    first.close.assert_called_once_with()
    assert build.call_count == 2
    assert build.call_args.kwargs["score_threshold"] == 0.5

    session.close()
    changed.close.assert_called_once_with()


def test_build_pipeline_loads_a_segment_model_the_session_lacks() -> None:
    session = RestorationSession(device=MagicMock(), restoration_pipeline=MagicMock())
    segments = (SegmentRange(1, 2), SegmentRange(3, 4, SegmentRestoration("ltx", 9)))

    with (
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.transformer.require_kernels"),
        patch("jasna.engine_compiler.ensure_engines_compiled"),
        patch("jasna.ltx.model_files.LtxModelFiles.from_dir") as from_dir,
        patch("jasna.pipeline.Pipeline"),
    ):
        build_pipeline(session_config(), session, Path("in.mp4"), Path("out.mp4"), segments=segments)
        build_pipeline(session_config(), session, Path("in.mp4"), Path("out.mp4"), segments=segments)

    from_dir.assert_called_once_with(default_restoration_model_path("ltx"), "distilled", fast=False)
    assert session.ltx_files is from_dir.return_value


def test_build_pipeline_keeps_a_session_that_holds_the_job_model() -> None:
    pipeline = MagicMock()
    session = RestorationSession(device=MagicMock(), restoration_pipeline=pipeline)

    with patch("jasna.pipeline.Pipeline"):
        build_pipeline(session_config(), session, Path("in.mp4"), Path("out.mp4"), segments=(SegmentRange(1, 2),))

    assert session.restoration_pipeline is pipeline
    assert session.ltx_files is None


def _auto_split(config, *, samples, validate=None, plan_error=None):
    from fractions import Fraction
    from tempfile import TemporaryDirectory

    from jasna.media.splice import KeyframeIndex
    from jasna.session_factory import ltx_auto_split

    session = MagicMock()
    progress = MagicMock()
    with (
        TemporaryDirectory() as temp,
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock(codec_name="hevc", duration=6.0, num_frames=180, time_base=Fraction(1,30))),
        patch("jasna.media.splice.validate_smart_render", side_effect=validate),
        patch("jasna.media.splice.probe_keyframes", return_value=KeyframeIndex((0, 60, 120), Fraction(1, 30), 0, 180)),
        patch("jasna.mosaic_scan.scan_mosaic_sizes", return_value=samples) as scan,
    ):
        source, output = Path(temp) / "in.mp4", Path(temp) / "out.mp4"
        source.touch()
        if plan_error is not None:
            with patch("jasna.media.splice.build_splice_plan", side_effect=plan_error):
                return ltx_auto_split(config, session, source, output, progress_callback=progress), scan
        return ltx_auto_split(config, session, source, output, progress_callback=progress), scan


def _samples(sizes):
    from jasna.mosaic_scan import MosaicSamples

    return MosaicSamples(times=(1.0, 3.0), scores=(0.9, 0.9), sizes=sizes, stride=1.0)


def test_ltx_auto_split_sends_big_mosaics_to_ltx_and_keeps_the_source_codec() -> None:
    config = session_config(restoration_model_name="ltx", ltx_seed=3, ltx_auto_split_percent=12.0)
    split, scan = _auto_split(config, samples=_samples((0.05, 0.3)))
    assert split.config == config
    assert split.segments == (
        SegmentRange(0.5, 2.0, SegmentRestoration("basicvsrpp", None)),
        SegmentRange(2.0, 4.5, SegmentRestoration("ltx", 3)),
    )
    assert len(split.splice_plan.render_spans) == 2
    assert scan.call_args.kwargs["batch_size"] == config.batch_size


def test_ltx_auto_split_gives_the_source_codec_its_default_cq_not_settings_meant_for_another(caplog) -> None:
    from jasna.accelerator import vendor_for_device
    from jasna.media.encoder_settings import encoder_cq_spec

    config = session_config(
        restoration_model_name="ltx", ltx_auto_split_percent=12.0, codec="av1", encoder_settings={"cq": 38, "tune": "uhq"}
    )
    split, _ = _auto_split(config, samples=_samples((0.05, 0.3)))
    default_cq = encoder_cq_spec("hevc", vendor_for_device(config.device)).default
    assert (split.config.codec, split.config.encoder_settings) == ("hevc", {"cq": default_cq})
    assert "source codec HEVC" in caplog.text and "for AV1" in caplog.text


@pytest.mark.parametrize(
    "overrides",
    [dict(restoration_model_name="ltx", ltx_auto_split_percent=None), dict(ltx_auto_split_percent=12.0)],
)
def test_ltx_auto_split_is_off_without_the_setting_or_for_basicvsrpp(overrides) -> None:
    split, scan = _auto_split(session_config(**overrides), samples=_samples((0.3, 0.3)))
    assert split is None
    scan.assert_not_called()


def test_ltx_auto_split_rejects_vr_side_by_side_before_scanning() -> None:
    from fractions import Fraction

    from jasna.media.probe import UnsupportedVideoError
    from jasna.session_factory import ltx_auto_split

    config = session_config(restoration_model_name="ltx", ltx_auto_split_percent=12.0, vr_mode="sbs")
    metadata = MagicMock(
        codec_name="hevc", duration=6.0, video_width=3840, video_height=1920, sample_aspect_ratio=Fraction(1, 1)
    )
    with (
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes") as keyframes,
        patch("jasna.mosaic_scan.scan_mosaic_sizes") as scan,
        pytest.raises(UnsupportedVideoError, match="VR180 side-by-side"),
    ):
        ltx_auto_split(config, MagicMock(), Path("in.mp4"), Path("out.mp4"), progress_callback=None)
    keyframes.assert_not_called()
    scan.assert_not_called()


def test_ltx_auto_split_falls_back_to_the_whole_video(caplog) -> None:
    from jasna.media.splice import SmartRenderCompatibilityError

    config = session_config(restoration_model_name="ltx", ltx_auto_split_percent=12.0)
    split, scan = _auto_split(config, samples=_samples((0.3, 0.3)), validate=SmartRenderCompatibilityError("VFR"))
    assert split is None
    scan.assert_not_called()
    assert "VFR" in caplog.text

    split, _ = _auto_split(config, samples=_samples((0.3, 0.3)), plan_error=SmartRenderCompatibilityError("one span"))
    assert split is None

    split, _ = _auto_split(config, samples=MagicMock(times=(1.0,), scores=(0.0,), sizes=(0.0,), stride=1.0))
    assert split is None
    assert "no mosaics" in caplog.text


@pytest.mark.parametrize(
    ("nvidia", "capability", "allowed"),
    [(True, (12, 0), True), (True, (8, 9), False), (False, (12, 0), False)],
)
def test_fast_ltx_needs_a_blackwell_nvidia_gpu(nvidia, capability, allowed) -> None:
    import torch

    from jasna.session_factory import _ltx_model_files

    config = session_config(restoration_model_name="ltx", ltx_fast=True, ltx_trial=True)
    with (
        patch("jasna.ltx.transformer.require_kernels"),
        patch("jasna.accelerator.is_nvidia_device", return_value=nvidia),
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch("jasna.engine_compiler.ensure_engines_compiled") as compiled,
    ):
        if allowed:
            _ltx_model_files(config, torch.device("cuda:0"), log_callback=None)
            compiled.assert_called_once()
        else:
            with pytest.raises(ValueError, match="Blackwell"):
                _ltx_model_files(config, torch.device("cuda:0"), log_callback=None)


def test_fast_ltx_refuses_unsupported_hardware_before_loading_kernels(monkeypatch):
    import torch
    from jasna.session_factory import _ltx_model_files

    monkeypatch.setattr("jasna.accelerator.is_nvidia_device", lambda device: False)
    monkeypatch.setattr("jasna.ltx.transformer.require_kernels", lambda device: pytest.fail("loaded unsupported kernels"))
    config = session_config(restoration_model_name="ltx", ltx_fast=True, ltx_trial=True)
    with pytest.raises(ValueError, match="Blackwell"):
        _ltx_model_files(config, torch.device("cuda:0"), log_callback=None)
