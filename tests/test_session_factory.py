from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import requires_nvidia
from jasna.engine_paths import default_restoration_model_path
from jasna.segments import SegmentRange, SegmentRestoration
from factories import session_config
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline, build_restoration_session


def _enter_if_available(stack: ExitStack, target: str):
    """Enter ``patch(target)`` on ``stack``, or return None when it is absent.

    `unet4x_secondary_restorer` and `rtx_superres_secondary_restorer` import
    TensorRT, which the AMD build does not carry. Mock resolves the target when
    the context manager is entered, so the guard has to wrap the enter, not the
    construction. The tests that exercise those two modes are NVIDIA-only and
    carry the `requires_nvidia` marker.
    """
    try:
        return stack.enter_context(patch(target))
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None


def _build_session(
    config: SessionConfig,
    *,
    amd: bool = False,
):
    compile_result = MagicMock(use_basicvsrpp_tensorrt=True)
    unet_cls = rtx_cls = None
    with ExitStack() as stack:
        stack.enter_context(patch("jasna.accelerator.is_amd_device", return_value=amd))
        compiled = stack.enter_context(
            patch(
                "jasna.engine_compiler.ensure_engines_compiled",
                return_value=compile_result,
            )
        )
        restorer_cls = stack.enter_context(
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer")
        )
        pipeline_cls = stack.enter_context(
            patch("jasna.restorer.restoration_pipeline.RestorationPipeline")
        )
        tvai_cls = stack.enter_context(
            patch("jasna.restorer.tvai_secondary_restorer.TvaiSecondaryRestorer")
        )
        unet_cls = _enter_if_available(
            stack, "jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer"
        )
        rtx_cls = _enter_if_available(
            stack, "jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer"
        )
        session = build_restoration_session(
            config,
            log_callback=None,
        )
    return session, compiled, restorer_cls, pipeline_cls, tvai_cls, unet_cls, rtx_cls


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
    session, _, _, pipeline_cls, tvai_cls, *_ = _build_session(
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


@requires_nvidia
def test_session_selects_unet_secondary() -> None:
    session, compiled, _, pipeline_cls, _, unet_cls, _ = _build_session(
        session_config(secondary_restoration="unet-4x")
    )

    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is unet_cls.return_value
    assert compiled.call_args.args[0].unet4x is True
    assert unet_cls.call_args.kwargs["fp16"] is True


@requires_nvidia
def test_session_selects_rtx_secondary_and_maps_none_levels() -> None:
    session, _, _, pipeline_cls, _, _, rtx_cls = _build_session(
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


@requires_nvidia
def test_session_close_closes_restorers() -> None:
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
    # Detectors stay resident in the shared registry: a settings change builds a
    # replacement and leaves the previous one in VRAM for the next session, so
    # neither is closed here - release_shared_models() is what frees them.
    first.close.assert_not_called()
    assert build.call_count == 2
    assert build.call_args.kwargs["score_threshold"] == 0.5

    session.close()
    changed.close.assert_not_called()


def test_build_pipeline_loads_a_segment_model_the_session_lacks() -> None:
    session = RestorationSession(device=MagicMock(), restoration_pipeline=MagicMock())
    segments = (SegmentRange(1, 2), SegmentRange(3, 4, SegmentRestoration("ltx", 9)))

    with (
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
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
