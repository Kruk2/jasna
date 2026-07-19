from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.segments import SegmentRange
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline, build_restoration_session


def _config(**overrides) -> SessionConfig:
    base = dict(
        device="cuda:0",
        fp16=True,
        batch_size=4,
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        detection_score_threshold=0.25,
        max_detection_gap=2,
        min_detection_duration=2,
        restoration_model_path=Path("restore.pth"),
        compile_basicvsrpp=True,
        max_clip_size=90,
        temporal_overlap=8,
        enable_crossfade=True,
        denoise_strength="none",
        denoise_step="after_primary",
        secondary_restoration="none",
        tvai_ffmpeg_path="ffmpeg.exe",
        tvai_model="iris-2",
        tvai_scale=4,
        tvai_args="noise=0",
        tvai_workers=2,
        rtx_scale=4,
        rtx_quality="high",
        rtx_denoise="medium",
        rtx_deblur="none",
        vr_mode="auto",
        codec="hevc",
        encoder_settings={"cq": 25},
        lut_path=None,
        retarget_high_fps=False,
        disable_progress=False,
        working_dir=None,
    )
    base.update(overrides)
    return SessionConfig(**base)


def _build_session(
    config: SessionConfig,
    *,
    disable_basicvsrpp_tensorrt: bool = False,
    amd: bool = False,
):
    compile_result = MagicMock(use_basicvsrpp_tensorrt=True)
    with (
        patch("jasna.accelerator.is_amd_device", return_value=amd),
        patch(
            "jasna.engine_compiler.ensure_engines_compiled",
            return_value=compile_result,
        ) as compiled,
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_cls,
        patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as pipeline_cls,
        patch("jasna.restorer.tvai_secondary_restorer.TvaiSecondaryRestorer") as tvai_cls,
        patch("jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer") as unet_cls,
        patch("jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer") as rtx_cls,
    ):
        session = build_restoration_session(
            config,
            disable_basicvsrpp_tensorrt=disable_basicvsrpp_tensorrt,
            log_callback=None,
        )
    return session, compiled, restorer_cls, pipeline_cls, tvai_cls, unet_cls, rtx_cls


def test_session_without_secondary() -> None:
    session, compiled, restorer_cls, pipeline_cls, *_ = _build_session(_config())

    assert session.secondary_restorer is None
    assert session.detection_model_name == "rfdetr-v5"
    assert session.detection_model_path == Path("det.onnx")
    request = compiled.call_args.args[0]
    assert request.basicvsrpp is True
    assert request.basicvsrpp_model_path == "restore.pth"
    assert request.basicvsrpp_max_clip_size == 90
    assert request.detection is True
    assert request.detection_model_name == "rfdetr-v5"
    assert request.detection_batch_size == 4
    assert request.unet4x is False
    assert restorer_cls.call_args.kwargs["use_tensorrt"] is True
    assert restorer_cls.call_args.kwargs["max_clip_size"] == 90
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is None


def test_session_selects_tvai_secondary() -> None:
    session, _, _, pipeline_cls, tvai_cls, *_ = _build_session(
        _config(secondary_restoration="tvai", tvai_scale=2, tvai_workers=1)
    )

    assert session.secondary_restorer is tvai_cls.return_value
    kwargs = tvai_cls.call_args.kwargs
    assert kwargs["ffmpeg_path"] == "ffmpeg.exe"
    assert kwargs["tvai_args"] == "model=iris-2:scale=2:noise=0"
    assert kwargs["scale"] == 2
    assert kwargs["num_workers"] == 1
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is tvai_cls.return_value


def test_session_selects_unet_secondary() -> None:
    session, compiled, _, _, _, unet_cls, _ = _build_session(
        _config(secondary_restoration="unet-4x")
    )

    assert session.secondary_restorer is unet_cls.return_value
    assert compiled.call_args.args[0].unet4x is True
    assert unet_cls.call_args.kwargs["fp16"] is True


def test_session_selects_rtx_secondary_and_maps_none_levels() -> None:
    session, _, _, _, _, _, rtx_cls = _build_session(
        _config(secondary_restoration="rtx-super-res", rtx_denoise="none", rtx_deblur="low")
    )

    assert session.secondary_restorer is rtx_cls.return_value
    kwargs = rtx_cls.call_args.kwargs
    assert kwargs["scale"] == 4
    assert kwargs["quality"] == "high"
    assert kwargs["denoise"] is None
    assert kwargs["deblur"] == "low"


def test_disable_basicvsrpp_tensorrt_gates_compilation() -> None:
    _, compiled, *_ = _build_session(_config(), disable_basicvsrpp_tensorrt=True)

    assert compiled.call_args.args[0].basicvsrpp is False


def test_amd_rejects_secondary_restoration() -> None:
    with pytest.raises(ValueError, match="not available in the AMD build"):
        _build_session(_config(secondary_restoration="tvai"), amd=True)


def test_amd_disables_basicvsrpp_compilation() -> None:
    _, compiled, *_ = _build_session(_config(), amd=True)

    assert compiled.call_args.args[0].basicvsrpp is False


def test_session_close_closes_restorers() -> None:
    session, *_ = _build_session(_config(secondary_restoration="unet-4x"))

    session.close()

    session.restoration_pipeline.restorer.close.assert_called_once_with()
    session.secondary_restorer.close.assert_called_once_with()


def test_build_pipeline_passes_through_config_and_session() -> None:
    config = _config(
        lut_path="lut.cube",
        retarget_high_fps=True,
        disable_progress=True,
        working_dir=Path("/scratch"),
    )
    session = RestorationSession(
        device=MagicMock(),
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        restoration_pipeline=MagicMock(),
        secondary_restorer=None,
    )
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
    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["input_video"] == Path("in.mp4")
    assert kwargs["output_video"] == Path("out.mp4")
    assert kwargs["detection_model_name"] == "rfdetr-v5"
    assert kwargs["detection_model_path"] == Path("det.onnx")
    assert kwargs["detection_score_threshold"] == 0.25
    assert kwargs["restoration_pipeline"] is session.restoration_pipeline
    assert kwargs["codec"] == "hevc"
    assert kwargs["encoder_settings"] == {"cq": 25}
    assert kwargs["batch_size"] == 4
    assert kwargs["device"] is session.device
    assert kwargs["max_clip_size"] == 90
    assert kwargs["temporal_overlap"] == 8
    assert kwargs["max_detection_gap"] == 2
    assert kwargs["min_detection_duration"] == 2
    assert kwargs["enable_crossfade"] is True
    assert kwargs["vr_mode"] == "auto"
    assert kwargs["fp16"] is True
    assert kwargs["disable_progress"] is True
    assert kwargs["progress_callback"] is progress_callback
    assert kwargs["lut_path"] == "lut.cube"
    assert kwargs["retarget_high_fps"] is True
    assert kwargs["segments"] == segments
    assert kwargs["splice_plan"] is splice_plan
    assert kwargs["working_dir"] == Path("/scratch")


def test_build_pipeline_defaults_optional_runtime_inputs() -> None:
    session = RestorationSession(
        device=MagicMock(),
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        restoration_pipeline=MagicMock(),
        secondary_restorer=None,
    )

    with patch("jasna.pipeline.Pipeline") as pipeline_cls:
        build_pipeline(_config(), session, Path("in.mp4"), Path("out.mp4"))

    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["progress_callback"] is None
    assert kwargs["segments"] is None
    assert kwargs["splice_plan"] is None
