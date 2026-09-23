"""Builders shared by tests that need a SessionConfig or a Pipeline."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from jasna.pipeline import Pipeline
from jasna.session_config import SessionConfig


def session_config(**overrides) -> SessionConfig:
    base = dict(
        device="cuda:0",
        fp16=True,
        batch_size=4,
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        detection_score_threshold=0.25,
        max_detection_gap=2,
        min_detection_duration=2,
        scene_detection=True,
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


def fake_session(*, device: torch.device, restoration_pipeline, detection_model=None) -> SimpleNamespace:
    detection_model = MagicMock() if detection_model is None else detection_model
    return SimpleNamespace(
        device=device,
        restoration_pipeline=restoration_pipeline,
        detection_model_for=lambda config: detection_model,
    )


def make_pipeline(
    *,
    device: torch.device = torch.device("cpu"),
    restoration_pipeline=None,
    detection_model=None,
    input_video: Path = Path("in.mp4"),
    output_video: Path = Path("out.mkv"),
    progress_callback=None,
    segments=None,
    splice_plan=None,
    **config_overrides,
) -> Pipeline:
    if restoration_pipeline is None:
        restoration_pipeline = MagicMock(
            secondary_restorer=None,
            secondary_num_workers=1,
            secondary_prefers_cpu_input=False,
        )
    return Pipeline(
        config=session_config(**config_overrides),
        session=fake_session(
            device=device,
            restoration_pipeline=restoration_pipeline,
            detection_model=detection_model,
        ),
        input_video=input_video,
        output_video=output_video,
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
    )
