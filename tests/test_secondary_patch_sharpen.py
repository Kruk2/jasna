"""The AMD secondary's sharpness box must reach the blend, and only that patch."""
from __future__ import annotations

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import SecondaryRestoreResult
from jasna.session_config import SessionConfig
from jasna.session_factory import _secondary_patch_sharpen
from tests.factories import session_config


def _config(**kwargs) -> SessionConfig:
    base = session_config(secondary_restoration="amd-upscale")
    return SessionConfig(**{**base.__dict__, **kwargs})


def test_sharpness_reaches_the_pipeline_only_for_amd_secondary() -> None:
    assert _secondary_patch_sharpen(
        _config(secondary_restoration="amd-upscale", amd_upscale_sharpness=0.35)
    ) == 0.35
    assert _secondary_patch_sharpen(
        _config(secondary_restoration="tvai", amd_upscale_sharpness=0.35)
    ) == 0.0


def test_sharpness_is_clamped_to_the_global_scale() -> None:
    assert _secondary_patch_sharpen(
        _config(secondary_restoration="amd-upscale", amd_upscale_sharpness=2.5)
    ) == 1.0
    assert _secondary_patch_sharpen(
        _config(secondary_restoration="amd-upscale", amd_upscale_sharpness=-1.0)
    ) == 0.0


def _result(*, sharpen_strength: float, fill: int = 40) -> SecondaryRestoreResult:
    return SecondaryRestoreResult(
        track_id=0,
        start_frame=0,
        frame_count=2,
        frame_shape=(32, 32),
        frame_device=torch.device("cpu"),
        masks=[torch.ones(32, 32), torch.ones(32, 32)],
        restored_frames=[torch.full((3, 8, 8), fill, dtype=torch.uint8)] * 2,
        keep_start=0,
        keep_end=2,
        crossfade_weights=None,
        enlarged_bboxes=[(8, 8, 24, 24)] * 2,
        crop_shapes=[(16, 16)] * 2,
        pad_offsets=[(0, 0)] * 2,
        # Offsets are stored in the 256-px restoration frame, so a full-frame crop
        # of an 8-px restored frame needs resize_shapes (256, 256).
        resize_shapes=[(256, 256)] * 2,
        sharpen_strength=sharpen_strength,
    )


def _blend(monkeypatch, strength: float) -> list[float]:
    from jasna import blend_buffer as mod

    calls: list[float] = []

    def fake_sharpen(patch, value):
        calls.append(float(value))
        return patch

    monkeypatch.setattr(mod, "sharpen_rgb_patch", fake_sharpen)
    buffer = BlendBuffer(torch.device("cpu"))
    original = torch.zeros(3, 32, 32, dtype=torch.uint8)
    blended = original.clone()
    buffer._apply_blend(blended, original, 1, 0, _result(sharpen_strength=strength), original.device)
    return calls


def test_blend_sharpens_exactly_once_with_the_result_strength(monkeypatch) -> None:
    assert _blend(monkeypatch, 0.4) == [0.4]


def test_blend_skips_the_sharpener_when_the_strength_is_zero(monkeypatch) -> None:
    assert _blend(monkeypatch, 0.0) == []
