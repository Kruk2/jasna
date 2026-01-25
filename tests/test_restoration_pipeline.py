import numpy as np
import torch

from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import TrackedClip


class _IdentityRestorer:
    dtype = torch.float32

    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        return crops


def test_restore_clip_uses_floor_ceil_xyxy_rounding(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    # Disable expansion so we can assert pure xyxy rounding + slicing.
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frame = torch.zeros((3, 10, 10), dtype=torch.uint8)
    bbox = np.array([2.1, 2.1, 6.2, 6.2], dtype=np.float32)  # xyxy floats
    mask = torch.zeros((4, 4), dtype=torch.bool)

    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(4, 4),
        bboxes=[bbox],
        masks=[mask],
    )

    restored = pipeline.restore_clip(clip, [frame])

    # floor(x1/y1)=2, ceil(x2/y2)=7; xyxy are exclusive for slicing.
    assert restored.enlarged_bboxes == [(2, 2, 7, 7)]
    assert restored.crop_shapes == [(5, 5)]
    assert len(restored.padded_masks) == 1
    assert restored.padded_masks[0].shape == (256, 256)


def test_restore_clip_drops_prefix_frames_from_output(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frame = torch.zeros((3, 10, 10), dtype=torch.uint8)
    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(4, 4),
        bboxes=[bbox],
        masks=[mask],
    )

    prefix = [torch.zeros((3, 256, 256), dtype=torch.uint8) for _ in range(3)]
    restored = pipeline.restore_clip(clip, [frame], prefix_restored_frames=prefix)
    assert len(restored.restored_frames) == 1
    assert len(restored.padded_masks) == 1

