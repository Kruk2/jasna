from __future__ import annotations

import pytest
import torch

from jasna.crop_buffer import RESTORATION_SIZE, CropBuffer, RawCrop, prepare_crops_for_restoration


def _make_raw_crop(value: int = 0) -> RawCrop:
    return RawCrop(
        crop=torch.full((3, 4, 4), value, dtype=torch.uint8),
        enlarged_bbox=(0, 0, 4, 4),
        crop_shape=(4, 4),
    )


class TestCropBufferSplitOverlap:
    def test_split_creates_child_with_overlap_crops(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(5):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=3)

        assert child.track_id == 1
        assert child.start_frame == 3
        assert child.frame_count == 2
        assert child.crops[0].crop.unique().item() == 3
        assert child.crops[1].crop.unique().item() == 4

    def test_split_does_not_modify_parent(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(5):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=3)

        assert parent.frame_count == 5
        assert child.frame_count == 2

    def test_split_child_is_shallow_copy(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(3):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=1)

        assert child.crops[0] is parent.crops[1]
        assert child.crops[1] is parent.crops[2]

    def test_mutating_child_does_not_affect_parent(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(3):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=1)
        child.add(_make_raw_crop(value=99))

        assert parent.frame_count == 3
        assert child.frame_count == 3

    def test_split_full_overlap(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(3):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=3, new_track_id=1, new_start_frame=0)
        assert child.frame_count == 3


class TestPrepareCropsDtype:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_output_dtype_follows_requested_precision(self, dtype: torch.dtype) -> None:
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("fp16 interpolate requires CUDA")
        device = torch.device("cuda" if dtype == torch.float16 else "cpu")

        crop = torch.randint(0, 256, (3, 40, 50), dtype=torch.uint8)
        raw = RawCrop(crop=crop, enlarged_bbox=(0, 0, 50, 40), crop_shape=(40, 50))

        crops, _, _ = prepare_crops_for_restoration([raw], device, dtype)

        assert len(crops) == 1
        out = crops[0]
        assert out.dtype == dtype
        assert out.shape == (3, RESTORATION_SIZE, RESTORATION_SIZE)
        assert out.device.type == device.type
        assert float(out.max()) <= 255.0
