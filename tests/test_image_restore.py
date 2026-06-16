from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from jasna.image_restore import (
    clamp_strength,
    group_boxes_by_iou,
    mask_overlay_rgb_chw,
    prepare_image_restore,
    render_prepared_image,
    restore_image,
    variant_output_paths,
)
from jasna.mosaic.detections import Detections


class _FakeDetector:
    def __init__(self, boxes: np.ndarray, masks: torch.Tensor, batch_size: int = 4):
        self.batch_size = batch_size
        self._boxes = boxes
        self._masks = masks
        self.calls = 0

    def __call__(self, batch, *, target_hw):
        self.calls += 1
        assert batch.shape[0] == self.batch_size
        return Detections(boxes_xyxy=[self._boxes], masks=[self._masks])

    def close(self):
        pass


class _FakeRestorer:
    def __init__(self, fill: int = 200):
        self.fill = fill
        self.seeds: list[int] = []

    def restore_crop(self, mosaic_01, mask_01, *, steps, strength, seed, freeu):
        self.seeds.append(seed)
        return torch.full((3, 512, 512), self.fill, dtype=torch.uint8)

    def close(self):
        pass


class TestVariantOutputPaths:
    def test_single_variant_uses_base(self):
        assert variant_output_paths(Path("out.png"), 1) == [Path("out.png")]

    def test_multiple_variants_suffixed(self):
        paths = variant_output_paths(Path("/x/img_out.png"), 3)
        assert [p.name for p in paths] == ["img_out_v1.png", "img_out_v2.png", "img_out_v3.png"]


class TestClampStrength:
    def test_caps_at_07(self):
        assert clamp_strength(0.9) == pytest.approx(0.7)
        assert clamp_strength(0.6) == pytest.approx(0.6)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            clamp_strength(0.0)


class TestGroupBoxesByIou:
    def test_overlapping_grouped(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
        groups = group_boxes_by_iou(boxes, 0.3)
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]

    def test_disjoint_not_grouped(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
        groups = group_boxes_by_iou(boxes, 0.3)
        assert len(groups) == 2


class TestRestoreImage:
    def test_zero_detections_returns_copies(self):
        img = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        detector = _FakeDetector(np.zeros((0, 4), np.float32), torch.zeros(0, 8, 8, dtype=torch.bool))
        restorer = _FakeRestorer()
        outs = restore_image(
            img, detector, restorer, device=torch.device("cpu"), fp16=False,
            steps=5, strength=0.6, seed=3, num_variants=2, freeu=None,
        )
        assert len(outs) == 2
        for o in outs:
            assert np.array_equal(o, img)
        assert restorer.seeds == []

    def test_variants_use_incrementing_seeds_and_paste(self):
        img = np.zeros((3, 64, 64), dtype=np.uint8)
        boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)
        masks = torch.ones(1, 8, 8, dtype=torch.bool)
        detector = _FakeDetector(boxes, masks)
        restorer = _FakeRestorer(fill=200)
        outs = restore_image(
            img, detector, restorer, device=torch.device("cpu"), fp16=False,
            steps=5, strength=0.6, seed=5, num_variants=3, freeu=None,
        )
        assert len(outs) == 3
        assert restorer.seeds == [5, 6, 7]
        # Mask covers the whole frame after upsample+dilate, so the restored fill
        # dominates: output must differ from the all-zero input.
        assert outs[0].max() > 0
        assert int(outs[0][0].max()) == 200


class TestPreparedImageRestore:
    def test_prepare_detects_once_and_render_reuses_prepared_data(self):
        img = np.zeros((3, 64, 64), dtype=np.uint8)
        boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)
        masks = torch.ones(1, 8, 8, dtype=torch.bool)
        detector = _FakeDetector(boxes, masks)
        restorer = _FakeRestorer(fill=180)

        prepared = prepare_image_restore(img, detector, device=torch.device("cpu"), fp16=False)
        assert detector.calls == 1
        assert prepared.has_detections

        out_a = render_prepared_image(prepared, restorer, steps=5, strength=0.6, seed=7, freeu=None)
        out_b = render_prepared_image(prepared, restorer, steps=5, strength=0.6, seed=8, freeu=None)

        assert detector.calls == 1
        assert restorer.seeds == [7, 8]
        assert int(out_a[0].max()) == 180
        assert int(out_b[0].max()) == 180

    def test_zero_detection_prepared_render_returns_copy(self):
        img = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        detector = _FakeDetector(np.zeros((0, 4), np.float32), torch.zeros(0, 8, 8, dtype=torch.bool))
        restorer = _FakeRestorer()

        prepared = prepare_image_restore(img, detector, device=torch.device("cpu"), fp16=False)
        out = render_prepared_image(prepared, restorer, steps=5, strength=0.6, seed=7, freeu=None)

        assert not prepared.has_detections
        assert np.array_equal(out, img)
        assert out is not img
        assert restorer.seeds == []

    def test_mask_overlay_tints_detected_area_red(self):
        img = np.zeros((3, 64, 64), dtype=np.uint8)
        boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)
        masks = torch.ones(1, 8, 8, dtype=torch.bool)
        detector = _FakeDetector(boxes, masks)

        prepared = prepare_image_restore(img, detector, device=torch.device("cpu"), fp16=False)
        overlay = mask_overlay_rgb_chw(prepared)

        assert overlay.shape == img.shape
        assert overlay[0].max() > 0
        assert overlay[1].max() == 0
        assert overlay[2].max() == 0
