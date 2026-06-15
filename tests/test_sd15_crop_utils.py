from __future__ import annotations

import numpy as np

from jasna.sd15_crop_utils import compute_crop_bbox, crop_and_resize_np, paste_back, reflect_pad_to_size


class TestComputeCropBbox:
    def test_square_and_in_frame(self):
        cx1, cy1, cx2, cy2 = compute_crop_bbox([100, 100, 180, 180], 640, 480, 512)
        assert 0 <= cx1 < cx2 <= 640
        assert 0 <= cy1 < cy2 <= 480
        # Centered square unless clamped by an edge (here it fits).
        assert (cx2 - cx1) == (cy2 - cy1)

    def test_clamped_at_edge(self):
        cx1, cy1, cx2, cy2 = compute_crop_bbox([0, 0, 40, 40], 640, 480, 512)
        assert cx1 == 0 and cy1 == 0
        assert cx2 <= 640 and cy2 <= 480


class TestCropAndResize:
    def test_returns_target_square_and_content_dims(self):
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        crop_bbox = compute_crop_bbox([100, 100, 300, 260], 640, 480, 512)
        crop, ch, cw = crop_and_resize_np(img, crop_bbox, 512)
        assert crop.shape == (512, 512, 3)
        assert ch == crop_bbox[3] - crop_bbox[1]
        assert cw == crop_bbox[2] - crop_bbox[0]

    def test_mask_2d(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        crop_bbox = compute_crop_bbox([100, 100, 200, 200], 640, 480, 512)
        crop, _, _ = crop_and_resize_np(mask, crop_bbox, 512)
        assert crop.shape == (512, 512)


class TestReflectPad:
    def test_pads_to_square_target(self):
        arr = np.zeros((300, 200, 3), dtype=np.uint8)
        out = reflect_pad_to_size(arr, 512)
        assert out.shape[0] >= 512 and out.shape[1] >= 512


class TestPasteBack:
    def test_hard_mask_writes_region(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        restored = np.full((512, 512, 3), 200, dtype=np.uint8)
        crop_bbox = (100, 100, 300, 260)  # content 160h x 200w
        content_h, content_w = 160, 200
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[120:200, 120:220] = 255
        paste_back(canvas, restored, crop_bbox, content_h, content_w, mask)
        assert int(canvas[150, 150].max()) == 200   # inside mask -> filled
        assert int(canvas[10, 10].max()) == 0        # outside -> untouched
