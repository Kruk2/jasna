import torch
import pytest

from jasna.mosaic.yolo import _letterbox_normalized_bchw


def test_letterbox_normalized_pads_with_114_over_255() -> None:
    x = torch.ones((2, 3, 100, 200), dtype=torch.float32)
    out, _ = _letterbox_normalized_bchw(x, new_shape=(608, 1056), stride=32)
    assert out.shape[0] == 2
    assert out.shape[1] == 3
    assert out.shape[2] % 32 == 0
    assert out.shape[3] % 32 == 0

    # Some padding must exist (aspect ratio differs)
    assert out.shape[2] >= 100 and out.shape[3] >= 200

    # Top-left corner is padding; should be the letterbox pad value.
    pad_val = 114.0 / 255.0
    assert float(out[0, 0, 0, 0].item()) == pytest.approx(pad_val)

