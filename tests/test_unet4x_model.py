from __future__ import annotations

import onnx
import pytest
import torch

from jasna.models.unet4x import Unet4x, match_color


def test_color_matching_preserves_an_image_with_matching_statistics():
    image = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    torch.testing.assert_close(match_color(image, image), image)


def test_color_matching_adjusts_contrast_without_changing_spatial_order():
    image = torch.tensor([[[[0.25, 0.75]]]])
    reference = torch.tensor([[[[0.4, 0.6]]]])
    torch.testing.assert_close(match_color(image, reference), torch.tensor([[[[0.37, 0.63]]]]))


def test_color_matching_handles_flat_channels_independently():
    image = torch.full((1, 3, 2, 2), 0.25, dtype=torch.float16)
    reference = torch.tensor([0.25, 0.5, 0.75])[:, None, None].expand(3, 2, 2)[None]
    matched = match_color(image, reference)
    torch.testing.assert_close(matched[0, :, 0, 0], torch.tensor([0.25, 0.45, 0.65]))
    assert matched.isfinite().all()


def test_unet_rejects_an_unrelated_onnx_model():
    graph = onnx.helper.make_graph([], "empty", [], [])
    with pytest.raises(ValueError, match="Unsupported UNet 4x ONNX architecture"):
        Unet4x(onnx.helper.make_model(graph).SerializeToString())
