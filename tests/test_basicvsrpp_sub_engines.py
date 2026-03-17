from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from jasna.restorer.basicvsrpp_sub_engines import (
    DIRECTIONS,
    FEATURE_SIZE,
    BasicVSRPlusPlusNetSplit,
    _BackboneWrapper,
    _DeformAlignWrapper,
    _UpsampleWrapper,
    _get_inference_generator,
    _sub_engine_dir,
    all_sub_engines_exist,
    get_sub_engine_paths,
    load_sub_engines,
)


def test_sub_engine_dir_uses_stem() -> None:
    path = r"C:\weights\model_v1.2.pth"
    result = _sub_engine_dir(path)
    assert "model_v1.2_sub_engines" in result
    assert result.startswith(r"C:\weights")


def test_get_sub_engine_paths_returns_10_paths() -> None:
    paths = get_sub_engine_paths("model_weights/model.pth", fp16=True)
    assert len(paths) == 10
    for d in DIRECTIONS:
        assert f"backbone_{d}" in paths
        assert f"deform_align_{d}" in paths
    assert "upsample" in paths
    assert "feat_extract" in paths
    for p in paths.values():
        assert p.endswith(".engine")
        assert "fp16" in p


def test_get_sub_engine_paths_fp32() -> None:
    paths = get_sub_engine_paths("model.pth", fp16=False)
    for p in paths.values():
        assert "fp32" in p


def test_all_sub_engines_exist_false_when_missing(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    assert not all_sub_engines_exist(model_path, fp16=True)


def test_all_sub_engines_exist_true_when_all_present(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    paths = get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")
    assert all_sub_engines_exist(model_path, fp16=True)


def test_all_sub_engines_exist_false_when_partial(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    paths = get_sub_engine_paths(model_path, fp16=True)
    items = list(paths.values())
    for p in items[:-1]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")
    assert not all_sub_engines_exist(model_path, fp16=True)


def test_load_sub_engines_returns_none_when_missing(tmp_path: Path) -> None:
    result = load_sub_engines(str(tmp_path / "model.pth"), torch.device("cpu"), fp16=True)
    assert result is None


def test_backbone_wrapper_forward_shape() -> None:
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import ResidualBlocksWithInputConv

    backbone = ResidualBlocksWithInputConv(128, 64, 3)
    wrapper = _BackboneWrapper(backbone)
    x = torch.randn(1, 128, FEATURE_SIZE, FEATURE_SIZE)
    out = wrapper(x)
    assert out.shape == (1, 64, FEATURE_SIZE, FEATURE_SIZE)


def test_upsample_wrapper_forward_shape() -> None:
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import (
        PixelShufflePack,
        ResidualBlocksWithInputConv,
    )

    reconstruction = ResidualBlocksWithInputConv(320, 64, 2)
    upsample1 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
    upsample2 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
    conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
    conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    wrapper = _UpsampleWrapper(reconstruction, upsample1, upsample2, conv_hr, conv_last)
    x = torch.randn(1, 320, FEATURE_SIZE, FEATURE_SIZE)
    out = wrapper(x)
    assert out.shape == (1, 3, FEATURE_SIZE * 4, FEATURE_SIZE * 4)


def test_get_inference_generator_prefers_ema() -> None:
    model = MagicMock()
    model.generator_ema = MagicMock(spec=nn.Module)
    model.generator = MagicMock(spec=nn.Module)
    assert _get_inference_generator(model) is model.generator_ema


def test_get_inference_generator_falls_back_to_generator() -> None:
    model = MagicMock()
    model.generator_ema = None
    model.generator = MagicMock(spec=nn.Module)
    assert _get_inference_generator(model) is model.generator


def test_split_forward_matches_pytorch_forward() -> None:
    """Verify that BasicVSRPlusPlusNetSplit produces the same output as the
    original BasicVSRPlusPlusNet when using PyTorch modules as 'engines'."""
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    torch.manual_seed(42)
    net = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=2, spynet_pretrained=None)
    net.eval()

    backbone_engines = {d: net.backbone[d] for d in DIRECTIONS}

    class _UpsamplePassthrough(nn.Module):
        def __init__(self, parent: BasicVSRPlusPlusNet):
            super().__init__()
            self.parent = parent

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.parent.reconstruction(x)
            x = self.parent.lrelu(self.parent.upsample1(x))
            x = self.parent.lrelu(self.parent.upsample2(x))
            x = self.parent.lrelu(self.parent.conv_hr(x))
            return self.parent.conv_last(x)

    upsample_engine = _UpsamplePassthrough(net)

    deform_align_engines = {
        d: _DeformAlignWrapper(net.deform_align[d]) for d in DIRECTIONS
    }

    split = BasicVSRPlusPlusNetSplit(
        net, backbone_engines, upsample_engine,
        feat_extract_engine=net.feat_extract,
        deform_align_engines=deform_align_engines,
    )
    split.eval()

    T = 3
    lqs = torch.randn(1, T, 3, 256, 256)

    with torch.inference_mode():
        ref = net(lqs)
        out = split(lqs)

    assert ref.shape == out.shape
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5), \
        f"max diff: {(ref - out).abs().max().item()}"
