import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ltx_encode as bench

from jasna.models.ltx_vae.loader import encoder_kwargs
from jasna.models.ltx_vae.video_encoder import ConvLaunch, VideoEncoder


def test_unknown_variant_option_is_rejected():
    with pytest.raises(ValueError, match="fast"):
        bench.variant_encoder("eager+fast", vae=None, device=None)


def test_psnr_is_over_the_reference_latent_range():
    reference = torch.tensor([0.0, 2.0, 1.0, 1.0])
    assert bench.psnr(reference, reference) == float("inf")
    assert bench.psnr(reference + 0.1, reference) == pytest.approx(10 * torch.log10(torch.tensor(4 / 0.01)).item(), rel=1e-4)


def test_head_options_take_their_block_count():
    assert bench.head_blocks(["cl3d", "kitchen6", "compile"], "kitchen") == 6
    assert bench.head_blocks(["cl3d", "fp16head2"], "kitchen") is None


def test_a_head_option_without_a_block_count_is_rejected():
    with pytest.raises(ValueError, match="kitchen"):
        bench.variant_encoder("kitchen+cl3d", vae=None, device=None)


@pytest.mark.parametrize(("variant", "kitchen"), [("kitchen1+cl3d", True), ("fp16head1+cl3d", False)])
def test_head_variants_leave_the_shipped_triton_convs_off(monkeypatch, variant, kitchen):
    blocks = [["res_x", {"num_layers": 1}], ["res_x", {"num_layers": 1}]]
    config = {"_class_name": "Encoder", "in_channels": 3, "out_channels": 8, "patch_size": 4, "blocks": blocks,
              "dims": 3, "norm_layer": "pixel_norm", "latent_log_var": "constant", "spatial_padding_mode": "zeros"}
    shipped = VideoEncoder(**encoder_kwargs({"encoder": config}))
    for conv in shipped.modules():
        if isinstance(conv, bench.CausalConv3d):
            conv.triton_launch = ConvLaunch(64, 8, 8, 4, 2, 1)
    monkeypatch.setattr(bench, "load_video_encoder", lambda vae, device: shipped)
    monkeypatch.setattr(bench, "compile_static", lambda fn, mode, device, dtype: fn)

    bench.variant_encoder(variant, vae=None, device=torch.device("cpu"))

    convs = [conv for conv in shipped.modules() if isinstance(conv, bench.CausalConv3d)]
    assert [conv.kitchen for conv in convs] == [kitchen] * len(convs)
    assert all(conv.triton_launch is None for conv in convs)
