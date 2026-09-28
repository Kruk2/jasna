import json

import pytest
import torch
from safetensors.torch import save_file

from jasna.models.ltx_vae import diffusion_tiling, load_video_encoder
from jasna.models.ltx_vae.diffusion_video_decoder import DiffusionVideoDecoder
from jasna.models.ltx_vae.loader import decoder_kwargs, decoder_state_dict, encoder_kwargs, encoder_state_dict
from jasna.models.ltx_vae.tiling import DimensionSizeConfig, TileSizeConfig, scale_by_masks_1d
from jasna.models.ltx_vae.video_encoder import VideoEncoder

TINY_ENCODER_CONFIG = {
    "_class_name": "Encoder",
    "dims": 3,
    "in_channels": 3,
    "out_channels": 8,
    "blocks": [
        ["res_x", {"num_layers": 1}],
        ["compress_space_res", {"multiplier": 2}],
        ["compress_time_res", {"multiplier": 2}],
        ["compress_all_res", {"multiplier": 2}],
        ["compress_all_res", {"multiplier": 1}],
    ],
    "patch_size": 4,
    "latent_log_var": "constant",
    "norm_layer": "pixel_norm",
    "spatial_padding_mode": "zeros",
}

LTX25_DECODER_CONFIG = {
    "_class_name": "NADiffusionDecoder",
    "in_channels": 128,
    "out_channels": 3,
    "patch_size": 4,
    "head_dim": 64,
    "stage_channels": [2048, 1024, 512, 512, 256],
    "stage_depths": [4, 6, 4, 2, 8],
    "stage_kernels": [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [11, 11, 11]],
    "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
    "spatial_padding_mode": "zeros",
    "resampler_kind": "linear",
    "stage5_kernel": [11, 11, 11],
    "timestep_scale_multiplier": 1000.0,
    "default_num_inference_steps": 1,
}


def _vae_config(encoder: dict, decoder: dict) -> dict:
    return {"_class_name": "CausalDiffusionVAE", "encoder": encoder, "decoder": decoder, "model_output_type": "x0"}


def _meta_ltx25_decoder() -> DiffusionVideoDecoder:
    with torch.device("meta"):
        decoder = DiffusionVideoDecoder(**decoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG)))
    return decoder.to(torch.bfloat16)


def test_decoder_state_dict_renames_splits_and_drops():
    qkv_weight = torch.arange(12.0).reshape(6, 2)
    qkv_bias = torch.arange(6.0)
    checkpoint = {
        "decoder.t_embedder.mlp.0.weight": torch.zeros(1),
        "decoder.t_embedder.mlp.2.bias": torch.zeros(1),
        "decoder.det_stages.0.0.attn.qkv.weight": qkv_weight,
        "decoder.diff_blocks.1.attn.qkv.bias": qkv_bias,
        "decoder.diff_blocks.1.attn.proj.weight": torch.ones(2, 2),
        "decoder.coarse_head.weight": torch.zeros(1),
        "decoder.type_emb": torch.zeros(1),
        "per_channel_statistics.mean-of-means": torch.zeros(1),
        "encoder.conv_in.conv.weight": torch.zeros(1),
    }

    state = decoder_state_dict(checkpoint)

    assert set(state) == {
        "t_embedder.timestep_embedder.linear_1.weight",
        "t_embedder.timestep_embedder.linear_2.bias",
        "det_stages.0.0.attn.qkv.to_q.weight",
        "det_stages.0.0.attn.qkv.to_k.weight",
        "det_stages.0.0.attn.qkv.to_v.weight",
        "diff_blocks.1.attn.qkv.to_q.bias",
        "diff_blocks.1.attn.qkv.to_k.bias",
        "diff_blocks.1.attn.qkv.to_v.bias",
        "diff_blocks.1.attn.proj.weight",
        "per_channel_statistics.mean-of-means",
    }
    assert torch.equal(state["det_stages.0.0.attn.qkv.to_k.weight"], qkv_weight[2:4])
    assert torch.equal(state["diff_blocks.1.attn.qkv.to_v.bias"], qkv_bias[4:])


def test_decoder_state_dict_rejects_gated_checkpoints():
    with pytest.raises(ValueError, match="gated"):
        decoder_state_dict({"decoder.diff_blocks.0.gate_msa": torch.zeros(1)})


def test_ltx25_decoder_keys_match_module():
    decoder = _meta_ltx25_decoder()
    expected = set(decoder.state_dict())
    assert "diff_blocks.7.attn.qkv.to_q.weight" in expected
    assert "t_embedder.timestep_embedder.linear_2.weight" in expected
    assert not any("stage4_upsample" in key for key in expected)


def test_encoder_state_dict_strips_prefix():
    state = encoder_state_dict(
        {
            "encoder.conv_in.conv.weight": torch.zeros(1),
            "per_channel_statistics.std-of-means": torch.ones(1),
            "decoder.conv_in.weight": torch.zeros(1),
        }
    )
    assert set(state) == {"conv_in.conv.weight", "per_channel_statistics.std-of-means"}


def test_unported_configs_are_rejected():
    with pytest.raises(ValueError, match="norm_layer"):
        encoder_kwargs(_vae_config({**TINY_ENCODER_CONFIG, "norm_layer": "group_norm"}, LTX25_DECODER_CONFIG))
    with pytest.raises(ValueError, match="model_output_type"):
        decoder_kwargs({**_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG), "model_output_type": "v"})
    with pytest.raises(ValueError, match="default_num_inference_steps"):
        decoder_kwargs(
            _vae_config(TINY_ENCODER_CONFIG, {**LTX25_DECODER_CONFIG, "default_num_inference_steps": 2})
        )


def test_tiny_encoder_output_shape_and_frame_crop():
    torch.manual_seed(0)
    encoder = VideoEncoder(**encoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG))).eval()
    with torch.inference_mode():
        latent = encoder(torch.rand(1, 3, 18, 64, 96) * 2 - 1)
    assert latent.shape == (1, 8, 3, 2, 3)


def test_load_video_encoder_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    source = VideoEncoder(**encoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG)))
    tensors = {f"encoder.{k}": v for k, v in source.state_dict().items() if not k.startswith("per_channel")}
    tensors["per_channel_statistics.std-of-means"] = torch.full((8,), 2.0)
    tensors["per_channel_statistics.mean-of-means"] = torch.full((8,), 0.5)
    path = tmp_path / "vae.safetensors"
    save_file(tensors, str(path), metadata={"config": json.dumps({"vae": _vae_config(TINY_ENCODER_CONFIG, {})})})

    encoder = load_video_encoder(path, torch.device("cpu"))

    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in encoder.parameters())
    assert torch.equal(encoder.conv_in.conv.weight, source.conv_in.conv.weight.bfloat16())
    assert torch.equal(encoder.per_channel_statistics.get_buffer("std-of-means"), torch.full((8,), 2.0).bfloat16())


def test_ltx25_tiling_needs_minimum_budget():
    decoder = _meta_ltx25_decoder()
    with pytest.raises(ValueError, match="Cannot fit"):
        decoder.recommended_tiling_config(height=512, width=512, num_frames=121, free_bytes=3 << 30)
    config = decoder.recommended_tiling_config(height=512, width=512, num_frames=121, free_bytes=4_200_000_000)
    assert config.frames == DimensionSizeConfig(tile_size=80, overlap=40)
    assert config.height.overlap == config.width.overlap == 160
    assert config.height.tile_size < 512 and config.width.tile_size < 512
    roomy = decoder.recommended_tiling_config(height=512, width=512, num_frames=121, free_bytes=24 << 30)
    assert (roomy.frames.tile_size, roomy.height.tile_size, roomy.width.tile_size) == (128, 512, 512)


def test_ltx25_tile_schedule_masks_partition_unity():
    decoder = _meta_ltx25_decoder()
    frames, height, width = 121, 512, 768
    config = TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=80, overlap=40),
        height=DimensionSizeConfig(tile_size=320, overlap=160),
        width=DimensionSizeConfig(tile_size=384, overlap=160),
    )
    strides = [tuple(u.stride) for u in decoder.upsamples]
    s4 = diffusion_tiling.stage4_thw_from_latent(strides, 16, 16, 24, drop_leading_frame=True)
    tiles = diffusion_tiling.prepare_tile_schedule(
        torch.Size([1, 128, *s4]),
        config,
        upsample3_stride=strides[3],
        patch_size=decoder.patch_size,
        min_tile_size=decoder.tile_min_sizes,
    )

    assert len({t.out_coords[2] for t in tiles}) > 1
    assert len({t.out_coords[3] for t in tiles}) > 1
    coverage = torch.zeros(1, 1, frames, height, width)
    for tile in tiles:
        region = torch.ones(1, 1, *(s.stop - s.start for s in tile.out_coords[2:]))
        coverage[tile.out_coords] += scale_by_masks_1d(region, tile.masks_1d)
    torch.testing.assert_close(coverage, torch.ones_like(coverage))


def _tiny_decoder() -> DiffusionVideoDecoder:
    torch.manual_seed(0)
    return DiffusionVideoDecoder(
        in_channels=8,
        out_channels=3,
        patch_size=4,
        head_dim=16,
        stage_channels=(128, 64, 32, 32, 16),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((3, 3, 3),) * 5,
        upsamples=(((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
        stage5_kernel=(3, 3, 3),
        stage5_channels=16,
        t_emb_dim=32,
        timestep_scale_multiplier=1000.0,
    ).eval()


def test_tiny_decoder_multi_tile_decode_is_seeded_and_shaped():
    decoder = _tiny_decoder()
    latent = torch.randn(1, 8, 3, 3, 4)
    config = TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=16, overlap=8),
        height=DimensionSizeConfig(tile_size=64, overlap=32),
        width=DimensionSizeConfig(tile_size=64, overlap=32),
    )

    def decode(seed: int) -> list[torch.Tensor]:
        with torch.inference_mode():
            return list(decoder.tiled_decode(latent, config, torch.Generator().manual_seed(seed)))

    chunks = decode(1)
    pixels = torch.cat(chunks, dim=2)
    assert len(chunks) > 1
    assert pixels.shape == (1, 3, 17, 96, 128)
    assert torch.isfinite(pixels).all()
    assert torch.equal(pixels, torch.cat(decode(1), dim=2))
    assert not torch.equal(pixels, torch.cat(decode(2), dim=2))
