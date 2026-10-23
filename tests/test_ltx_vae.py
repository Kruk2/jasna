import gc
import json
import weakref
from contextlib import contextmanager

import pytest
import torch
from safetensors.torch import save_file

from jasna.models.ltx_vae import diffusion_tiling, load_video_encoder, loader, video_encoder
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.diffusion_video_decoder import DiffusionVideoDecoder
from jasna.models.ltx_vae.eager_na import na3d
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


@pytest.mark.parametrize("failure", ("gated", "incomplete-qkv"))
def test_decoder_state_dict_failure_clears_discarded_qkv_clones(monkeypatch, failure):
    source = torch.arange(12.0).reshape(6, 2)
    checkpoint = {"decoder.first.qkv.weight": source}
    checkpoint["decoder.second.gate_msa" if failure == "gated" else "decoder.second.qkv.weight"] = torch.ones(1)
    clones = []
    original_clone = torch.Tensor.clone

    def clone(tensor, *args, **kwargs):
        result = original_clone(tensor, *args, **kwargs)
        clones.append(result)
        return result

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    with pytest.raises(ValueError):
        decoder_state_dict(checkpoint)
    assert clones
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in clones)
    assert torch.equal(source, torch.arange(12.0).reshape(6, 2))


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


def test_load_video_encoder_from_checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(0)
    source = VideoEncoder(**encoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG)))
    tensors = {f"encoder.{k}": v for k, v in source.state_dict().items() if not k.startswith("per_channel")}
    tensors["per_channel_statistics.std-of-means"] = torch.full((8,), 2.0)
    tensors["per_channel_statistics.mean-of-means"] = torch.full((8,), 0.5)
    path = tmp_path / "vae.safetensors"
    save_file(tensors, str(path), metadata={"config": json.dumps({"vae": _vae_config(TINY_ENCODER_CONFIG, {})})})

    opened = []
    open_tensors = loader.open_tensors

    @contextmanager
    def record_open(source):
        opened.append(source)
        with open_tensors(source) as reader:
            yield reader

    monkeypatch.setattr(loader, "open_tensors", record_open)
    monkeypatch.setattr(video_encoder, "_encode_cropped", video_encoder.encode_cropped)

    encoder = load_video_encoder(path, torch.device("cpu"))

    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in encoder.parameters())
    assert torch.equal(encoder.conv_in.conv.weight, source.conv_in.conv.weight.bfloat16())
    assert torch.equal(encoder.per_channel_statistics.get_buffer("std-of-means"), torch.full((8,), 2.0).bfloat16())
    assert encoder.conv_in.conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert video_encoder._encode_cropped is not video_encoder.encode_cropped
    assert opened == [path]


def test_load_video_encoder_before_sm80_compiles_fp16_with_a_float32_tail(tmp_path, monkeypatch):
    source = VideoEncoder(**encoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG)))
    path = tmp_path / "vae.safetensors"
    save_file({f"encoder.{k}" if not k.startswith("per_channel") else k: v.bfloat16()
               for k, v in source.state_dict().items()},
              str(path), metadata={"config": json.dumps({"vae": _vae_config(TINY_ENCODER_CONFIG, {})})})
    compiled = []
    monkeypatch.setattr(loader, "half_dtype", lambda device: torch.float16)
    monkeypatch.setattr(video_encoder, "_encode_cropped", video_encoder.encode_cropped)
    monkeypatch.setattr(video_encoder, "compile_static",
                        lambda fn, mode, device, dtype: compiled.append(dtype) or fn)

    encoder = load_video_encoder(path, torch.device("cpu"))

    assert encoder.conv_in.conv.weight.dtype == torch.float16
    assert torch.equal(encoder.conv_in.conv.weight, source.conv_in.conv.weight.bfloat16().half())
    assert encoder.conv_out.conv.weight.dtype == torch.float32
    assert encoder.conv_in.conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert compiled == [torch.float16]


def _amd_encoder_checkpoint(tmp_path):
    source = VideoEncoder(**encoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG)))
    path = tmp_path / "vae.safetensors"
    save_file({f"encoder.{k}" if not k.startswith("per_channel") else k: v.bfloat16()
               for k, v in source.state_dict().items()},
              str(path), metadata={"config": json.dumps({"vae": _vae_config(TINY_ENCODER_CONFIG, {})})})
    return path


@pytest.mark.parametrize("triton", (True, False))
def test_an_eager_amd_encoder_keeps_only_the_kitchen_head_channels_last(tmp_path, monkeypatch, triton):
    path = _amd_encoder_checkpoint(tmp_path)
    monkeypatch.setattr(loader, "is_amd_device", lambda device: True)
    monkeypatch.setattr(loader, "has_triton_package", lambda: triton)
    monkeypatch.setattr(video_encoder, "_kitchen_conv3d", False)
    monkeypatch.setattr(video_encoder, "_encode_cropped", video_encoder.encode_cropped)

    encoder = load_video_encoder(path, torch.device("cpu"))

    head = video_encoder.KITCHEN_DOWN_BLOCKS
    def channels_last(layer):
        return next(layer.parameters()).is_contiguous(memory_format=torch.channels_last_3d)
    assert [channels_last(layer) for layer in (encoder.conv_in, *encoder.down_blocks, encoder.conv_out)] == (
        [True] * (1 + head) + [triton] * (len(encoder.down_blocks) - head + 1))
    tail = encoder.down_blocks[head]
    sample = torch.randn(1, 3, 2, 4, 4, dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)
    converted = video_encoder._layer_input(sample, tail)
    tail_format = torch.channels_last_3d if triton else torch.contiguous_format
    assert converted.dtype == torch.bfloat16 and converted.is_contiguous(memory_format=tail_format)


def test_load_video_encoder_on_amd_runs_the_full_resolution_head_on_kitchen(tmp_path, monkeypatch):
    path = _amd_encoder_checkpoint(tmp_path)
    compiled = []
    monkeypatch.setattr(loader, "is_amd_device", lambda device: True)
    monkeypatch.setattr(video_encoder, "_kitchen_conv3d", False)
    monkeypatch.setattr(video_encoder, "_encode_cropped", video_encoder.encode_cropped)
    monkeypatch.setattr(video_encoder, "compile_static",
                        lambda fn, mode, device, dtype: compiled.append(dtype) or fn)

    encoder = load_video_encoder(path, torch.device("cpu"))

    head = video_encoder.KITCHEN_DOWN_BLOCKS
    assert [next(block.parameters()).dtype for block in encoder.down_blocks] == (
        [torch.float16] * head + [torch.bfloat16] * (len(encoder.down_blocks) - head))
    assert encoder.conv_in.conv.weight.dtype == torch.float16
    assert encoder.conv_out.conv.weight.dtype == torch.bfloat16
    assert video_encoder._kitchen_conv3d
    assert compiled == [torch.bfloat16]


def test_kitchen_conv3d_matches_the_causal_conv_and_fuses_the_residual(monkeypatch):
    torch.manual_seed(0)
    block = video_encoder.ResnetBlock3D(16).half()
    x = torch.randn(1, 16, 5, 6, 7).half()
    with torch.inference_mode():
        reference = block(x).float()
        monkeypatch.setattr(video_encoder, "_kitchen_conv3d", True)
        video_encoder.enable_kitchen_conv3d()
        out = block(x)

    assert out.dtype == torch.float16
    assert out.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(out.float(), reference, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("memory_format", (torch.contiguous_format, torch.channels_last_3d))
def test_causal_frames_repeat_the_first_frame_in_the_input_memory_format(memory_format):
    conv = video_encoder.CausalConv3d(4, 8)
    x = torch.randn(1, 4, 3, 5, 6).contiguous(memory_format=memory_format)

    padded = conv._causal_frames(x)

    assert torch.equal(padded, torch.cat((x[:, :, :1].repeat(1, 1, 2, 1, 1), x), dim=2))
    assert padded.is_contiguous(memory_format=memory_format)


def test_kitchen_conv3d_leaves_non_fp16_layers_on_the_torch_conv(monkeypatch):
    conv = video_encoder.CausalConv3d(4, 8).bfloat16()
    monkeypatch.setattr(video_encoder, "_kitchen_conv3d", True)
    monkeypatch.setattr(torch.ops.comfy_kitchen, "fp16_conv3d", None, raising=False)
    x = torch.randn(1, 4, 3, 5, 5).bfloat16()

    assert conv(x).dtype == torch.bfloat16


def test_fp16_with_float32_tail_splits_at_the_fp16_safe_blocks():
    config = {**TINY_ENCODER_CONFIG, "blocks": [["res_x", {"num_layers": 1}]] * (video_encoder.FP16_DOWN_BLOCKS + 2)}
    encoder = VideoEncoder(**encoder_kwargs(_vae_config(config, {}))).to_fp16_with_float32_tail()

    dtypes = [next(block.parameters()).dtype for block in encoder.down_blocks]

    assert dtypes == [torch.float16] * video_encoder.FP16_DOWN_BLOCKS + [torch.float32] * 2
    assert encoder.conv_in.conv.weight.dtype == torch.float16
    assert encoder.conv_out.conv.weight.dtype == torch.float32
    assert encoder.get_buffer("per_channel_statistics.std-of-means").dtype == torch.float32


def test_fp16_with_float32_tail_encodes_close_to_float32_and_returns_bf16():
    torch.manual_seed(0)
    encoder = VideoEncoder(**encoder_kwargs(_vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG))).eval()
    pixels = torch.rand(1, 3, 17, 64, 96) * 2 - 1
    with torch.inference_mode():
        reference = encoder(pixels)
        latent = encoder.to_fp16_with_float32_tail()(pixels.half())

    assert latent.dtype == torch.bfloat16
    torch.testing.assert_close(latent.float(), reference.float(), atol=3e-2, rtol=3e-2)


def test_pixel_norm_fp16_survives_squares_past_fp16_range():
    x = torch.tensor([300.0, -400.0, 0.5, 2e3]).view(1, 4, 1, 1, 1)

    normed = video_encoder.PixelNorm()(x.half())

    assert normed.dtype == torch.float16
    torch.testing.assert_close(normed.float(), video_encoder.PixelNorm()(x), atol=2e-3, rtol=2e-3)


def test_pixel_norm_fp16_keeps_eps_for_all_zero_locations():
    assert torch.equal(video_encoder.PixelNorm()(torch.zeros(1, 4, 1, 1, 1, dtype=torch.float16)),
                       torch.zeros(1, 4, 1, 1, 1, dtype=torch.float16))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_weight_state_cleanup_preserves_the_returned_cpu_model(dtype):
    source = torch.ones(2, 2, dtype=dtype)
    model = loader._build(lambda: torch.nn.Linear(2, 2, bias=False), {"weight": source}, torch.device("cpu"), torch.bfloat16)
    assert torch.equal(model.weight, torch.ones(2, 2, dtype=torch.bfloat16))
    if dtype == torch.bfloat16:
        assert source.untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
        assert torch.count_nonzero(source).item() == 4
    else:
        assert torch.count_nonzero(source).item() == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_weight_state_failure_clears_sources_and_partially_assigned_parameters(dtype):
    sources = {"weight": torch.ones(2, 2, dtype=dtype), "unexpected": torch.ones(1, dtype=dtype)}
    constructed = []

    def factory():
        model = torch.nn.Linear(2, 2, bias=False)
        constructed.append(model)
        return model

    with pytest.raises(RuntimeError, match="Unexpected key"):
        loader._build(factory, sources, torch.device("cpu"), torch.bfloat16)
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in sources.values())
    assert torch.count_nonzero(constructed[0].weight).item() == 0


@pytest.mark.parametrize("failure", ("stream", "unknown"))
def test_decoder_load_failure_clears_the_already_built_base_and_owned_sources(tmp_path, monkeypatch, failure):
    base = {"decoder.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    tuned = {"unexpected": torch.ones(1, dtype=torch.bfloat16)}
    base_path, tuned_path = tmp_path / "base", tmp_path / "tuned"
    save_file(base, str(base_path))
    save_file(tuned, str(tuned_path))
    constructed = []
    loaded = []
    read_tensors = loader._read_tensors

    def decoder(**kwargs):
        model = torch.nn.Linear(2, 2, bias=False)
        constructed.append(model)
        return model

    def tensors(reader, prefixes):
        if loaded and failure == "stream":
            raise ValueError("synthetic tuned final-check failure")
        result = read_tensors(reader, prefixes)
        loaded.append(result)
        return result

    monkeypatch.setattr(loader, "DiffusionVideoDecoder", decoder)
    monkeypatch.setattr(loader, "read_vae_config", lambda path: _vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG))
    monkeypatch.setattr(loader, "_read_tensors", tensors)
    exception, message = (ValueError, "final-check") if failure == "stream" else (KeyError, "not in the decoder")
    with pytest.raises(exception, match=message):
        loader.load_video_decoder(base_path, tuned_path, torch.device("cpu"))
    assert torch.count_nonzero(constructed[0].weight).item() == 0
    assert torch.count_nonzero(loaded[0]["decoder.weight"]).item() == 0
    if failure == "unknown":
        assert torch.count_nonzero(loaded[1]["unexpected"]).item() == 0


def test_decoder_load_success_preserves_cpu_weights_and_clears_overlay_sources(tmp_path, monkeypatch):
    base = {"decoder.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    tuned = {"weight": torch.full((2, 2), 2, dtype=torch.bfloat16)}
    base_path, tuned_path = tmp_path / "base", tmp_path / "tuned"
    save_file(base, str(base_path))
    save_file(tuned, str(tuned_path))
    loaded = []
    read_tensors = loader._read_tensors

    def tensors(reader, prefixes):
        result = read_tensors(reader, prefixes)
        loaded.append(result)
        return result
    monkeypatch.setattr(loader, "DiffusionVideoDecoder", lambda **kwargs: torch.nn.Linear(2, 2, bias=False))
    monkeypatch.setattr(loader, "read_vae_config", lambda path: _vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG))
    monkeypatch.setattr(loader, "_read_tensors", tensors)
    model = loader.load_video_decoder(base_path, tuned_path, torch.device("cpu"))
    assert torch.equal(model.weight, torch.full((2, 2), 2, dtype=torch.bfloat16))
    assert loaded[0]["decoder.weight"].untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
    assert torch.count_nonzero(loaded[1]["weight"]).item() == 0


def test_ltx25_tiling_needs_minimum_budget():
    decoder = _meta_ltx25_decoder()
    with pytest.raises(ValueError, match="Cannot fit"):
        decoder.recommended_tiling_config(height=512, width=512, num_frames=121, free_bytes=2_800_000_000)
    config = decoder.recommended_tiling_config(height=512, width=512, num_frames=121, free_bytes=4_200_000_000)
    assert config.frames == DimensionSizeConfig(tile_size=80, overlap=40)
    assert config.height.overlap == config.width.overlap == 160
    assert config.height.tile_size < 512 and config.width.tile_size < 512
    roomy = decoder.recommended_tiling_config(height=512, width=512, num_frames=121, free_bytes=24 << 30)
    assert (roomy.frames.tile_size, roomy.height.tile_size, roomy.width.tile_size) == (128, 512, 512)


def test_ltx25_tiling_fits_when_free_bytes_already_excludes_the_resident_decoder():
    decoder = _meta_ltx25_decoder()
    rtx_3070_ti_free_after_decoder_load = 3_465_125_888
    config = decoder.recommended_tiling_config(
        height=512, width=512, num_frames=121, free_bytes=rtx_3070_ti_free_after_decoder_load
    )
    assert (config.frames.tile_size, config.height.tile_size, config.width.tile_size) == (88, 352, 352)


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


def test_load_video_decoder_opens_base_once(tmp_path, monkeypatch):
    source = _tiny_decoder()
    config = {
        **LTX25_DECODER_CONFIG,
        "in_channels": 8,
        "head_dim": 16,
        "stage_channels": [128, 64, 32, 32, 16],
        "stage_depths": [1] * 5,
        "stage_kernels": [[3, 3, 3]] * 5,
        "stage5_kernel": [3, 3, 3],
        "stage5_channels": 16,
        "t_emb_dim": 32,
    }
    base, tuned = tmp_path / "base.safetensors", tmp_path / "tuned.safetensors"
    save_file({f"decoder.{name}": tensor for name, tensor in source.state_dict().items()}, str(base),
              metadata={"config": json.dumps({"vae": _vae_config(TINY_ENCODER_CONFIG, config)})})
    name, tensor = next(iter(source.state_dict().items()))
    overlay = torch.full_like(tensor, 2)
    save_file({name: overlay}, str(tuned))
    opened = []
    open_tensors = loader.open_tensors

    @contextmanager
    def record_open(path):
        opened.append(path)
        with open_tensors(path) as reader:
            yield reader

    monkeypatch.setattr(loader, "open_tensors", record_open)
    decoder = loader.load_video_decoder(base, tuned, torch.device("cpu"))
    assert torch.equal(decoder.state_dict()[name], overlay.bfloat16())
    assert opened == [base, tuned]


def test_load_video_decoder_builds_and_compiles_in_the_devices_half_dtype(tmp_path, monkeypatch):
    source = _tiny_decoder()
    config = {**LTX25_DECODER_CONFIG, "in_channels": 8, "head_dim": 16, "stage_channels": [128, 64, 32, 32, 16],
              "stage_depths": [1] * 5, "stage_kernels": [[3, 3, 3]] * 5, "stage5_kernel": [3, 3, 3],
              "stage5_channels": 16, "t_emb_dim": 32}
    base, tuned = tmp_path / "base.safetensors", tmp_path / "tuned.safetensors"
    save_file({f"decoder.{name}": tensor for name, tensor in source.state_dict().items()}, str(base),
              metadata={"config": json.dumps({"vae": _vae_config(TINY_ENCODER_CONFIG, config)})})
    name, tensor = next(iter(source.state_dict().items()))
    save_file({name: tensor.bfloat16()}, str(tuned))
    compiled = []
    monkeypatch.setattr(loader, "half_dtype", lambda device: torch.float16)
    monkeypatch.setattr(loader, "compile_diffusion_blocks", lambda device, dtype: compiled.append(dtype))

    decoder = loader.load_video_decoder(base, tuned, torch.device("cpu"))

    assert {p.dtype for p in decoder.parameters()} == {torch.float16}
    assert compiled == [torch.float16]


def test_decode_latent_computes_in_the_decoder_dtype_and_returns_the_latent_dtype(monkeypatch):
    decoder = _tiny_decoder()
    tiled_decode = decoder.tiled_decode
    seen = []
    monkeypatch.setattr(decoder, "tiled_decode",
                        lambda latent, tiling, generator: seen.append(latent.dtype) or tiled_decode(latent, tiling, generator))
    latent = torch.randn(1, 8, 3, 3, 4).bfloat16()

    video = loader.decode_latent(decoder, latent, 4 << 30, torch.Generator().manual_seed(1))
    reference = loader.decode_latent(decoder, latent.float(), 4 << 30, torch.Generator().manual_seed(1))

    assert seen == [torch.float32, torch.float32]
    assert video.dtype == torch.bfloat16
    assert torch.equal(video, reference.bfloat16())


@pytest.mark.parametrize("stage", ("encoder", "decoder"))
def test_vae_reader_close_failure_clears_loaded_weights(tmp_path, monkeypatch, stage):
    path = tmp_path / "vae.safetensors"
    config = _vae_config(TINY_ENCODER_CONFIG, LTX25_DECODER_CONFIG)
    save_file({f"{stage}.weight": torch.ones(2)}, str(path), metadata={"config": json.dumps({"vae": config})})
    retained = []
    open_tensors = loader.open_tensors

    @contextmanager
    def fail_on_close(source):
        with open_tensors(source) as reader:
            load = reader.read_tensors_into

            def record(group_names, groups):
                def capture():
                    for group in groups:
                        retained.extend(group.values())
                        yield group

                load(group_names, capture())

            reader.read_tensors_into = record
            yield reader
        raise ValueError("synthetic reader-close failure")

    monkeypatch.setattr(loader, "open_tensors", fail_on_close)
    with pytest.raises(ValueError, match="reader-close failure"):
        if stage == "encoder":
            loader.load_video_encoder(path, torch.device("cpu"))
        else:
            loader.load_video_decoder(path, tmp_path / "unused", torch.device("cpu"))
    assert len(retained) == 1
    assert torch.count_nonzero(retained[0]).item() == 0


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


def test_decode_latent_frees_activations_left_in_reference_cycles(monkeypatch):
    """torch.compile keeps a call that compiled a new tile shape in reference cycles; the next
    window's budget must not count those activations as used."""
    decoder = _tiny_decoder()
    tiled_decode = decoder.tiled_decode
    left_behind = []

    def tiled_decode_leaving_a_cycle(latent, tiling, generator):
        cycle = {"activation": torch.ones(64)}
        cycle["self"] = cycle
        left_behind.append(weakref.ref(cycle["activation"]))
        return tiled_decode(latent, tiling, generator)

    monkeypatch.setattr(decoder, "tiled_decode", tiled_decode_leaving_a_cycle)
    monkeypatch.setattr(loader, "_decoded_layouts", set())
    gc.collect()
    gc.disable()
    try:
        loader.decode_latent(decoder, torch.randn(1, 8, 3, 3, 4), 4 << 30, torch.Generator().manual_seed(1))
        assert left_behind[0]() is None
    finally:
        gc.enable()


def test_modulating_mlp_chunks_match_one_unchunked_pass(monkeypatch):
    monkeypatch.setattr(T, "SWIGLU_TILE_TOKENS", 7)
    torch.manual_seed(0)
    mlp = T.SwiGLU(16, 32)
    norm = torch.nn.RMSNorm(16)
    x = torch.randn(1, 2, 3, 4, 16)
    scale = torch.randn(1, 1, 1, 1, 16)
    shift = torch.randn(1, 1, 1, 1, 16)

    with torch.inference_mode():
        y = torch.nn.functional.rms_norm(x, (16,), norm.weight, 1e-6) * (1 + scale) + shift
        expected = x + mlp.w_down(torch.nn.functional.silu(mlp.w_gate(y)) * mlp.w_up(y))
        actual = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)

    torch.testing.assert_close(actual, expected)


def test_decoder_blocks_compile_once_with_static_shapes(monkeypatch):
    monkeypatch.setattr(T, "_slab_attention", T._attn_on_w_slab)
    monkeypatch.setattr(T, "_mlp_chunk", T.modulated_swiglu)
    monkeypatch.setattr(T, "_add_context", T.add_folded_context)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append((fn, kwargs)) or (lambda *a: fn(*a)))

    T.compile_diffusion_blocks(torch.device("cpu"), torch.bfloat16)
    T.compile_diffusion_blocks(torch.device("cpu"), torch.bfloat16)
    assert [fn for fn, _ in calls] == [T._attn_on_w_slab, T.modulated_swiglu, T.add_folded_context]
    assert all(kwargs == {"dynamic": False, "mode": T.DECODER_COMPILE_MODE} for _, kwargs in calls)


def test_slab_attention_of_core_columns_matches_full_slab_attention():
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(32, (3, 5, 5), head_dim=16)
    norm = torch.nn.RMSNorm(32)
    x = torch.randn(1, 4, 6, 12, 32)
    scale, shift = torch.randn(2, 1, 1, 1, 1, 32) * 0.1
    w_pos = torch.arange(12, dtype=torch.float32) + 7
    halo, core_len = 2, 7

    with torch.inference_mode():
        y = torch.nn.functional.rms_norm(x, (32,), norm.weight, 1e-6) * (1 + scale) + shift
        q, k, v = attn.project_qkv(y)
        q = T.apply_abs_rope_slab(attn.q_norm(q) * attn.scale, attn, w_pos)
        k = T.apply_abs_rope_slab(attn.k_norm(k), attn, w_pos)
        full = na3d(q, k, v, attn.kernel_size, 0).reshape(1, 4, 6, 12, 32)
        expected = attn.proj(full)[:, :, :, halo : halo + core_len]
        columns = torch.arange(halo, 12)
        actual = T._attn_on_w_slab(attn, norm, x[:, :, :, :halo], x, columns, w_pos, scale, shift, core_len)

    torch.testing.assert_close(actual, expected)


def test_decode_latent_logs_and_collects_compile_garbage_only_on_a_layouts_first_decode(monkeypatch, caplog):
    """A full gc.collect costs ~90 ms per window with a job's heap; only a new layout compiles."""
    decoder = _tiny_decoder()
    collections = []
    monkeypatch.setattr(loader, "_decoded_layouts", set())
    monkeypatch.setattr(loader, "_free_compile_garbage", lambda: collections.append(True))

    with caplog.at_level("INFO", logger=loader.__name__):
        for _ in range(3):
            loader.decode_latent(decoder, torch.randn(1, 8, 3, 3, 4), 4 << 30, torch.Generator().manual_seed(1))
        assert len(collections) == 1
        loader.decode_latent(decoder, torch.randn(1, 8, 3, 4, 4), 4 << 30, torch.Generator().manual_seed(1))

    assert len(collections) == 2
    assert [r.getMessage() for r in caplog.records] == [
        "LTX decode: 17 frames x 96x128 px in tiles of 24 frames x 96x128 px",
        "LTX decode: 17 frames x 128x128 px in tiles of 24 frames x 128x128 px",
    ]


@pytest.mark.parametrize("drop_leading_frame", (True, False))
def test_folded_context_matches_upsample_then_context_proj(drop_leading_frame):
    torch.manual_seed(0)
    upsample = T.LinearPixelShuffleUpsample(8, (2, 2, 2), 2)
    context_proj = torch.nn.Linear(4, 16)
    stage4 = torch.randn(1, 3, 4, 9, 8)
    frames = 5 if drop_leading_frame else 6
    x = torch.randn(1, frames, 8, 17, 16)

    with torch.inference_mode():
        context = context_proj(upsample(stage4, drop_leading_frame=drop_leading_frame))
        expected = x + context[:, :, :, :17]
        actual = T.inject_deferred_context(
            x.clone(), stage4, upsample, context_proj, drop_leading_frame=drop_leading_frame
        )

    torch.testing.assert_close(actual, expected)


def test_w_chunked_attention_equals_one_slab_with_edge_replicated_halos():
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(32, (3, 5, 5), head_dim=16).double()
    norm = torch.nn.RMSNorm(32).double()
    x = torch.randn(1, 4, 6, 16, 32, dtype=torch.float64)
    scale, shift = torch.randn(2, 1, 1, 1, 1, 32, dtype=torch.float64) * 0.1
    halo = 2

    with torch.inference_mode():
        left, right = x[:, :, :, :1], x[:, :, :, -1:]
        padded = torch.cat([left.expand(-1, -1, -1, halo, -1), x, right.expand(-1, -1, -1, halo, -1)], dim=3)
        w_pos = torch.arange(16 + 2 * halo, dtype=torch.float32) - halo
        columns = torch.arange(halo, 16 + 2 * halo)
        expected = x + T._attn_on_w_slab(attn, norm, padded[:, :, :, :halo], padded, columns, w_pos, scale, shift, 16)
        actual = T.w_chunked_attention_residual(x.clone(), attn, norm, scale, shift)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("compiling", (True, False))
def test_abs_rope_rotates_each_pair_by_its_axis_position(compiling, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    attn = T.NeighborhoodAttention3D(64, (3, 3, 3), head_dim=32)
    x = torch.randn(1, 5, 3, 4, 2, 32)
    w_pos = torch.arange(4, dtype=torch.float32) + 9

    d_t, d_h, d_w = attn.rope_dim_split
    positions = torch.stack(torch.meshgrid(torch.arange(5.0), torch.arange(3.0), w_pos, indexing="ij"), -1)
    axis = torch.tensor([0] * (d_t // 2) + [1] * (d_h // 2) + [2] * (d_w // 2))
    angles = positions[..., axis] * torch.cat([attn.rope_inv_t, attn.rope_inv_h, attn.rope_inv_w])
    pairs = torch.view_as_complex(x.reshape(1, 5, 3, 4, 2, 16, 2).contiguous())
    expected = torch.view_as_real(pairs * torch.polar(torch.ones_like(angles), angles)[:, :, :, None]).reshape(x.shape)

    torch.testing.assert_close(T.apply_abs_rope_slab(x, attn, w_pos), expected)
