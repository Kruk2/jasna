"""Build the encoder / decoder from a ``CausalDiffusionVAE`` checkpoint (its metadata config + weights)."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

import torch
from torch import nn

from jasna.ltx.model_files import open_tensors
from jasna.models.ltx_vae.diffusion_video_decoder import DiffusionVideoDecoder
from jasna.models.ltx_vae.tiling import TileSizeConfig
from jasna.models.ltx_vae.transformer import compile_diffusion_blocks, enable_fused_attention
from jasna.models.ltx_vae.video_encoder import VideoEncoder, compile_video_encoder
from jasna.tensor_utils import compiles_bf16

_VAE_CLASS_NAME = "CausalDiffusionVAE"
_DECODER_CLASS_NAME = "NADiffusionDecoder"
_GATE_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")
_QKV_LEAVES = ("weight", "bias")
_UNUSED_DECODER_KEYS = frozenset({"type_emb"})

_Model = TypeVar("_Model", bound=nn.Module)

logger = logging.getLogger(__name__)


def read_vae_config(reader) -> dict:
    config = json.loads(reader.metadata()["config"])["vae"]
    if config.get("_class_name") != _VAE_CLASS_NAME:
        raise ValueError(f"VAE class {config.get('_class_name')!r} is not supported")
    return config


def _require(config: dict, key: str, expected: object, section: str) -> None:
    if config.get(key) != expected:
        raise ValueError(f"VAE {section} config {key}={config.get(key)!r} is not supported (expected {expected!r})")


def encoder_kwargs(config: dict) -> dict:
    """``VideoEncoder`` kwargs from a ``CausalDiffusionVAE`` config; rejects unported variants."""
    encoder = config["encoder"]
    _require(encoder, "dims", 3, "encoder")
    _require(encoder, "norm_layer", "pixel_norm", "encoder")
    _require(encoder, "latent_log_var", "constant", "encoder")
    _require(encoder, "spatial_padding_mode", "zeros", "encoder")
    return {
        "in_channels": encoder["in_channels"],
        "out_channels": encoder["out_channels"],
        "encoder_blocks": encoder["blocks"],
        "patch_size": encoder["patch_size"],
    }


def decoder_kwargs(config: dict) -> dict:
    """``DiffusionVideoDecoder`` kwargs from a ``CausalDiffusionVAE`` config; rejects unported variants."""
    decoder = config["decoder"]
    _require(decoder, "_class_name", _DECODER_CLASS_NAME, "decoder")
    _require(config, "model_output_type", "x0", "top-level")
    _require(decoder, "default_num_inference_steps", 1, "decoder")
    _require(decoder, "resampler_kind", "linear", "decoder")
    _require(decoder, "spatial_padding_mode", "zeros", "decoder")
    stage_channels = tuple(decoder["stage_channels"])
    return {
        "in_channels": decoder["in_channels"],
        "out_channels": decoder["out_channels"],
        "patch_size": decoder["patch_size"],
        "head_dim": decoder["head_dim"],
        "stage_channels": stage_channels,
        "stage_depths": tuple(decoder["stage_depths"]),
        "stage_kernels": tuple(tuple(kernel) for kernel in decoder["stage_kernels"]),
        "upsamples": tuple((tuple(stride), reduction) for stride, reduction in decoder["upsamples"]),
        "stage5_kernel": tuple(decoder["stage5_kernel"]),
        "stage5_channels": decoder.get("stage5_channels", stage_channels[-1]),
        "t_emb_dim": decoder.get("t_emb_dim", 384),
        "timestep_scale_multiplier": decoder["timestep_scale_multiplier"],
    }


def _read_tensors(reader, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    tensors = {name: torch.empty(shape, dtype=dtype, device="cpu")
               for name, (dtype, shape) in reader.tensor_specs().items() if name.startswith(prefixes)}
    try:
        reader.read_tensors_into((tuple(tensors),), iter((tensors,)))
        return tensors
    except BaseException:
        _clear_tensors(tensors.values(), None)
        raise


def _clear_tensors(tensors: Iterable[torch.Tensor], retained: nn.Module | None) -> None:
    preserved = {(tensor.device, tensor.untyped_storage().data_ptr())
                 for tensor in retained.state_dict().values()} if retained is not None else set()
    devices = set()
    with torch.no_grad():
        for tensor in tensors:
            if tensor.is_meta or (tensor.device, tensor.untyped_storage().data_ptr()) in preserved:
                continue
            tensor.zero_()
            if tensor.device.type == "cuda":
                devices.add(tensor.device)
    for device in devices:
        torch.cuda.synchronize(device)


def encoder_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint ``encoder.*`` / ``per_channel_statistics.*`` tensors -> ``VideoEncoder`` keys."""
    return {
        key.removeprefix("encoder."): value
        for key, value in checkpoint.items()
        if key.startswith(("encoder.", "per_channel_statistics."))
    }


def decoder_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint ``decoder.*`` / ``per_channel_statistics.*`` tensors -> ``DiffusionVideoDecoder`` keys.

    Strips ``decoder.``, renames ``t_embedder.mlp.{0,2}`` to ``t_embedder.timestep_embedder.linear_{1,2}``,
    splits fused ``qkv`` into ``to_q`` / ``to_k`` / ``to_v`` and drops the ``coarse_*`` preview head
    and the unused ``type_emb``.
    """
    out: dict[str, torch.Tensor] = {}
    copies = []
    try:
        for key, value in checkpoint.items():
            if key.startswith("per_channel_statistics."):
                out[key] = value
                continue
            if not key.startswith("decoder."):
                continue
            key = key.removeprefix("decoder.")
            if key.startswith("coarse_") or key in _UNUSED_DECODER_KEYS:
                continue
            if key.endswith(_GATE_SUFFIXES):
                raise ValueError(f"gated decoder checkpoints are not supported (found {key!r})")
            key = key.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
            key = key.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
            leaf = key.rsplit(".", 1)[-1]
            if key.endswith(".qkv." + leaf) and leaf in _QKV_LEAVES:
                prefix = key[: -len(leaf)]
                for name, part in zip(("to_q", "to_k", "to_v"), value.chunk(3), strict=True):
                    copy = part.clone()
                    copies.append(copy)
                    out[f"{prefix}{name}.{leaf}"] = copy
                continue
            out[key] = value
        return out
    except BaseException:
        _clear_tensors(copies, None)
        raise


def _build_bf16(factory: Callable[[], _Model], state: dict[str, torch.Tensor], device: torch.device) -> _Model:
    model, result = None, None
    converted = {}
    try:
        with torch.device("meta"):
            model = factory()
        for name, tensor in state.items():
            converted[name] = tensor.to(torch.bfloat16)
        model.load_state_dict(converted, strict=True, assign=True)
        result = model.to(device).eval().requires_grad_(False)
        return result
    finally:
        tensors = [*state.values(), *converted.values()]
        if result is None and model is not None:
            tensors.extend(model.state_dict().values())
        _clear_tensors(tensors, result)


def load_video_encoder(vae_path: Path, device: torch.device) -> VideoEncoder:
    """bf16 eval ``VideoEncoder``; call it on ``(B, 3, 1+8k, H, W)`` pixels in [-1, 1]."""
    checkpoint, encoder = {}, None
    try:
        with open_tensors(vae_path) as reader:
            kwargs = encoder_kwargs(read_vae_config(reader))
            checkpoint = _read_tensors(reader, ("encoder.", "per_channel_statistics."))
        encoder = _build_bf16(lambda: VideoEncoder(**kwargs), encoder_state_dict(checkpoint), device)
        if compiles_bf16(device):
            contiguous_weights = list(encoder.state_dict().values())
            encoder.to(memory_format=torch.channels_last_3d)
            _clear_tensors(contiguous_weights, encoder)
            compile_video_encoder(device)
        return encoder
    finally:
        _clear_tensors(checkpoint.values(), encoder)


def load_video_decoder(vae_path: Path, tuned_decoder_path: Path, device: torch.device) -> DiffusionVideoDecoder:
    """bf16 eval ``DiffusionVideoDecoder`` with the fine-tuned decoder tensors overlaid."""
    decoder, complete = None, False
    checkpoint, tuned, overlay = {}, {}, {}
    try:
        with open_tensors(vae_path) as reader:
            kwargs = decoder_kwargs(read_vae_config(reader))
            checkpoint = _read_tensors(reader, ("decoder.", "per_channel_statistics."))
        state = decoder_state_dict(checkpoint)
        decoder = _build_bf16(lambda: DiffusionVideoDecoder(**kwargs), state, device)
        with open_tensors(tuned_decoder_path) as reader:
            tuned = _read_tensors(reader, ("",))
        own = decoder.state_dict()
        unknown = sorted(set(tuned) - set(own))
        if unknown:
            raise KeyError(f"{tuned_decoder_path}: {len(unknown)} tensors not in the decoder, e.g. {unknown[:3]}")
        for name, tensor in tuned.items():
            overlay[name] = tensor.to(own[name].dtype)
        decoder.load_state_dict(overlay, strict=False)
        enable_fused_attention(device)
        compile_diffusion_blocks(device)
        complete = True
        return decoder
    except BaseException:
        if decoder is not None:
            _clear_tensors(decoder.state_dict().values(), None)
        raise
    finally:
        _clear_tensors([*checkpoint.values(), *tuned.values(), *overlay.values()], decoder if complete else None)


_decoded_layouts: set[tuple[int, int, int, TileSizeConfig]] = set()
"""Tile shapes are static per layout, so only a layout's first decode can compile."""


@torch.inference_mode()
def decode_latent(
    decoder: DiffusionVideoDecoder,
    latent: torch.Tensor,
    free_bytes: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Tiled decode of a normalized ``(B, 128, T, h, w)`` latent to ``(B, 3, F, H, W)`` pixels in ~[-1, 1].

    The tile layout is chosen from ``free_bytes`` (activation budget); raises ``ValueError``
    when even the minimum tile does not fit.
    """
    _b, _c, frames, height, width = latent.shape
    scale = decoder.video_downscale_factors
    pixel_frames, pixel_height, pixel_width = (frames - 1) * scale.time + 1, height * scale.height, width * scale.width
    tiling = decoder.recommended_tiling_config(
        height=pixel_height, width=pixel_width, num_frames=pixel_frames, free_bytes=free_bytes
    )
    layout = (pixel_frames, pixel_height, pixel_width, tiling)
    first_of_layout = layout not in _decoded_layouts
    if first_of_layout:
        _log_tiling(*layout)
    video = torch.cat(list(decoder.tiled_decode(latent, tiling, generator)), dim=2)
    if first_of_layout:
        _decoded_layouts.add(layout)
        _free_compile_garbage()
    return video


def _free_compile_garbage() -> None:
    """A call that compiled a new tile shape stays referenced from torch.compile's frames in
    reference cycles, activations included (GiBs at 768 px); collect them so the next window's
    decode budget sees that memory free. A full collection costs ~90 ms with a job's heap."""
    gc.collect()
    torch.cuda.empty_cache()


def _log_tiling(frames: int, height: int, width: int, tiling: TileSizeConfig) -> None:
    """A window decoded in several tiles is noticeably slower."""
    logger.info(
        "LTX decode: %d frames x %dx%d px in tiles of %d frames x %dx%d px",
        frames,
        height,
        width,
        tiling.frames.tile_size,
        tiling.height.tile_size,
        tiling.width.tile_size,
    )
