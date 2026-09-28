"""Build the encoder / decoder from a ``CausalDiffusionVAE`` checkpoint (its metadata config + weights)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch
from torch import nn

from jasna.ltx.model_files import open_tensors
from jasna.models.ltx_vae.diffusion_video_decoder import DiffusionVideoDecoder
from jasna.models.ltx_vae.video_encoder import VideoEncoder

_VAE_CLASS_NAME = "CausalDiffusionVAE"
_DECODER_CLASS_NAME = "NADiffusionDecoder"
_GATE_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")
_QKV_LEAVES = ("weight", "bias")
_UNUSED_DECODER_KEYS = frozenset({"type_emb"})

_Model = TypeVar("_Model", bound=nn.Module)


def read_vae_config(vae_path: Path) -> dict:
    with open_tensors(vae_path) as handle:
        config = json.loads(handle.metadata()["config"])["vae"]
    if config.get("_class_name") != _VAE_CLASS_NAME:
        raise ValueError(f"{vae_path}: VAE class {config.get('_class_name')!r} is not supported")
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


def _read_tensors(path: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    with open_tensors(path) as handle:
        return {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefixes)}


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
                out[f"{prefix}{name}.{leaf}"] = part.clone()
            continue
        out[key] = value
    return out


def _build_bf16(factory: Callable[[], _Model], state: dict[str, torch.Tensor], device: torch.device) -> _Model:
    with torch.device("meta"):
        model = factory()
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=True, assign=True)
    return model.to(device).eval().requires_grad_(False)


def load_video_encoder(vae_path: Path, device: torch.device) -> VideoEncoder:
    """bf16 eval ``VideoEncoder``; call it on ``(B, 3, 1+8k, H, W)`` pixels in [-1, 1]."""
    kwargs = encoder_kwargs(read_vae_config(vae_path))
    state = encoder_state_dict(_read_tensors(vae_path, ("encoder.", "per_channel_statistics.")))
    return _build_bf16(lambda: VideoEncoder(**kwargs), state, device)


def load_video_decoder(vae_path: Path, tuned_decoder_path: Path, device: torch.device) -> DiffusionVideoDecoder:
    """bf16 eval ``DiffusionVideoDecoder`` with the fine-tuned decoder tensors overlaid."""
    kwargs = decoder_kwargs(read_vae_config(vae_path))
    state = decoder_state_dict(_read_tensors(vae_path, ("decoder.", "per_channel_statistics.")))
    decoder = _build_bf16(lambda: DiffusionVideoDecoder(**kwargs), state, device)

    tuned = _read_tensors(tuned_decoder_path, ("",))
    own = decoder.state_dict()
    unknown = sorted(set(tuned) - set(own))
    if unknown:
        raise KeyError(f"{tuned_decoder_path}: {len(unknown)} tensors not in the decoder, e.g. {unknown[:3]}")
    decoder.load_state_dict({k: v.to(own[k].dtype) for k, v in tuned.items()}, strict=False)
    return decoder


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
    tiling = decoder.recommended_tiling_config(
        height=height * scale.height,
        width=width * scale.width,
        num_frames=(frames - 1) * scale.time + 1,
        free_bytes=free_bytes,
    )
    return torch.cat(list(decoder.tiled_decode(latent, tiling, generator)), dim=2)
