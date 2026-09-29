"""The LTX restoration transformer and its fused-chain sampler.

A track's windows are denoised in lockstep: every step runs all windows through each
block before the next block is fetched, so a streamed block crosses PCIe once per step
per window group. After each step the windows' shared latent frames are replaced by a
ramp-weighted mean (MultiDiffusion-style overlap fusion). Sampler: Euler over a fixed
sigma schedule, CFG 1, STG on one block whose perturbed pass reuses the conditional
pass's hidden state up to that block.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from jasna.ltx import transformer as T
from jasna.ltx.blocks import BlockStore
from jasna.ltx.model_files import open_tensors
from jasna.ltx.plan import FUSION_RAMP, LEFT_SHARED_LATENTS, RIGHT_SHARED_LATENTS

logger = logging.getLogger(__name__)

LOWTAIL15_SIGMAS = (
    1.0, 0.982, 0.959, 0.93, 0.892, 0.839, 0.762, 0.637, 0.4,
    0.32, 0.184, 0.106, 0.061, 0.035, 0.02, 0.0,
)  # fmt: skip
STG_SCALE = 1.0
STG_BLOCK = 28
ACTIVATION_RESERVE_BYTES = 3 << 30
_ADALN_PREFIXES = ("adaln_single.", "prompt_adaln_single.")


def patchify(latent: torch.Tensor) -> torch.Tensor:
    """``[1, C, F, H, W]`` -> ``[1, F*H*W, C]``."""
    return latent.flatten(2).transpose(1, 2)


def unpatchify(tokens: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    return tokens.transpose(1, 2).reshape(1, -1, frames, height, width)


def fuse_overlaps(latents: Sequence[torch.Tensor], tokens_per_frame: int, right_weights: Sequence[float]) -> None:
    """Write the weighted mean of each consecutive pair's shared latent frames into both,
    in place. All-zero weights copy the earlier window into the later one."""
    for left, right in zip(latents, latents[1:]):
        a = left[:, LEFT_SHARED_LATENTS.start * tokens_per_frame : LEFT_SHARED_LATENTS.stop * tokens_per_frame]
        b = right[:, RIGHT_SHARED_LATENTS.start * tokens_per_frame : RIGHT_SHARED_LATENTS.stop * tokens_per_frame]
        weight = torch.tensor(right_weights, dtype=torch.float32, device=a.device)
        weight = weight.repeat_interleave(tokens_per_frame).view(1, -1, 1)
        fused = (a.float() * (1.0 - weight) + b.float() * weight).to(a.dtype)
        a.copy_(fused)
        b.copy_(fused)


def _window_state_bytes(tokens: int) -> int:
    """Hidden states one window keeps between blocks (conditional + perturbed)."""
    return 2 * 2 * tokens * 4096 * 2


def _window_workspace_bytes(tokens: int) -> int:
    """Transient activations of one window inside one block."""
    return 2 * tokens * 4096 * 2 * 10 + T.CHUNK_ROWS * 16384 * 4 * 3


class LtxTransformer:
    """The restoration transformer loaded from a jasna LTX model file."""

    def __init__(self, path: Path, device: torch.device, *, sigmas: Sequence[float] = LOWTAIL15_SIGMAS) -> None:
        self.device = device
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        with open_tensors(path) as handle:
            keys = list(handle.keys())
            top_keys = [k for k in keys if not k.startswith("blocks.") and k != "prompt_context"]
            top = {k: handle.get_tensor(k).to(device) for k in top_keys}
            self.context = handle.get_tensor("prompt_context").to(device)
            self.conditions = [T.step_conditioning(top, float(s), device) for s in self.sigmas[:-1]]
            self.top = {k: v for k, v in top.items() if not k.startswith(_ADALN_PREFIXES)}
            del top
            blocks = [
                {k.split(".", 2)[2]: handle.get_tensor(k) for k in keys if k.startswith(f"blocks.{i}.")}
                for i in range(T.NUM_BLOCKS)
            ]
        self.blocks = BlockStore(blocks, device, resident=self._resident_blocks(blocks))
        self._ropes: dict[tuple[int, int, int], T.Rope] = {}
        logger.info("LTX self-attention: %s", "SageAttention 2" if T.enable_sage_attention() else "SDPA")
        self._block_forward = compiled_block_forward()

    def _resident_blocks(self, blocks: list[dict[str, torch.Tensor]]) -> int:
        sizes = [sum(t.numel() * t.element_size() for t in b.values()) for b in blocks]
        free, _ = torch.cuda.mem_get_info(self.device)
        budget = free - ACTIVATION_RESERVE_BYTES - 2 * max(sizes)
        resident, used = 0, 0
        for size in sizes:
            if used + size > budget:
                break
            used += size
            resident += 1
        return resident

    def close(self) -> None:
        self.blocks.close()
        self.top.clear()

    def _rope(self, frames: int, height: int, width: int) -> T.Rope:
        key = (frames, height, width)
        if key not in self._ropes:
            dim = self.top["patchify_proj.weight"].shape[0]
            self._ropes[key] = T.build_rope(frames, height, width, dim=dim, heads=T.HEADS, device=self.device)
        return self._ropes[key]

    def _group_size(self, windows: int, tokens: int) -> int:
        free, _ = torch.cuda.mem_get_info(self.device)
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        fits = (free - _window_workspace_bytes(tokens)) // _window_state_bytes(tokens)
        return max(1, min(windows, int(fits)))

    @torch.inference_mode()
    def denoise_chain(self, references: Sequence[torch.Tensor], seeds: Sequence[int]) -> list[torch.Tensor]:
        """Final latents ``[1, 128, F, H, W]`` of one track's windows, given each window's
        reference latent (the encoded mosaic crop) and noise seed."""
        frames, height, width = references[0].shape[2:]
        tokens_per_frame = height * width
        tokens = frames * tokens_per_frame
        rope = self._rope(frames, height, width)
        refs = [patchify(r.to(self.device, torch.bfloat16)) for r in references]
        latents = []
        for ref, seed in zip(refs, seeds):
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            noise = torch.randn(1, 2 * tokens, ref.shape[-1], device=self.device, dtype=torch.bfloat16, generator=generator)
            latents.append(noise[:, :tokens].clone())
        fuse_overlaps(latents, tokens_per_frame, [0.0] * len(FUSION_RAMP))
        group = self._group_size(len(latents), 2 * tokens)
        sigmas = self.sigmas.to(self.device)
        for step, cond in enumerate(self.conditions):
            for start in range(0, len(latents), group):
                members = range(start, min(start + group, len(latents)))
                denoised = self._denoise(
                    [latents[i] for i in members], [refs[i] for i in members], cond, rope, tokens_per_frame, sigmas[step]
                )
                for i, value in zip(members, denoised):
                    latents[i] = _euler(latents[i], value, sigmas, step)
            fuse_overlaps(latents, tokens_per_frame, FUSION_RAMP)
        return [unpatchify(latent, frames, height, width) for latent in latents]

    def _denoise(
        self,
        latents: list[torch.Tensor],
        refs: list[torch.Tensor],
        cond: T.StepConditioning,
        rope: T.Rope,
        tokens_per_frame: int,
        sigma: torch.Tensor,
    ) -> list[torch.Tensor]:
        xs = [T.embed_tokens(self.top, torch.cat([lat, ref], dim=1), tokens_per_frame) for lat, ref in zip(latents, refs)]
        perturbed: list[torch.Tensor | None] = [None] * len(xs)
        for index in range(T.NUM_BLOCKS):
            weights = self.blocks.acquire(index)
            for j in range(len(xs)):
                if STG_SCALE and index >= STG_BLOCK:
                    source = xs[j] if index == STG_BLOCK else perturbed[j]
                    perturbed[j] = self._block_forward(
                        weights, source, cond, self.context, rope, skip_self_attention=index == STG_BLOCK
                    )
                xs[j] = self._block_forward(weights, xs[j], cond, self.context, rope)
            self.blocks.release(index)
        out = []
        for lat, x, xp in zip(latents, xs, perturbed):
            denoised = (lat.float() - T.velocity(self.top, x, cond).float() * sigma).to(lat.dtype)
            if STG_SCALE:
                weak = (lat.float() - T.velocity(self.top, xp, cond).float() * sigma).to(lat.dtype)
                denoised = (denoised.float() + STG_SCALE * (denoised.float() - weak.float())).to(lat.dtype)
            out.append(denoised)
        return out


def compiled_block_forward():
    """``transformer.block_forward`` compiled with static shapes when Triton is available
    (one graph per canvas and per STG variant, shared by all blocks), else eager."""
    if importlib.util.find_spec("triton") is None:
        return T.block_forward
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
    return torch.compile(T.block_forward, dynamic=False)


def _euler(sample: torch.Tensor, denoised: torch.Tensor, sigmas: torch.Tensor, step: int) -> torch.Tensor:
    sigma = sigmas[step]
    velocity = ((sample.float() - denoised.float()) / sigma.item()).to(sample.dtype)
    return (sample.float() + velocity.float() * (sigmas[step + 1] - sigma)).to(sample.dtype)
