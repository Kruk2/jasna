"""The LTX restoration transformer and its fused-chain sampler.

A track's windows are denoised in lockstep: every step runs all windows through each
block before the next block is fetched, so a streamed block crosses PCIe once per step
per window group. After each step the windows' shared latent frames are replaced by a
ramp-weighted mean (MultiDiffusion-style overlap fusion). Sampler: Euler over the model
file's sigma schedule, CFG 1, and the file's STG scale on one block, whose perturbed pass
reuses the conditional pass's hidden state up to that block (skipped at scale 0).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from jasna._frozen import is_frozen
from jasna.ltx import native
from jasna.ltx import transformer as T
from jasna.ltx.blocks import BlockStore, OpaqueBlockStore
from jasna.ltx.model_files import PlaceholderHandle, open_tensors
from jasna.ltx.plan import FUSION_RAMP, LEFT_SHARED_LATENTS, RIGHT_SHARED_LATENTS
from jasna.tensor_utils import compile_static

logger = logging.getLogger(__name__)

STG_BLOCK = 28
ACTIVATION_RESERVE_BYTES = 3 << 30
PINNED_CHUNK_BYTES = 64 << 20
_ADALN_PREFIXES = ("adaln_single.", "prompt_adaln_single.")


def patchify(latent: torch.Tensor) -> torch.Tensor:
    """``[1, C, F, H, W]`` -> ``[1, F*H*W, C]``."""
    return latent.flatten(2).transpose(1, 2)


def unpatchify(tokens: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    return tokens.transpose(1, 2).reshape(1, -1, frames, height, width)


def fuse_overlaps(latents: Sequence[torch.Tensor], tokens_per_frame: int, right_weights: Sequence[float]) -> None:
    """Write the weighted mean of each consecutive pair's shared latent frames into both,
    in place. All-zero weights copy the earlier window into the later one."""
    weight = torch.tensor(right_weights, dtype=torch.float32).repeat_interleave(tokens_per_frame).view(1, -1, 1)
    weight = weight.to(latents[0].device, non_blocking=True)
    for left, right in zip(latents, latents[1:]):
        a = left[:, LEFT_SHARED_LATENTS.start * tokens_per_frame : LEFT_SHARED_LATENTS.stop * tokens_per_frame]
        b = right[:, RIGHT_SHARED_LATENTS.start * tokens_per_frame : RIGHT_SHARED_LATENTS.stop * tokens_per_frame]
        fused = (a.float() * (1.0 - weight) + b.float() * weight).to(a.dtype)
        a.copy_(fused)
        b.copy_(fused)


def sampler_settings(metadata: dict[str, str]) -> tuple[torch.Tensor, float]:
    """The model file's sigma schedule and STG scale (0 for a distilled student)."""
    return torch.tensor(json.loads(metadata["sigmas"]), dtype=torch.float32), float(json.loads(metadata["stg_scale"]))


def _window_state_bytes(tokens: int, *, stg: bool) -> int:
    """Hidden states one window keeps between blocks (conditional, plus perturbed with STG)."""
    return (2 if stg else 1) * 2 * tokens * 4096 * 2


def _window_workspace_bytes(tokens: int) -> int:
    """Transient activations of one window inside one block."""
    return 2 * tokens * 4096 * 2 * 10 + T.CHUNK_ROWS * 16384 * 4 * 3


def _read_tensors(reader, names: tuple[str, ...], specs, device: torch.device) -> dict[str, torch.Tensor]:
    sources = {name: torch.empty(specs[name][1], dtype=specs[name][0], device="cpu") for name in names}
    result = {}
    try:
        reader.read_tensors_into((names,), iter((sources,)))
        for name, source in sources.items():
            result[name] = source.to(device)
        return result
    except BaseException:
        for tensor in result.values():
            tensor.view(-1).view(torch.uint8).zero_()
        raise
    finally:
        for name, source in sources.items():
            if result.get(name) is not source:
                source.view(-1).view(torch.uint8).zero_()


def _opaque_blocks(reader) -> bool:
    """Whether the blocks go to the native opaque store: licensed files in frozen releases.
    Trial placeholders are plain random tensors and always use ``BlockStore``."""
    return is_frozen() and not isinstance(reader, PlaceholderHandle)


def _block_functions(
    device: torch.device, block_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]]
) -> tuple[Callable, Callable, str]:
    """(block forward, prompt key/value, backend name) for plain-tensor blocks: the native block
    when it can run here, with eager prompt keys/values (once per block and step), else both
    compiled."""
    native_forward = native.load(device, block_specs)
    if native_forward is not None:
        return native_forward, T.prompt_key_value, "native"
    T.enable_int8_attention()
    return (
        compiled_block_forward(device),
        compile_static(T.prompt_key_value, mode="max-autotune-no-cudagraphs", device=device),
        "torch.compile",
    )


class LtxTransformer:
    """The restoration transformer loaded from a jasna LTX model file; ``free_vram(device)``
    gives the VRAM free on the whole GPU, which sizes the resident blocks and window groups."""

    def __init__(self, path: Path, device: torch.device, free_vram: Callable[[torch.device], int]) -> None:
        self.device = device
        self._free_vram = free_vram
        self.top: dict[str, torch.Tensor] = {}
        self.conditions: list[T.StepConditioning] = []
        self.context: torch.Tensor | None = None
        self.blocks: BlockStore | OpaqueBlockStore | None = None
        self.block_backend = "native opaque store"
        self._ropes: dict[tuple[int, int, int], T.Rope] = {}
        try:
            T.require_kernels(device)
            with open_tensors(path) as handle:
                self.sigmas, self.stg_scale = sampler_settings(handle.metadata())
                specs = handle.tensor_specs()
                self.top = _read_tensors(handle, tuple(name for name in specs if not name.startswith("blocks.")),
                                         specs, device)
                self.context = self.top.pop("prompt_context")
                for sigma in self.sigmas[:-1]:
                    self.conditions.append(T.step_conditioning(self.top, float(sigma), device))
                for name in list(self.top):
                    if name.startswith(_ADALN_PREFIXES):
                        self.top.pop(name).view(-1).view(torch.uint8).zero_()
                names = [{k.split(".", 2)[2]: k for k in specs if k.startswith(f"blocks.{index}.")}
                         for index in range(T.NUM_BLOCKS)]
                sizes = [sum(math.prod(specs[name][1]) * specs[name][0].itemsize for name in block.values())
                         for block in names]
                resident = self._resident_blocks(sizes)
                block_specs = {name: specs[source] for name, source in names[0].items()}
                if _opaque_blocks(handle):
                    self.blocks = OpaqueBlockStore.from_reader(handle, names, device, resident=resident)
                else:
                    self.blocks = BlockStore.from_reader(handle, names, device, resident=resident)
            logger.info("LTX quantized linears: %s", "Triton kernels" if T.enable_quant_kernels(device) else "torch")
            if isinstance(self.blocks, OpaqueBlockStore):
                T.enable_int8_attention()
            else:
                self._block_forward, self._prompt_key_value, self.block_backend = _block_functions(device, block_specs)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    def _resident_blocks(self, sizes: list[int]) -> int:
        budget = self._free_vram(self.device) - ACTIVATION_RESERVE_BYTES - 2 * max(sizes)
        resident, used = 0, 0
        for size in sizes:
            if used + size > budget:
                break
            used += size
            resident += 1
        return resident

    def native_vram_bytes(self) -> int:
        return self.blocks.native_vram_bytes() if self.blocks is not None else 0

    def close(self) -> None:
        if self.blocks is not None:
            self.blocks.close()
            self.blocks = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tensors = list(self.top.values())
        for condition in self.conditions:
            tensors.extend((condition.block_rows, condition.output_rows, condition.prompt_rows))
        if self.context is not None:
            tensors.append(self.context)
        for tensor in tensors:
            tensor.view(-1).view(torch.uint8).zero_()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.top.clear()
        self.conditions.clear()
        self.context = None
        self._ropes.clear()

    def _rope(self, frames: int, height: int, width: int) -> T.Rope:
        key = (frames, height, width)
        if key not in self._ropes:
            dim = self.top["patchify_proj.weight"].shape[0]
            self._ropes[key] = T.build_rope(frames, height, width, dim=dim, heads=T.HEADS, device=self.device)
        return self._ropes[key]

    def _group_size(self, windows: int, tokens: int) -> int:
        free = self._free_vram(self.device)
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        fits = (free - _window_workspace_bytes(tokens)) // _window_state_bytes(tokens, stg=bool(self.stg_scale))
        return max(1, min(windows, int(fits)))

    @torch.inference_mode()
    def denoise_chain(
        self, references: Sequence[torch.Tensor], seeds: Sequence[int], advance: Callable[[int], None]
    ) -> list[torch.Tensor]:
        """Final latents ``[1, 128, F, H, W]`` (on the CPU) of one track's windows, given each window's
        reference latent (the encoded mosaic crop) and noise seed. ``advance(n)`` reports n
        window-steps done."""
        frames, height, width = references[0].shape[2:]
        tokens_per_frame = height * width
        rope = self._rope(frames, height, width)

        def denoise(inputs: list[torch.Tensor], step: int) -> list[torch.Tensor]:
            return self._denoise(inputs, self.conditions[step], rope, tokens_per_frame, self.sigmas[step])

        return denoise_windows(
            references,
            seeds,
            self.sigmas,
            steps=len(self.conditions),
            group=self._group_size(len(references), 2 * frames * tokens_per_frame),
            denoise=denoise,
            advance=advance,
            device=self.device,
        )

    def _denoise(
        self,
        inputs: list[torch.Tensor],
        cond: T.StepConditioning,
        rope: T.Rope,
        tokens_per_frame: int,
        sigma: torch.Tensor,
    ) -> list[torch.Tensor]:
        latents = [x[:, : x.shape[1] // 2] for x in inputs]
        xs = [T.embed_tokens(self.top, x, tokens_per_frame) for x in inputs]
        perturbed: list[torch.Tensor | None] = [None] * len(xs)
        if isinstance(self.blocks, OpaqueBlockStore):
            self.blocks.begin_step(xs[0].shape[1], len(xs), bool(self.stg_scale))
            for index in range(T.NUM_BLOCKS):
                self.blocks.activate(index)
                for j in range(len(xs)):
                    if self.stg_scale and index >= STG_BLOCK:
                        source = xs[j] if index == STG_BLOCK else perturbed[j]
                        perturbed[j] = self.blocks.block_forward(
                            source, cond, self.context, rope, stg_branch=True,
                        )
                    xs[j] = self.blocks.block_forward(xs[j], cond, self.context, rope)
        else:
            for index in range(T.NUM_BLOCKS):
                weights = self.blocks.acquire(index)
                prompt = self._prompt_key_value(weights, cond, self.context)
                for j in range(len(xs)):
                    if self.stg_scale and index >= STG_BLOCK:
                        source = xs[j] if index == STG_BLOCK else perturbed[j]
                        perturbed[j] = self._block_forward(
                            weights, source, cond, prompt, rope, skip_self_attention=index == STG_BLOCK
                        )
                    xs[j] = self._block_forward(weights, xs[j], cond, prompt, rope)
                self.blocks.release(index)
        out = []
        for lat, x, xp in zip(latents, xs, perturbed):
            denoised = (lat.float() - T.velocity(self.top, x, cond).float() * sigma).to(lat.dtype)
            if self.stg_scale:
                weak = (lat.float() - T.velocity(self.top, xp, cond).float() * sigma).to(lat.dtype)
                denoised = (denoised.float() + self.stg_scale * (denoised.float() - weak.float())).to(lat.dtype)
            out.append(denoised)
        return out


def _pinned_windows(count: int, shape: tuple[int, ...]) -> list[torch.Tensor]:
    """``count`` bf16 host tensors of ``shape`` in pinned memory, packed into chunks of
    ``PINNED_CHUNK_BYTES`` (the pinned allocator rounds every allocation up to a power of two)."""
    per_chunk = max(1, PINNED_CHUNK_BYTES // (math.prod(shape) * torch.bfloat16.itemsize))
    windows: list[torch.Tensor] = []
    for start in range(0, count, per_chunk):
        chunk = torch.empty((min(per_chunk, count - start), *shape), dtype=torch.bfloat16, pin_memory=True)
        windows.extend(chunk.unbind(0))
    return windows


class _FusedChain:
    """Windows' latents pushed in order: each is fused on the GPU with the one before it,
    which then goes back to its host window."""

    def __init__(self, host: list[torch.Tensor], tokens: int, tokens_per_frame: int) -> None:
        self._host = host
        self._tokens = tokens
        self._tokens_per_frame = tokens_per_frame
        self._held: tuple[int, torch.Tensor] | None = None

    def push(self, index: int, latent: torch.Tensor, right_weights: Sequence[float]) -> None:
        if self._held is not None:
            fuse_overlaps([self._held[1], latent], self._tokens_per_frame, right_weights)
            self._store(*self._held)
        self._held = (index, latent)

    def flush(self) -> None:
        self._store(*self._held)
        self._held = None

    def _store(self, index: int, latent: torch.Tensor) -> None:
        self._host[index][:, : self._tokens].copy_(latent, non_blocking=True)


def denoise_windows(
    references: Sequence[torch.Tensor],
    seeds: Sequence[int],
    sigmas: torch.Tensor,
    *,
    steps: int,
    group: int,
    denoise: Callable[[list[torch.Tensor], int], list[torch.Tensor]],
    advance: Callable[[int], None],
    device: torch.device,
) -> list[torch.Tensor]:
    """Final latents of a chain of overlapping windows. Each window waits in pinned host
    memory as ``[1, 2T, C]`` latent-then-reference tokens; only the ``group`` windows being
    denoised are on the GPU, so a long take needs no more VRAM than a short one.
    ``denoise(inputs, step)`` returns the denoised latents of a group's GPU inputs."""
    frames, height, width = references[0].shape[2:]
    tokens_per_frame = height * width
    tokens = frames * tokens_per_frame
    host = _pinned_windows(len(references), (1, 2 * tokens, references[0].shape[1]))
    for window, reference in zip(host, references):
        window[:, tokens:].copy_(patchify(reference))
    chain = _FusedChain(host, tokens, tokens_per_frame)
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(1, 2 * tokens, host[0].shape[-1], device=device, dtype=torch.bfloat16, generator=generator)
        chain.push(index, noise[:, :tokens].clone(), [0.0] * len(FUSION_RAMP))
    chain.flush()
    for step in range(steps):
        for start in range(0, len(host), group):
            members = range(start, min(start + group, len(host)))
            inputs = [host[i].to(device, non_blocking=True) for i in members]
            denoised = denoise(inputs, step)
            for i, x, value in zip(members, inputs, denoised):
                chain.push(i, _euler(x[:, :tokens], value, sigmas, step), FUSION_RAMP)
            advance(len(members))
        chain.flush()
    torch.cuda.current_stream(device).synchronize()
    return [unpatchify(window[:, :tokens], frames, height, width) for window in host]


def compiled_block_forward(device: torch.device):
    """``transformer.block_forward`` compiled with static shapes when Triton is available
    (one graph per canvas and per STG variant, shared by all blocks), else eager. Kernels
    are autotuned on the first compile and the picks kept in the compile cache
    (``engine_paths.configure_torch_compile_cache``), so later runs reuse them and a seed
    keeps its pixels. No CUDA graphs: every block passes its own weight tensors."""
    return compile_static(T.block_forward, mode="max-autotune-no-cudagraphs", device=device)


def _euler(sample: torch.Tensor, denoised: torch.Tensor, sigmas: torch.Tensor, step: int) -> torch.Tensor:
    """One Euler step; ``sigmas`` stays on the CPU so its scalars reach the kernels without a device sync."""
    sigma = sigmas[step]
    velocity = ((sample.float() - denoised.float()) / sigma.item()).to(sample.dtype)
    return (sample.float() + velocity.float() * (sigmas[step + 1] - sigma)).to(sample.dtype)
