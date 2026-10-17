"""The sampler's per-step arithmetic on the GPU: no host syncs, same bits as device-side sigmas."""

from __future__ import annotations

import pytest
import torch

from jasna.ltx.plan import LATENT_FRAMES, LESS_OVERLAP_WINDOW_HOP, WINDOW_HOP, shared_latents
from jasna.ltx.sampler import _euler, denoise_windows, fuse_overlaps, patchify, unpatchify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SIGMAS = torch.tensor([1.0, 0.99375, 0.9875, 0.975, 0.909375, 0.725, 0.421875, 0.0], dtype=torch.float32)


def test_euler_with_cpu_sigmas_matches_device_sigmas_bit_for_bit():
    generator = torch.Generator(device="cuda").manual_seed(0)
    sample = torch.randn(1, 4096, 128, device="cuda", generator=generator).bfloat16()
    for step in range(len(SIGMAS) - 1):
        denoised = torch.randn(1, 4096, 128, device="cuda", generator=generator).bfloat16()
        expected = _euler(sample, denoised, SIGMAS.cuda(), step)
        sample = _euler(sample, denoised, SIGMAS, step)
        assert torch.equal(sample, expected)


def test_velocity_scaled_by_a_cpu_sigma_matches_a_device_sigma():
    velocity = torch.randn(1, 4096, 128, device="cuda").bfloat16()
    for sigma in SIGMAS:
        assert torch.equal(velocity.float() * sigma, velocity.float() * sigma.cuda())


def test_euler_and_overlap_fusion_do_not_sync_the_device():
    latents = [torch.randn(1, 16 * 64, 128, device="cuda").bfloat16() for _ in range(3)]
    denoised = torch.randn(1, 16 * 64, 128, device="cuda").bfloat16()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        latents = [_euler(latent, denoised, SIGMAS, 0) for latent in latents]
        shared = shared_latents(WINDOW_HOP)
        fuse_overlaps(latents, 64, shared, shared.ramp)
    finally:
        torch.cuda.set_sync_debug_mode(0)


def _windows(count: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randn(1, 128, LATENT_FRAMES, 4, 4, generator=generator).bfloat16() for _ in range(count)]


def _fake_denoise(peaks: list[int]):
    def denoise(inputs, step):
        peaks.append(torch.cuda.memory_allocated())
        half = inputs[0].shape[1] // 2
        return [(x[:, :half].float() * 0.7 + x[:, half:].float() * 0.2 + 0.01 * step).bfloat16() for x in inputs]

    return denoise


def _all_on_gpu_chain(references, seeds, group, denoise, window_hop):
    """The sampler as it was with every window's latent and reference on the GPU."""
    shared = shared_latents(window_hop)
    frames, height, width = references[0].shape[2:]
    tokens_per_frame = height * width
    tokens = frames * tokens_per_frame
    refs = [patchify(r.cuda()) for r in references]
    latents = []
    for seed in seeds:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(1, 2 * tokens, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        latents.append(noise[:, :tokens].clone())
    fuse_overlaps(latents, tokens_per_frame, shared, [0.0] * len(shared.ramp))
    for step in range(len(SIGMAS) - 1):
        for start in range(0, len(latents), group):
            members = range(start, min(start + group, len(latents)))
            denoised = denoise([torch.cat([latents[i], refs[i]], dim=1) for i in members], step)
            for i, value in zip(members, denoised):
                latents[i] = _euler(latents[i], value, SIGMAS, step)
        fuse_overlaps(latents, tokens_per_frame, shared, shared.ramp)
    return [unpatchify(latent, frames, height, width) for latent in latents]


def _denoise_windows(references, group, peaks, window_hop):
    return denoise_windows(
        references,
        list(range(len(references))),
        SIGMAS,
        window_hop=window_hop,
        steps=len(SIGMAS) - 1,
        group=group,
        denoise=_fake_denoise(peaks),
        advance=lambda _n: None,
        device=torch.device("cuda"),
    )


@pytest.mark.parametrize("window_hop", [WINDOW_HOP, LESS_OVERLAP_WINDOW_HOP])
@pytest.mark.parametrize("group", [1, 3, 7])
def test_denoise_windows_matches_the_all_on_gpu_chain_bit_for_bit(group, window_hop):
    references = _windows(7)
    expected = _all_on_gpu_chain(references, list(range(7)), group, _fake_denoise([]), window_hop)
    out = _denoise_windows(references, group, [], window_hop)
    assert len(out) == 7
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(out, expected))


def test_denoise_windows_keeps_only_the_group_being_denoised_on_the_gpu():
    peaks = {}
    for count in (4, 64):
        peaks[count] = []
        _denoise_windows(_windows(count), 2, peaks[count], WINDOW_HOP)
    window_bytes = 2 * 128 * LATENT_FRAMES * 4 * 4 * 2
    assert max(peaks[64]) - max(peaks[4]) < window_bytes
