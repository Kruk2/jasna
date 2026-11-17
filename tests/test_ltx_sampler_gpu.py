"""The sampler's per-step arithmetic on the GPU: no host syncs, same bits as device-side sigmas."""

from __future__ import annotations

import pytest
import torch

from jasna.ltx.windows import LATENT_FRAMES, WINDOW_HOP, shared_latents, window_hop
from jasna.ltx.sampler import _euler, denoise_windows, fuse_overlaps

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


@pytest.mark.parametrize("window_hop", [WINDOW_HOP, window_hop(17)])
@pytest.mark.parametrize("group", [3, 7])
def test_denoising_windows_in_groups_gives_the_same_bits_as_one_at_a_time(group, window_hop):
    references = _windows(7)
    expected = _denoise_windows(references, 1, [], window_hop)
    out = _denoise_windows(references, group, [], window_hop)
    assert len(out) == 7
    assert all(torch.equal(a, b) for a, b in zip(out, expected))


def test_denoise_windows_keeps_only_the_group_being_denoised_on_the_gpu():
    peaks = {}
    for count in (4, 64):
        peaks[count] = []
        _denoise_windows(_windows(count), 2, peaks[count], WINDOW_HOP)
    window_bytes = 2 * 128 * LATENT_FRAMES * 4 * 4 * 2
    assert max(peaks[64]) - max(peaks[4]) < window_bytes
