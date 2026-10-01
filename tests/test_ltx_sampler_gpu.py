"""The sampler's per-step arithmetic on the GPU: no host syncs, same bits as device-side sigmas."""

from __future__ import annotations

import pytest
import torch

from jasna.ltx.plan import FUSION_RAMP
from jasna.ltx.sampler import _euler, fuse_overlaps

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
        fuse_overlaps(latents, 64, FUSION_RAMP)
    finally:
        torch.cuda.set_sync_debug_mode(0)
