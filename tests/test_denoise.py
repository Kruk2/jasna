import torch

from jasna.restorer.denoise import DenoiseStrength, apply_denoise_u8, spatial_denoise


def test_spatial_denoise_reduces_noise() -> None:
    torch.manual_seed(42)
    T, C, H, W = 4, 3, 32, 32
    clean = torch.zeros(T, C, H, W)
    clean[:, :, :H // 2, :W // 2] = 0.2
    clean[:, :, :H // 2, W // 2:] = 0.5
    clean[:, :, H // 2:, :W // 2] = 0.7
    clean[:, :, H // 2:, W // 2:] = 0.9
    noise = torch.randn_like(clean) * 0.05
    noisy = (clean + noise).clamp(0, 1)

    denoised = spatial_denoise(noisy, kernel_size=7, sigma_spatial=3.0, sigma_range=0.12)

    noisy_mse = ((noisy - clean) ** 2).mean()
    denoised_mse = ((denoised - clean) ** 2).mean()
    assert denoised_mse < noisy_mse


def test_spatial_denoise_constant_image_unchanged() -> None:
    constant = torch.full((2, 3, 16, 16), 0.5)
    result = spatial_denoise(constant, kernel_size=5, sigma_spatial=2.0, sigma_range=0.10)
    assert torch.allclose(result, constant, atol=1e-6)


def test_spatial_denoise_preserves_edges() -> None:
    T, C, H, W = 1, 3, 32, 32
    frames = torch.zeros(T, C, H, W)
    frames[:, :, :, W // 2:] = 1.0

    denoised = spatial_denoise(frames, kernel_size=7, sigma_spatial=3.0, sigma_range=0.08)

    left = denoised[0, 0, H // 2, 0].item()
    right = denoised[0, 0, H // 2, W - 1].item()
    assert left < 0.05
    assert right > 0.95


def test_spatial_denoise_per_frame_independence() -> None:
    torch.manual_seed(0)
    frame_a = torch.rand(1, 3, 16, 16)
    frame_b = torch.rand(1, 3, 16, 16)

    single_a = spatial_denoise(frame_a, kernel_size=5, sigma_spatial=2.0, sigma_range=0.10)
    batched = spatial_denoise(torch.cat([frame_a, frame_b], dim=0), kernel_size=5, sigma_spatial=2.0, sigma_range=0.10)

    assert torch.allclose(single_a[0], batched[0], atol=1e-6)


def test_apply_denoise_u8_none_returns_unchanged() -> None:
    x = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    out = apply_denoise_u8(x, DenoiseStrength.NONE)
    assert torch.equal(out, x)
    single = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    out_single = apply_denoise_u8(single, DenoiseStrength.NONE)
    assert torch.equal(out_single, single)


def test_apply_denoise_u8_returns_uint8_same_shape() -> None:
    batch = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
    out = apply_denoise_u8(batch, DenoiseStrength.LOW)
    assert out.dtype == torch.uint8
    assert out.shape == batch.shape
    single = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    out_single = apply_denoise_u8(single, DenoiseStrength.LOW)
    assert out_single.dtype == torch.uint8
    assert out_single.shape == single.shape
