import pytest
import torch

import jasna.restorer.denoise as denoise
from jasna.restorer.denoise import DenoiseStrength, apply_denoise_u8, spatial_denoise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture
def torch_path(monkeypatch):
    """Runs spatial_denoise through the Torch expression the kernel replaced."""

    def run(callable_):
        monkeypatch.setattr(denoise, "is_nvidia_device", lambda device: False)
        try:
            return callable_()
        finally:
            monkeypatch.undo()

    return run


CASES = [
    (4, 64, 64, 5, 1.5, 0.07),
    (4, 64, 64, 7, 3.0, 0.12),
    (2, 33, 41, 5, 2.0, 0.10),
    (1, 5, 5, 5, 2.0, 0.10),
]


@pytest.mark.parametrize(("count", "height", "width", "size", "sigma_s", "sigma_r"), CASES)
def test_kernel_matches_the_torch_expression(
    torch_path, count, height, width, size, sigma_s, sigma_r
):
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand(
        count, 3, height, width, generator=generator, device=_device()
    )

    got = spatial_denoise(frames, size, sigma_s, sigma_r)
    reference = torch_path(lambda: spatial_denoise(frames, size, sigma_s, sigma_r))

    assert torch.allclose(got, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("strength", [DenoiseStrength.LOW, DenoiseStrength.MEDIUM, DenoiseStrength.HIGH])
def test_uint8_output_stays_within_one_code(torch_path, strength):
    generator = torch.Generator(device="cuda").manual_seed(1)
    clip = torch.randint(
        0, 256, (6, 3, 256, 256), generator=generator, device=_device(), dtype=torch.uint8
    )

    got = apply_denoise_u8(clip, strength)
    reference = torch_path(lambda: apply_denoise_u8(clip, strength))

    # The kernel sums the window in a different order than 25 chained tensor
    # ops, which is worth about 2e-7 and only matters where a sample sits on a
    # rounding boundary: no sample seen off by more than one code, and under a
    # thousandth of a percent off at all even at the strongest setting.
    difference = (got.int() - reference.int()).abs()
    assert difference.max().item() <= 1
    assert (difference != 0).float().mean().item() < 1e-5


def test_constant_image_is_unchanged():
    frames = torch.full((2, 3, 32, 32), 0.4, device=_device())

    out = spatial_denoise(frames, 5, 2.0, 0.10)

    assert torch.allclose(out, frames, atol=1e-6)


def test_frames_are_filtered_independently():
    generator = torch.Generator(device="cuda").manual_seed(2)
    a = torch.rand(1, 3, 24, 24, generator=generator, device=_device())
    b = torch.rand(1, 3, 24, 24, generator=generator, device=_device())

    alone = spatial_denoise(a, 5, 2.0, 0.10)
    batched = spatial_denoise(torch.cat([a, b], dim=0), 5, 2.0, 0.10)

    assert torch.allclose(alone[0], batched[0], atol=1e-6)


def test_edges_use_reflect_padding(torch_path):
    """A window at the border must mirror, not clamp or wrap."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    frames = torch.rand(1, 3, 9, 9, generator=generator, device=_device())

    got = spatial_denoise(frames, 5, 2.0, 0.10)
    reference = torch_path(lambda: spatial_denoise(frames, 5, 2.0, 0.10))

    assert torch.allclose(got[:, :, :2], reference[:, :, :2], atol=1e-6)
    assert torch.allclose(got[:, :, -2:], reference[:, :, -2:], atol=1e-6)
    assert torch.allclose(got[:, :, :, :2], reference[:, :, :, :2], atol=1e-6)
    assert torch.allclose(got[:, :, :, -2:], reference[:, :, :, -2:], atol=1e-6)


def test_output_is_a_fresh_tensor():
    frames = torch.rand(2, 3, 16, 16, device=_device())

    out = spatial_denoise(frames, 5, 2.0, 0.10)

    assert out.data_ptr() != frames.data_ptr()
    assert out.shape == frames.shape
    assert out.dtype is frames.dtype
