import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ltx_encode as bench


def test_unknown_variant_option_is_rejected():
    with pytest.raises(ValueError, match="fast"):
        bench.variant_encoder("eager+fast", encoder=None)


def test_psnr_is_over_the_reference_latent_range():
    reference = torch.tensor([0.0, 2.0, 1.0, 1.0])
    assert bench.psnr(reference, reference) == float("inf")
    assert bench.psnr(reference + 0.1, reference) == pytest.approx(10 * torch.log10(torch.tensor(4 / 0.01)).item(), rel=1e-4)
