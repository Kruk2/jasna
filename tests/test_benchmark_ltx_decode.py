import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ltx_decode as bench


def test_unknown_variant_option_is_rejected():
    with pytest.raises(ValueError, match="fast"):
        bench.variant_patches("eager+fast")


def test_psnr_of_identical_decodes_is_infinite():
    video = torch.rand(1, 3, 2, 4, 4) - 0.5
    assert bench.psnr(video, video) == float("inf")
    assert bench.psnr(video, video + 0.1) == pytest.approx(10 * torch.log10(torch.tensor(4 / 0.01)).item(), rel=1e-4)
