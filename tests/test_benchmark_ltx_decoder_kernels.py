import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ltx_decoder_kernels as bench


def test_the_distance_of_a_decode_is_measured_on_the_frames_compose_makes():
    levels = torch.randint(10, 240, (3, 2, 8, 8)).float()
    video = levels / 255 * 2 - 1
    two_levels_brighter = (levels + 2) / 255 * 2 - 1

    psnr, mean_abs = bench.distance(two_levels_brighter, video)

    assert bench.distance(video, video) == (math.inf, 0.0)
    assert mean_abs == 2.0
    assert psnr == pytest.approx(10 * math.log10(4 / (4 / 255) ** 2), abs=1e-3)


def test_sweeping_one_mlp_kernel_keeps_the_other_widths_and_kernels():
    from jasna.models.ltx_vae import diffusion_triton

    entry = diffusion_triton.NVIDIA_DEFAULT_LAUNCH
    swept = bench.swiglu_entry(entry, 512, down=(1, 2, 3, 4, 5))

    assert swept.swiglu[512] == (entry.swiglu[512][0], (1, 2, 3, 4, 5))
    assert {dim: configs for dim, configs in swept.swiglu.items() if dim != 512} == {
        dim: configs for dim, configs in entry.swiglu.items() if dim != 512}
    assert swept._replace(swiglu=entry.swiglu) == entry
