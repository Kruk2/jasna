"""The histogram median of the LTX colour match on the GPU against torch.quantile."""

import pytest
import torch

from jasna.ltx.compose import masked_median

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_masked_median_equals_quantile_without_a_host_sync():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    for spread in (0, 1, 7, 255):
        for density in (0.0, 1e-5, 0.3, 1.0):
            difference = torch.randint(-spread, spread + 1, (3, 300, 500), generator=generator, device=device).float()
            mask = torch.rand(300, 500, generator=generator, device=device) < density
            torch.cuda.set_sync_debug_mode("error")
            try:
                median = masked_median(difference, mask)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            if mask.any():
                expected = torch.quantile(difference[:, mask], 0.5, dim=1)
            else:
                expected = torch.zeros(3, device=device)
            assert torch.equal(median, expected)
