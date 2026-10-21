import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media.cuda_kernel import has_precompiled_kernels
from jasna.media.yuv_to_rgb import YuvToRgbConverter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_precompiled_kernels(torch.device("cuda:0")),
    reason="needs a GPU with precompiled kernels",
)

CPU = torch.device("cpu")
GPU = torch.device("cuda:0")
COLOR_SPACES = [AvColorspace.ITU601, AvColorspace.ITU709, AvColorspace.BT2020]


def _planes(height: int, width: int, ten_bit: bool) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    peak = 1024 if ten_bit else 256
    dtype = torch.uint16 if ten_bit else torch.uint8
    shift = 6 if ten_bit else 0
    y = torch.randint(0, peak, (height, width), generator=generator, dtype=torch.int32) << shift
    uv = torch.randint(0, peak, (height // 2, width // 2, 2), generator=generator, dtype=torch.int32) << shift
    return y.to(dtype), uv.to(dtype)


@pytest.mark.parametrize("ten_bit", [False, True], ids=["nv12", "p010"])
@pytest.mark.parametrize("full_range", [False, True], ids=["limited", "full"])
@pytest.mark.parametrize("color_space", COLOR_SPACES, ids=["bt601", "bt709", "bt2020"])
def test_kernel_matches_the_eager_reference_within_one_code(color_space, full_range, ten_bit):
    height, width = 64, 96
    y, uv = _planes(height, width, ten_bit)
    reference = YuvToRgbConverter(height, width, color_space, full_range, ten_bit, CPU).convert(y, uv)

    converter = YuvToRgbConverter(height, width, color_space, full_range, ten_bit, GPU)
    assert converter._cuda_kernel is not None
    ours = converter.convert(y.to(GPU), uv.to(GPU)).cpu()

    assert (ours.int() - reference.int()).abs().max().item() <= 1
