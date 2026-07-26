import pytest
import torch

from jasna.media.cas import GpuCasSharpener

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _luma(ten_bit: bool) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    if ten_bit:
        codes = torch.randint(
            64, 941, (48, 64), generator=generator, device="cuda", dtype=torch.int32
        )
        return (codes << 6).to(torch.int16)
    return torch.randint(
        16, 236, (48, 64), generator=generator, device="cuda", dtype=torch.uint8
    )


@pytest.mark.parametrize("ten_bit", [False, True])
def test_sharpen_into_matches_the_in_place_path(ten_bit):
    device = torch.device("cuda:0")
    sharpener = GpuCasSharpener(0.6, ten_bit=ten_bit, device=device)
    source = _luma(ten_bit)

    packed = torch.cat([source, source[: source.shape[0] // 2]])
    sharpener.apply_luma_(packed, source.shape[0])

    destination = torch.empty_like(source)
    sharpener.sharpen_into(source, destination)

    assert torch.equal(destination, packed[: source.shape[0]])


def test_sharpen_into_leaves_the_source_untouched():
    device = torch.device("cuda:0")
    sharpener = GpuCasSharpener(0.6, ten_bit=False, device=device)
    source = _luma(False)
    original = source.clone()

    sharpener.sharpen_into(source, torch.empty_like(source))

    assert torch.equal(source, original)
