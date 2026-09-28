import pytest
import torch

from jasna.ltx.blocks import BlockStore

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _blocks(count: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "a.qdata": torch.randint(-127, 128, (64, 48), dtype=torch.int8, generator=generator),
            "a.scales": torch.rand(64, generator=generator),
            "norm.weight": torch.randn(33, generator=generator).to(torch.bfloat16),
        }
        for _ in range(count)
    ]


@pytest.mark.parametrize("resident", [0, 2, 5])
def test_streamed_and_resident_blocks_hand_out_the_same_weights(resident):
    blocks = _blocks(5)
    store = BlockStore(blocks, torch.device("cuda"), resident=resident)
    for _ in range(3):
        for index, expected in enumerate(blocks):
            weights = store.acquire(index)
            for name, tensor in expected.items():
                assert torch.equal(weights[name].cpu(), tensor)
            store.release(index)
    store.close()
