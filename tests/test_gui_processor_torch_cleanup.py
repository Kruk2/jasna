import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cleanup_torch_releases_cached_memory_and_resets_the_peak():
    from jasna.gui.processor import _cleanup_torch

    reserved_before = torch.cuda.memory_reserved()
    block = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    del block
    assert torch.cuda.memory_reserved() > reserved_before

    _cleanup_torch(torch)

    assert torch.cuda.memory_reserved() <= reserved_before
    assert torch.cuda.max_memory_allocated() == torch.cuda.memory_allocated()
