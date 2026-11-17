import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_session_memory_release_frees_cached_memory():
    from jasna.gui.video_session import release_session_memory

    reserved_before = torch.cuda.memory_reserved()
    block = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    del block
    assert torch.cuda.memory_reserved() > reserved_before

    release_session_memory(torch.device("cuda:0"))

    assert torch.cuda.memory_reserved() <= reserved_before
