import copy
import warnings

import pytest
import torch

rfdetr = pytest.importorskip("rfdetr")

from jasna.mosaic.rfdetr_sync_free import export_without_host_syncs

RESOLUTION = 312


def _random_seg_core(device: str) -> torch.nn.Module:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper = rfdetr.RFDETRSegNano(num_classes=1, resolution=RESOLUTION, pretrain_weights=None, device=device)
    return wrapper.model.model.to(device).eval()


def test_matches_rfdetrs_export_forward():
    reference = _random_seg_core("cpu")
    sync_free = copy.deepcopy(reference)
    reference.export()
    export_without_host_syncs(sync_free)
    x = torch.randn(2, 3, RESOLUTION, RESOLUTION, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        expected = reference(x)
        actual = sync_free(x)

    assert len(actual) == len(expected) == 3
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_forward_never_waits_for_the_gpu():
    core = _random_seg_core("cuda").half()
    export_without_host_syncs(core)
    x = torch.randn(2, 3, RESOLUTION, RESOLUTION, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        core(x)
        torch.cuda.set_sync_debug_mode("error")
        try:
            core(x)
        finally:
            torch.cuda.set_sync_debug_mode("default")
