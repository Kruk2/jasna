import torch

from jasna.ltx import restore


def test_decode_window_converts_a_float32_decode(monkeypatch):
    decoded = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3, 1, 1).expand(1, 3, 3, 1, 1).contiguous()
    monkeypatch.setattr(restore, "decode_latent", lambda decoder, latent, free_bytes, generator: decoded.clone())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))
    frames = restore._decode_window(None, torch.zeros(1, 128, 1, 1, 1), 0, torch.device("cpu"))
    assert frames.dtype == torch.uint8 and frames.shape == (3, 3, 1, 1)
    assert frames[:, 0, 0, 0].tolist() == [0, 128, 255]


def test_large_canvas_needs_ten_gib_free():
    gib = 1 << 30
    assert restore.large_canvas_fits(True, 10 * gib)
    assert not restore.large_canvas_fits(True, 9 * gib)
    assert not restore.large_canvas_fits(False, 30 * gib)
