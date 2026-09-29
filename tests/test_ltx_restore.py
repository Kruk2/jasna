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


def test_denoise_reports_every_window_step(monkeypatch):
    class Bar:
        def __init__(self, total):
            self.total, self.n = total, 0

        def update(self, n):
            self.n += n

        def close(self):
            pass

    class Store:
        def get(self, index):
            return index

        def put(self, index, latent):
            pass

        def delete(self, index):
            pass

    class Transformer:
        conditions = [None] * 8

        def denoise_chain(self, references, seeds, advance):
            for _ in self.conditions:
                advance(len(references))
            return list(references)

    bars = []
    monkeypatch.setattr(restore.Progress, "bar", lambda self, name, total, unit="frame": bars.append(Bar(total)) or bars[-1])
    window = lambda i: restore.Window(index=i, track_id=0, start=0, real_frames=1, crops=())
    plans = [
        restore.TrackPlan(track_id=0, start=0, polygons=(), windows=(window(0), window(1), window(2))),
        restore.TrackPlan(track_id=1, start=0, polygons=(), windows=(window(3),)),
    ]
    restore.denoise(
        plans, Transformer(), Store(), Store(), seed=0,
        progress=restore.Progress(frames=0, disable=True), cancel=restore.threading.Event(),
    )
    assert bars[0].total == bars[0].n == 32
