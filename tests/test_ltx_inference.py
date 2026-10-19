from __future__ import annotations

import threading
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jasna.ltx import inference, restore
from jasna.ltx.model_files import LtxModelFiles
from jasna import license_api


def _files(suffix: str = "") -> LtxModelFiles:
    return LtxModelFiles(*(Path(name + suffix) for name in ("transformer", "vae", "decoder")))


@pytest.mark.parametrize("frozen,encrypted,expected", [(False, False, False), (True, False, True), (False, True, True)])
def test_licensed_bundle_routes_to_worker(monkeypatch, frozen, encrypted, expected):
    monkeypatch.setattr(inference, "is_frozen", lambda: frozen)
    assert inference.protected_bundle(_files(".enc" if encrypted else "")) is expected


def test_frozen_trial_stays_local_and_mixed_trial_is_rejected(monkeypatch):
    monkeypatch.setattr(inference, "is_frozen", lambda: True)
    trial = LtxModelFiles.placeholder("distilled", fast=False)
    assert not inference.protected_bundle(trial)
    with pytest.raises(ValueError, match="cannot be mixed"):
        inference.protected_bundle(LtxModelFiles(trial.transformer, Path("vae.enc"), Path("decoder.enc")))


def test_protected_runtime_never_loads_models_in_parent(monkeypatch):
    def forbidden(*_args):
        pytest.fail("protected model loaded in the parent")

    monkeypatch.setattr(inference, "load_video_encoder", forbidden)
    monkeypatch.setattr(inference, "load_video_decoder", forbidden)
    monkeypatch.setattr(inference, "LtxTransformer", forbidden)
    monkeypatch.setattr(inference, "device_free_bytes", lambda device: 9 << 30)
    runtime = inference.ModelRuntime(_files(".enc"), torch.device("cpu"), threading.Event())
    requests = []
    pixels = torch.zeros(1, 3, 121, 2, 2, dtype=torch.uint8)
    latent = torch.zeros(1, 128, 16, 1, 1, dtype=torch.bfloat16)
    decoded = torch.zeros(121, 3, 2, 2, dtype=torch.uint8)

    def request(op, params, tensors, advance=None):
        requests.append((op, params, tensors))
        if op == "load_transformer":
            return {"steps": 8, "resident": 40, "count": 48}, []
        if op == "encode":
            return {}, [latent]
        if op == "denoise_chain":
            advance(8)
            return {}, [latent]
        if op == "decode":
            return {}, [decoded]
        return {}, []

    monkeypatch.setattr(runtime, "request", request)
    encoder = runtime.load_encoder()
    assert encoder(pixels) is latent
    assert runtime.load_encoder() is encoder
    runtime.unload_encoder()
    transformer = runtime.load_transformer()
    steps = []
    assert transformer.denoise_chain([latent], [43], 104, steps.append) == [latent]
    assert steps == [8]
    assert len(transformer.conditions) == 8
    assert transformer.blocks == inference.BlockLayout(40, 48)
    assert not [value for value in vars(transformer).values() if isinstance(value, torch.Tensor)]
    transformer.close()
    decoder = runtime.load_decoder()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert restore._decode_window(decoder, latent, 43, 512, lambda _canvas: 123, torch.device("cpu")) is decoded
    runtime.unload_decoder()
    assert [op for op, _, _ in requests] == [
        "load_encoder", "encode", "unload_encoder", "load_transformer", "denoise_chain",
        "unload_transformer", "load_decoder", "decode", "unload_decoder",
    ]
    assert requests[3][1] == {"free_bytes": 9 << 30}
    assert requests[4][1] == {"seeds": [43], "window_hop": 104}
    assert requests[7][1] == {"seed": 43, "free_bytes": 123}


def test_runtime_construction_and_unused_close_do_not_start_worker(monkeypatch):
    runtime = inference.ModelRuntime(_files(".enc"), torch.device("cpu"), threading.Event())
    monkeypatch.setattr(runtime, "request", lambda *_args: pytest.fail("unused worker started"))
    runtime.close()
    assert runtime._client is None


@pytest.fixture
def allocator(monkeypatch):
    """PyTorch's peak allocation stats for a backend on the CPU (the worker runs on CUDA)."""
    stats = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: stats.update(peak=0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: stats["peak"])
    return stats


def test_backend_rejects_inference_before_load_and_arbitrary_dispatch(allocator):
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    for op in ("encode", "denoise_chain", "decode"):
        with pytest.raises(RuntimeError, match="is not loaded"):
            backend.dispatch(op, {}, [], lambda _n: None)
    with pytest.raises(ValueError, match="unsupported"):
        backend.dispatch("state_dict", {}, [], lambda _n: None)
    backend.close()


def test_backend_module_unload_wipes_parameters(monkeypatch, allocator):
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    backend.encoder = torch.nn.Linear(2, 2)
    parameters = list(backend.encoder.parameters())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    backend.dispatch("unload_encoder", {}, [], lambda _n: None)
    assert backend.encoder is None
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in parameters)


def test_runtime_reuses_transport_updates_cancel_and_translates_cancellation(monkeypatch):
    clients = []

    class WorkerCancelled(Exception):
        pass

    class Client:
        def __init__(self, config, cancel):
            self.config, self.cancel, self.closed = config, cancel, False
            clients.append(self)

        def request(self, op, params, tensors, advance=None):
            if self.cancel.is_set():
                raise WorkerCancelled()
            return {}, []

        def close(self):
            self.closed = True

    monkeypatch.setattr(license_api, "require_protection", lambda: None)
    monkeypatch.setitem(sys.modules, "jasna.protection.ltx_worker", SimpleNamespace(WorkerClient=Client, WorkerCancelled=WorkerCancelled))
    runtime = inference.ModelRuntime(_files(".enc"), torch.device("cuda:0"), threading.Event())
    runtime.request("load_encoder", {}, [])
    assert Path(clients[0].config["transformer"]).is_absolute()
    next_cancel = threading.Event()
    runtime.cancel = next_cancel
    runtime.request("load_decoder", {}, [])
    assert len(clients) == 1 and clients[0].cancel is next_cancel
    next_cancel.set()
    with pytest.raises(restore.Cancelled):
        runtime.request("load_transformer", {}, [])
    assert clients[0].closed and runtime._client is None


def test_worker_peak_vram_counts_the_native_block_store(monkeypatch, allocator):
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class Transformer:
        def native_vram_bytes(self):
            return 13 << 30

        def denoise_chain(self, references, seeds, window_hop, advance):
            allocator["peak"] = 5 << 30
            return references

        def close(self):
            pass

    backend.encoder = lambda pixels: allocator.update(peak=9 << 30) or torch.zeros(1, 128, 16, 16, 16)
    backend.dispatch("encode", {}, [torch.zeros(1, 3, 121, 512, 512, dtype=torch.uint8)], lambda _n: None)
    backend.transformer = Transformer()
    backend.dispatch("denoise_chain", {"seeds": [1], "window_hop": 88}, [torch.zeros(1)], lambda _n: None)
    backend.dispatch("unload_transformer", {}, [], lambda _n: None)

    stats, _ = backend.dispatch("stats", {}, [], lambda _n: None)
    assert stats["peak_vram_bytes"] == (5 << 30) + (13 << 30)


def test_worker_sizes_the_transformer_from_the_parents_device_wide_free_vram(monkeypatch, allocator):
    """On Windows the worker's CUDA view misses the parent's and other programs' VRAM."""
    gib = 1 << 30
    worker_view = {"free": 30 * gib}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (worker_view["free"], 32 * gib))
    readers = []

    class Transformer:
        conditions = [None] * 8
        blocks = SimpleNamespace(resident=40, count=48)

        def __init__(self, path, device, free_vram):
            readers.append(free_vram)

        def native_vram_bytes(self):
            return 0

    monkeypatch.setattr(inference, "TransformerSource", lambda path: path)
    monkeypatch.setattr(inference, "LtxTransformer", Transformer)
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    backend.dispatch("load_transformer", {"free_bytes": 14 * gib}, [], lambda _n: None)
    worker_view["free"] = 25 * gib
    assert readers[0](torch.device("cpu")) == 9 * gib


def test_local_transformer_reads_device_wide_free_vram(monkeypatch):
    received = []
    monkeypatch.setattr(inference, "TransformerSource", lambda path: path)
    monkeypatch.setattr(inference, "LtxTransformer", lambda path, device, free_vram: received.append(free_vram))
    runtime = inference.ModelRuntime(LtxModelFiles.placeholder("distilled", fast=False), torch.device("cpu"), threading.Event())
    runtime.load_transformer()
    assert received == [inference.device_free_bytes]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_worker_decode_returns_its_vram_before_the_parent_reads_the_next_budget(monkeypatch):
    """The parent sizes the next window's decode from device-wide free VRAM, which counts the
    worker's allocator cache."""
    device = torch.device("cuda")
    monkeypatch.setattr(
        inference, "decode_latent", lambda decoder, latent, free_bytes, generator: torch.rand(1, 3, 33, 512, 512, device=device)
    )
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cuda"})
    backend.decoder = object()
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(device)
    _, (frames,) = backend.dispatch("decode", {"seed": 1, "free_bytes": 4 << 30}, [torch.zeros(1, 128, 5, 16, 16)], lambda _n: None)
    assert frames.device.type == "cpu" and frames.shape == (33, 3, 512, 512)
    assert torch.cuda.memory_reserved(device) == reserved


def test_worker_opens_the_transformer_file_ahead_on_its_own_thread(monkeypatch, allocator):
    """``prepare_transformer`` runs while the encoder holds the GPU: the file is opened on another
    thread and the later load takes that source instead of opening the file again."""
    opened, loaded, released = [], [], []

    class Source:
        def __init__(self, path):
            opened.append((path, threading.current_thread() is not threading.main_thread()))

        def close(self):
            released.append(self)

    class Transformer:
        conditions = [None] * 4
        blocks = SimpleNamespace(resident=48, count=48)

        def __init__(self, source, device, free_vram):
            loaded.append(source)

        def native_vram_bytes(self):
            return 0

        def close(self):
            pass

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(inference, "TransformerSource", Source)
    monkeypatch.setattr(inference, "LtxTransformer", Transformer)
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    backend.dispatch("prepare_transformer", {}, [], lambda _n: None)
    backend.dispatch("prepare_transformer", {}, [], lambda _n: None)
    backend.dispatch("load_transformer", {"free_bytes": 8 << 30}, [], lambda _n: None)
    backend.dispatch("prepare_transformer", {}, [], lambda _n: None)
    assert opened == [(Path("t.enc"), True)] and len(loaded) == 1 and isinstance(loaded[0], Source)
    backend.dispatch("unload_transformer", {}, [], lambda _n: None)
    backend.dispatch("prepare_transformer", {}, [], lambda _n: None)
    backend.close()
    assert len(opened) == 2 and len(released) == 1 and released[0] is not loaded[0]


def test_unused_transformer_file_opened_ahead_closes_with_the_worker_even_after_failing(monkeypatch, allocator):
    def failing(path):
        raise RuntimeError("synthetic license failure")

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 << 30, 32 << 30))
    monkeypatch.setattr(inference, "TransformerSource", failing)
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    backend.dispatch("prepare_transformer", {}, [], lambda _n: None)
    backend.close()
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    backend.dispatch("prepare_transformer", {}, [], lambda _n: None)
    with pytest.raises(RuntimeError, match="synthetic license failure"):
        backend.dispatch("load_transformer", {"free_bytes": 8 << 30}, [], lambda _n: None)
    backend.close()


def test_runtime_asks_the_worker_to_open_the_transformer_only_while_none_is_loaded(monkeypatch):
    requests = []
    runtime = inference.ModelRuntime(_files(".enc"), torch.device("cpu"), threading.Event())
    monkeypatch.setattr(inference, "device_free_bytes", lambda device: 9 << 30)

    def request(op, params, tensors, advance=None):
        requests.append(op)
        return ({"steps": 4, "resident": 48, "count": 48} if op == "load_transformer" else {}), []

    monkeypatch.setattr(runtime, "request", request)
    runtime.prepare_transformer()
    runtime.load_transformer()
    runtime.prepare_transformer()
    assert requests == ["prepare_transformer", "load_transformer"]
    local = inference.ModelRuntime(LtxModelFiles.placeholder("distilled", fast=False), torch.device("cpu"), threading.Event())
    monkeypatch.setattr(local, "request", lambda *_args: pytest.fail("trial files have no worker"))
    local.prepare_transformer()
