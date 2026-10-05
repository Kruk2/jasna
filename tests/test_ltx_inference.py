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
    assert transformer.denoise_chain([latent], [43], steps.append) == [latent]
    assert steps == [8]
    assert len(transformer.conditions) == 8
    assert transformer.blocks == inference.BlockLayout(40, 48)
    assert not hasattr(transformer, "top")
    assert not hasattr(transformer, "context")
    transformer.close()
    decoder = runtime.load_decoder()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert restore._decode_window(decoder, latent, 43, 512, lambda _canvas: 123, torch.device("cpu")) is decoded
    runtime.unload_decoder()
    assert [op for op, _, _ in requests] == [
        "load_encoder", "encode", "unload_encoder", "load_transformer", "denoise_chain",
        "unload_transformer", "load_decoder", "decode", "unload_decoder",
    ]
    assert requests[4][1] == {"seeds": [43]}
    assert requests[7][1] == {"seed": 43, "free_bytes": 123}


def test_runtime_construction_and_unused_close_do_not_start_worker(monkeypatch):
    runtime = inference.ModelRuntime(_files(".enc"), torch.device("cpu"), threading.Event())
    monkeypatch.setattr(runtime, "request", lambda *_args: pytest.fail("unused worker started"))
    runtime.close()
    assert runtime._client is None


def test_backend_rejects_inference_before_load_and_arbitrary_dispatch():
    backend = inference.WorkerBackend({"transformer": "t.enc", "vae": "v.enc", "tuned_decoder": "d.enc", "device": "cpu"})
    for op in ("encode", "denoise_chain", "decode"):
        with pytest.raises(RuntimeError, match="is not loaded"):
            backend.dispatch(op, {}, [], lambda _n: None)
    with pytest.raises(ValueError, match="unsupported"):
        backend.dispatch("state_dict", {}, [], lambda _n: None)
    backend.close()


def test_backend_module_unload_wipes_parameters(monkeypatch):
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
