"""LTX model ownership and the tensor-only protected inference boundary."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from jasna._frozen import is_frozen
from jasna.ltx.model_files import LtxModelFiles, PlaceholderTensors
from jasna.ltx.sampler import LtxTransformer
from jasna.models.ltx_vae import decode_latent, load_video_decoder, load_video_encoder


def protected_bundle(files: LtxModelFiles) -> bool:
    sources = (files.transformer, files.vae, files.tuned_decoder)
    if all(isinstance(source, PlaceholderTensors) for source in sources):
        return False
    if any(isinstance(source, PlaceholderTensors) for source in sources):
        raise ValueError("LTX trial and licensed model files cannot be mixed")
    return is_frozen() or any(source.suffix == ".enc" for source in sources)


@dataclass(frozen=True)
class BlockLayout:
    resident: int
    count: int


class WorkerEncoder:
    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        _, tensors = self._runtime.request("encode", {}, [pixels])
        return tensors[0]


class WorkerTransformer:
    def __init__(self, runtime: ModelRuntime, info: dict) -> None:
        self._runtime = runtime
        self.conditions = range(info["steps"])
        self.blocks = BlockLayout(info["resident"], info["count"])

    def denoise_chain(
        self, references: Sequence[torch.Tensor], seeds: Sequence[int], advance: Callable[[int], None]
    ) -> list[torch.Tensor]:
        _, tensors = self._runtime.request("denoise_chain", {"seeds": list(seeds)}, list(references), advance)
        return tensors

    def close(self) -> None:
        self._runtime.unload_transformer()


class WorkerDecoder:
    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    def decode_window(self, latent: torch.Tensor, seed: int, free_bytes: int, device: torch.device) -> torch.Tensor:
        _, tensors = self._runtime.request("decode", {"seed": seed, "free_bytes": free_bytes}, [latent])
        return tensors[0].to(device)


class ModelRuntime:
    """One lazy worker reused across LTX passes; plaintext development and trial stay local."""

    def __init__(self, files: LtxModelFiles, device: torch.device, cancel: threading.Event) -> None:
        self.files = files
        self.device = device
        self.cancel = cancel
        self.protected = protected_bundle(files)
        self._client = None
        self._encoder = None
        self._transformer = None
        self._decoder = None

    def request(self, op: str, params: dict, tensors: list[torch.Tensor], advance=None):
        if self._client is None:
            from jasna.license_api import require_protection

            require_protection()
            from jasna.protection.ltx_worker import WorkerClient

            self._client = WorkerClient(
                {"transformer": str(self.files.transformer.resolve()), "vae": str(self.files.vae.resolve()),
                 "tuned_decoder": str(self.files.tuned_decoder.resolve()), "device": str(self.device)}, self.cancel
            )
        self._client.cancel = self.cancel
        from jasna.protection.ltx_worker import WorkerCancelled

        try:
            return self._client.request(op, params, tensors, advance=advance)
        except WorkerCancelled:
            from jasna.ltx.restore import Cancelled

            self.close()
            raise Cancelled() from None

    def load_encoder(self):
        if self._encoder is None:
            if self.protected:
                self.request("load_encoder", {}, [])
                self._encoder = WorkerEncoder(self)
            else:
                self._encoder = load_video_encoder(self.files.vae, self.device)
        return self._encoder

    def unload_encoder(self) -> None:
        if self._encoder is not None:
            self._encoder = None
            if self.protected:
                self.request("unload_encoder", {}, [])
            else:
                torch.cuda.empty_cache()

    def load_transformer(self):
        if self._transformer is None:
            if self.protected:
                info, _ = self.request("load_transformer", {}, [])
                self._transformer = WorkerTransformer(self, info)
            else:
                self._transformer = LtxTransformer(self.files.transformer, self.device)
        return self._transformer

    def unload_transformer(self) -> None:
        if self._transformer is not None:
            model, self._transformer = self._transformer, None
            if self.protected:
                self.request("unload_transformer", {}, [])
            else:
                model.close()
                del model
                torch.cuda.empty_cache()

    def load_decoder(self):
        if self._decoder is None:
            if self.protected:
                self.request("load_decoder", {}, [])
                self._decoder = WorkerDecoder(self)
            else:
                self._decoder = load_video_decoder(self.files.vae, self.files.tuned_decoder, self.device)
        return self._decoder

    def unload_decoder(self) -> None:
        if self._decoder is not None:
            self._decoder = None
            if self.protected:
                self.request("unload_decoder", {}, [])
            else:
                torch.cuda.empty_cache()

    def close(self) -> None:
        if self.protected:
            self._encoder = self._transformer = self._decoder = None
            if self._client is not None:
                self._client.close()
                self._client = None
        else:
            self.unload_encoder()
            self.unload_transformer()
            self.unload_decoder()

    def peak_vram_bytes(self) -> int:
        parent_peak = torch.cuda.max_memory_allocated(self.device)
        if self.protected and self._client is not None:
            stats, _ = self.request("stats", {}, [])
            return parent_peak + stats["peak_vram_bytes"]
        return parent_peak

    def __enter__(self) -> ModelRuntime:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class WorkerBackend:
    """Only the authenticated child owns models; dispatch exposes whole inference operations."""

    def __init__(self, config: dict) -> None:
        self.device = torch.device(config["device"])
        self.files = LtxModelFiles(*(Path(config[key]) for key in ("transformer", "vae", "tuned_decoder")))
        self.encoder = None
        self.transformer = None
        self.decoder = None

    def _unload_module(self, name: str) -> None:
        from jasna.models.ltx_vae.loader import _clear_tensors

        model = getattr(self, name)
        if model is not None:
            setattr(self, name, None)
            _clear_tensors(model.state_dict().values(), None)
            del model
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def dispatch(self, op: str, params: dict, tensors: list[torch.Tensor], advance: Callable[[int], None]):
        if op == "load_encoder":
            if self.encoder is None:
                self.encoder = load_video_encoder(self.files.vae, self.device)
        elif op == "unload_encoder":
            self._unload_module("encoder")
        elif op == "encode":
            if self.encoder is None:
                raise RuntimeError("LTX encoder is not loaded")
            pixels = tensors[0].to(self.device).float().div_(127.5).sub_(1.0).to(torch.bfloat16)
            return {}, [self.encoder(pixels).to("cpu").contiguous()]
        elif op == "load_transformer":
            if self.transformer is None:
                self.transformer = LtxTransformer(self.files.transformer, self.device)
            return {"steps": len(self.transformer.conditions), "resident": self.transformer.blocks.resident,
                    "count": self.transformer.blocks.count}, []
        elif op == "unload_transformer":
            if self.transformer is not None:
                self.transformer.close()
                self.transformer = None
                torch.cuda.empty_cache()
        elif op == "denoise_chain":
            if self.transformer is None:
                raise RuntimeError("LTX transformer is not loaded")
            return {}, self.transformer.denoise_chain(tensors, params["seeds"], advance)
        elif op == "load_decoder":
            if self.decoder is None:
                self.decoder = load_video_decoder(self.files.vae, self.files.tuned_decoder, self.device)
        elif op == "unload_decoder":
            self._unload_module("decoder")
        elif op == "decode":
            if self.decoder is None:
                raise RuntimeError("LTX decoder is not loaded")
            torch.cuda.empty_cache()
            generator = torch.Generator(device=self.device).manual_seed(params["seed"])
            video = decode_latent(self.decoder, tensors[0].to(self.device), params["free_bytes"], generator)
            video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)[0].float()
            frames = video.mul_(255.0).round_().to(torch.uint8).permute(1, 0, 2, 3).contiguous()
            return {}, [frames.to("cpu")]
        elif op == "stats":
            return {"peak_vram_bytes": torch.cuda.max_memory_allocated(self.device)}, []
        else:
            raise ValueError("unsupported LTX inference operation")
        return {}, []

    def close(self) -> None:
        self._unload_module("encoder")
        if self.transformer is not None:
            self.transformer.close()
            self.transformer = None
        self._unload_module("decoder")
