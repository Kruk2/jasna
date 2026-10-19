"""LTX model ownership and the tensor-only protected inference boundary."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from jasna._frozen import is_frozen
from jasna.accelerator import device_free_bytes
from jasna.ltx.model_files import LtxModelFiles, PlaceholderTensors
from jasna.ltx.sampler import LtxTransformer, TransformerSource
from jasna.models.ltx_vae import decode_latent, load_video_decoder, load_video_encoder

logger = logging.getLogger(__name__)


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
        self,
        references: Sequence[torch.Tensor],
        seeds: Sequence[int],
        window_hop: int,
        advance: Callable[[int], None],
    ) -> list[torch.Tensor]:
        _, tensors = self._runtime.request(
            "denoise_chain", {"seeds": list(seeds), "window_hop": window_hop}, list(references), advance
        )
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

    def prepare_transformer(self) -> None:
        """Let the worker open the transformer file and read its host-side part while the GPU is
        still busy (encode); nothing of the transformer reaches the GPU before ``load_transformer``."""
        if self.protected and self._transformer is None:
            self.request("prepare_transformer", {}, [])

    def load_transformer(self):
        if self._transformer is None:
            if self.protected:
                info, _ = self.request("load_transformer", {"free_bytes": device_free_bytes(self.device)}, [])
                self._transformer = WorkerTransformer(self, info)
            else:
                self._transformer = LtxTransformer(TransformerSource(self.files.transformer), self.device,
                                                   device_free_bytes)
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


def free_vram_beside_parent(device: torch.device, parent_free_bytes: int) -> Callable[[torch.device], int]:
    """Device-wide free VRAM inside the worker, from the parent's device-wide figure taken just
    before loading: on Windows the worker's CUDA view misses every other process's memory (the
    parent's included), so that gap is subtracted from it from then on."""
    hidden = max(0, torch.cuda.mem_get_info(device)[0] - parent_free_bytes)
    return lambda current: max(0, torch.cuda.mem_get_info(current)[0] - hidden)


class WorkerBackend:
    """Only the authenticated child owns models; dispatch exposes whole inference operations."""

    def __init__(self, config: dict) -> None:
        self.device = torch.device(config["device"])
        self.files = LtxModelFiles(*(Path(config[key]) for key in ("transformer", "vae", "tuned_decoder")))
        self.encoder = None
        self.transformer = None
        self.decoder = None
        self.peak_vram_bytes = 0
        self._preparer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxTransformerSource")
        self._prepared: Future[TransformerSource] | None = None

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
        if op == "stats":
            return {"peak_vram_bytes": self.peak_vram_bytes}, []
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(self.device)
        result = self._run(op, params, tensors, advance)
        if torch.cuda.is_initialized():
            self._record_peak_vram()
        return result

    def _record_peak_vram(self) -> None:
        """The operation's PyTorch peak plus the transformer blocks the native store holds
        outside PyTorch's allocator."""
        native = self.transformer.native_vram_bytes() if self.transformer is not None else 0
        self.peak_vram_bytes = max(self.peak_vram_bytes, torch.cuda.max_memory_allocated(self.device) + native)

    def _run(self, op: str, params: dict, tensors: list[torch.Tensor], advance: Callable[[int], None]):
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
        elif op == "prepare_transformer":
            if self.transformer is None and self._prepared is None:
                self._prepared = self._preparer.submit(TransformerSource, self.files.transformer)
        elif op == "load_transformer":
            if self.transformer is None:
                free_vram = free_vram_beside_parent(self.device, params["free_bytes"])
                self.transformer = LtxTransformer(self._transformer_source(), self.device, free_vram)
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
            return {}, self.transformer.denoise_chain(tensors, params["seeds"], params["window_hop"], advance)
        elif op == "load_decoder":
            if self.decoder is None:
                self.decoder = load_video_decoder(self.files.vae, self.files.tuned_decoder, self.device)
        elif op == "unload_decoder":
            self._unload_module("decoder")
        elif op == "decode":
            if self.decoder is None:
                raise RuntimeError("LTX decoder is not loaded")
            frames = self._decode_to_cpu(tensors[0], params["seed"], params["free_bytes"])
            torch.cuda.empty_cache()
            return {}, [frames]
        else:
            raise ValueError("unsupported LTX inference operation")
        return {}, []

    def _transformer_source(self) -> TransformerSource:
        """The file opened ahead by ``prepare_transformer``, else opened now."""
        prepared, self._prepared = self._prepared, None
        return prepared.result() if prepared is not None else TransformerSource(self.files.transformer)

    def _drop_prepared(self) -> None:
        prepared, self._prepared = self._prepared, None
        if prepared is None:
            return
        try:
            source = prepared.result()
        except Exception as error:
            logger.debug("LTX transformer file opened ahead was never used; opening it had failed: %r", error)
            return
        source.close()

    def _decode_to_cpu(self, latent: torch.Tensor, seed: int, free_bytes: int) -> torch.Tensor:
        """Decoded window as uint8 ``[F, 3, S, S]`` on the CPU; its GPU copies are freed on return,
        so the caller can hand that cache back before the parent reads the next window's budget."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video = decode_latent(self.decoder, latent.to(self.device), free_bytes, generator)
        video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)[0].float()
        return video.mul_(255.0).round_().to(torch.uint8).permute(1, 0, 2, 3).contiguous().to("cpu")

    def close(self) -> None:
        self._drop_prepared()
        self._preparer.shutdown()
        self._unload_module("encoder")
        if self.transformer is not None:
            self.transformer.close()
            self.transformer = None
        self._unload_module("decoder")
