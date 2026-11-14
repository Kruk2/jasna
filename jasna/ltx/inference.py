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
from jasna.accelerator import device_free_bytes, load_gpu_libraries_ahead
from jasna.ltx.model_files import LtxModelFiles, PlaceholderTensors
from jasna.ltx.sampler import LtxTransformer, TransformerSource
from jasna.models.ltx_vae import decode_latent, load_video_decoder, load_video_encoder
from jasna.tensor_utils import half_dtype

logger = logging.getLogger(__name__)

GPU_WINDOWS_MIN_FREE_BYTES = 18 << 30
"""Decode budget from which decoded windows stay on the GPU: a window only stays there while the
budget it was decoded with reaches this, so the ones held (~95 MB each at 512 px, ~210 MB at
768) still leave every later decode the 14.75 GiB that decodes a 768 px window in one tile, so
no tile layout changes, and each frame skips uploading its canvases. From it on, an allocator's
cached blocks stay cached between decodes: freeing them could change neither."""


def protected_bundle(files: LtxModelFiles) -> bool:
    sources = (files.transformer, files.vae, files.tuned_decoder, files.video_encoder)
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

    def send(self, pixels: torch.Tensor) -> PendingReply:
        """Send one window's uint8 ``[1, 3, F, S, S]`` host pixels to the worker's encoder; its latent comes
        through the reply."""
        return self._runtime.send("encode", {}, [pixels])


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
        self._worker_keeps_cache = False

    def send_window(self, latent: torch.Tensor, seed: int, free_bytes: int) -> PendingReply:
        """Send one window to the worker's decoder; its uint8 ``[F, 3, S, S]`` frames come through the reply.
        After a decode with a budget of ``GPU_WINDOWS_MIN_FREE_BYTES`` the worker keeps its allocator's cache."""
        self._worker_keeps_cache = free_bytes >= GPU_WINDOWS_MIN_FREE_BYTES
        return self._runtime.send("decode", {"seed": seed, "free_bytes": free_bytes}, [latent])

    def release_cache(self) -> None:
        """Have the worker hand the cache a roomy decode left back to the driver, so the next
        budget reading counts it as free (device-wide free VRAM counts it as used)."""
        if self._worker_keeps_cache:
            self._runtime.request("empty_cache", {}, [])
            self._worker_keeps_cache = False


class PendingReply:
    """A request the worker answers in request order (``answered`` once its result header is in, ``tensors`` once
    its tensors are too); a cancelled job raises ``Cancelled`` from either."""

    def __init__(self, runtime: ModelRuntime, reply) -> None:
        self._runtime = runtime
        self._reply = reply

    def answered(self) -> None:
        self._runtime._cancellable(self._reply.answered)

    def tensors(self) -> list[torch.Tensor]:
        return self._runtime._cancellable(self._reply.result)[1]


def worker_starts_with_job(models: frozenset[str], *, protected: bool, resuming: bool) -> bool:
    """Whether a job starts its LTX worker (license grant, transformer file, encoder) when the job
    starts, beside its setup and scan, instead of at the scan: only for protected
    weights (trial and plaintext run in this process), only when LTX is the job's only model
    (BasicVSR++ ranges run first, and the worker would hold GPU memory beside them), and not
    when resuming (saved work may need no encoder)."""
    return protected and models == {"ltx"} and not resuming


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
        self._transformer_opening = False
        self._decoder = None
        self._transformer_source: TransformerSource | None = None
        self._ahead: Future[None] | None = None

    def start_ahead(self) -> None:
        """Start the worker on a thread of its own: it takes its license grant, opens the
        transformer file and loads the encoder while this process does other work. Every later
        call waits for that start, and its failure raises there."""
        starting = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxWorkerStart")
        self._ahead = starting.submit(self._load_ahead)
        starting.shutdown(wait=False)

    def _load_ahead(self) -> None:
        self._prepare_transformer()
        self._load_encoder()

    def _wait_ahead(self) -> None:
        if self._ahead is not None:
            self._ahead.result()

    def request(self, op: str, params: dict, tensors: list[torch.Tensor], advance=None):
        client = self._worker_client()
        return self._cancellable(lambda: client.request(op, params, tensors, advance=advance))

    def send(self, op: str, params: dict, tensors: list[torch.Tensor]) -> PendingReply:
        """Send a request without waiting for its reply, so the next one can go out while this one's result still
        crosses the pipe."""
        client = self._worker_client()
        return PendingReply(self, self._cancellable(lambda: client.send(op, params, tensors)))

    def _worker_client(self):
        if self._client is None:
            from jasna.license_api import require_protection

            require_protection()
            from jasna.protection.ltx_worker import WorkerClient

            self._client = WorkerClient(
                {"transformer": str(self.files.transformer.resolve()), "vae": str(self.files.vae.resolve()),
                 "tuned_decoder": str(self.files.tuned_decoder.resolve()),
                 "encoder": str(self.files.video_encoder.resolve()), "device": str(self.device)}, self.cancel
            )
        self._client.cancel = self.cancel
        return self._client

    def _cancellable(self, call: Callable):
        from jasna.protection.ltx_worker import WorkerCancelled

        try:
            return call()
        except WorkerCancelled:
            from jasna.ltx.restore import Cancelled

            self._stop()
            raise Cancelled() from None

    def load_encoder(self):
        self._wait_ahead()
        return self._load_encoder()

    def _load_encoder(self):
        if self._encoder is None:
            if self.protected:
                self.request("load_encoder", {}, [])
                self._encoder = WorkerEncoder(self)
            else:
                self._encoder = load_video_encoder(self.files.vae, self.device, self.files.encoder)
        return self._encoder

    def unload_encoder(self) -> None:
        if self._encoder is not None:
            self._encoder = None
            if self.protected:
                self.request("unload_encoder", {}, [])
            else:
                torch.cuda.empty_cache()

    def prepare_transformer(self) -> None:
        """Open the transformer file and read its host-side part (in the worker when protected) while
        the GPU is still busy (scan, encode); nothing of the transformer reaches the GPU before
        ``load_transformer``. Locally it also loads the GPU libraries the later passes call first."""
        self._wait_ahead()
        self._prepare_transformer()

    def _prepare_transformer(self) -> None:
        if self._transformer is not None:
            return
        if self.protected:
            if not self._transformer_opening:
                self.request("prepare_transformer", {}, [])
                self._transformer_opening = True
        elif self._transformer_source is None:
            load_gpu_libraries_ahead(self.device, torch.bfloat16)
            self._transformer_source = TransformerSource(self.files.transformer)

    def load_transformer(self):
        self._wait_ahead()
        if self._transformer is None:
            if self.protected:
                info, _ = self.request("load_transformer", {"free_bytes": self._free_bytes_for_worker()}, [])
                self._transformer = WorkerTransformer(self, info)
                self._transformer_opening = False
            else:
                source, self._transformer_source = self._transformer_source, None
                self._transformer = LtxTransformer(source or TransformerSource(self.files.transformer), self.device,
                                                   device_free_bytes)
        return self._transformer

    def _free_bytes_for_worker(self) -> int:
        """Device-wide free VRAM once this process's allocator has handed back its cached blocks
        (an encode pass's window pixels): only this process could reuse them, not the worker."""
        torch.cuda.empty_cache()
        return device_free_bytes(self.device)

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
        self._wait_ahead()
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
        """Waits for a start begun by ``start_ahead`` first; its failure is only logged, since
        nothing needs that worker any more."""
        ahead, self._ahead = self._ahead, None
        if ahead is not None and (error := ahead.exception()) is not None:
            logger.debug("LTX worker started ahead of a job that ended first failed: %r", error)
        self._stop()

    def _stop(self) -> None:
        if self.protected:
            self._encoder = self._transformer = self._decoder = None
            self._transformer_opening = False
            client, self._client = self._client, None
            if client is not None:
                client.close()
        else:
            self.unload_encoder()
            self.unload_transformer()
            self.unload_decoder()
            if self._transformer_source is not None:
                self._transformer_source.close()
                self._transformer_source = None

    def peak_vram_bytes(self) -> tuple[int, int | None]:
        """This process's peak and the worker's (``None`` without one). They are kept apart:
        the two peaks need not happen at the same time, so their sum overstates the use."""
        parent_peak = torch.cuda.max_memory_allocated(self.device)
        if self.protected and self._client is not None:
            stats, _ = self.request("stats", {}, [])
            return parent_peak, stats["peak_vram_bytes"]
        return parent_peak, None

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


def operation_peak_vram_bytes(torch_peak: int, torch_end: int, native_before: int, native_after: int) -> int:
    """One operation's peak: PyTorch's allocator plus the transformer blocks the native store
    holds outside it. PyTorch's peak only coexists with the native bytes there at the start:
    loading probes VRAM and frees it before the store allocates the blocks."""
    return max(torch_peak + native_before, torch_end + native_after)


class WorkerBackend:
    """Only the authenticated child owns models; dispatch exposes whole inference operations."""

    def __init__(self, config: dict) -> None:
        self.device = torch.device(config["device"])
        self.files = LtxModelFiles(*(Path(config[key]) for key in ("transformer", "vae", "tuned_decoder")),
                                   None if config["encoder"] == config["vae"] else Path(config["encoder"]))
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
        native_before = self._native_vram_bytes()
        result = self._run(op, params, tensors, advance)
        if torch.cuda.is_initialized():
            self.peak_vram_bytes = max(self.peak_vram_bytes, operation_peak_vram_bytes(
                torch.cuda.max_memory_allocated(self.device), torch.cuda.memory_allocated(self.device),
                native_before, self._native_vram_bytes(),
            ))
        return result

    def _native_vram_bytes(self) -> int:
        return self.transformer.native_vram_bytes() if self.transformer is not None else 0

    def _run(self, op: str, params: dict, tensors: list[torch.Tensor], advance: Callable[[int], None]):
        if op == "load_encoder":
            if self.encoder is None:
                self.encoder = load_video_encoder(self.files.vae, self.device, self.files.encoder)
        elif op == "unload_encoder":
            self._unload_module("encoder")
        elif op == "encode":
            if self.encoder is None:
                raise RuntimeError("LTX encoder is not loaded")
            pixels = tensors[0].to(self.device).float().div_(127.5).sub_(1.0).to(half_dtype(self.device))
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

            def completed(count: int) -> None:
                # The parent times progress on its CPU clock, outside this GPU process.
                torch.cuda.synchronize(self.device)
                advance(count)

            return {}, self.transformer.denoise_chain(tensors, params["seeds"], params["window_hop"], completed)
        elif op == "load_decoder":
            if self.decoder is None:
                self.decoder = load_video_decoder(self.files.vae, self.files.tuned_decoder, self.device)
        elif op == "unload_decoder":
            self._unload_module("decoder")
        elif op == "decode":
            if self.decoder is None:
                raise RuntimeError("LTX decoder is not loaded")
            frames = self._decode_to_cpu(tensors[0], params["seed"], params["free_bytes"])
            if params["free_bytes"] < GPU_WINDOWS_MIN_FREE_BYTES:
                # the parent's next budget reading counts this cache as used
                torch.cuda.empty_cache()
            return {}, [frames]
        elif op == "empty_cache":
            torch.cuda.empty_cache()
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
