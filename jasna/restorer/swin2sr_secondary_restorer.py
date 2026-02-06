from __future__ import annotations

import queue
from dataclasses import dataclass

import torch

from jasna.tensor_utils import pad_batch_with_last


@dataclass(frozen=True)
class _Completed:
    meta: object
    frame_u8: torch.Tensor

    def to_frame_u8(self, device: torch.device) -> torch.Tensor:
        if self.frame_u8.device == device:
            return self.frame_u8
        return self.frame_u8.to(device=device, non_blocking=False)


class Swin2srSecondaryRestorer:
    name = "swin2sr"

    def __init__(self, *, device: torch.device, fp16: bool, batch_size: int, use_tensorrt: bool) -> None:
        from pathlib import Path

        self.device = torch.device(device)
        self.dtype = torch.float16 if (bool(fp16) and self.device.type == "cuda") else torch.float32
        self.batch_size = int(batch_size)
        self.engine = None
        self.model = None
        self._completed: queue.SimpleQueue[_Completed] = queue.SimpleQueue()

        if bool(use_tensorrt) and self.device.type == "cuda" and self.dtype == torch.float16:
            import os

            from jasna.restorer.swin2sr_tensorrt_compilation import (
                compile_swin2sr_engine,
                get_compiled_swin2sr_engine_path,
                load_engine,
            )

            engine_dir = str(Path("model_weights"))
            engine_path = get_compiled_swin2sr_engine_path(engine_dir=engine_dir, batch_size=self.batch_size, fp16=True)
            if not os.path.isfile(engine_path):
                engine_path = compile_swin2sr_engine(
                    engine_dir=engine_dir,
                    batch_size=self.batch_size,
                    device=self.device,
                    fp16=True,
                )
            if os.path.isfile(engine_path):
                self.engine = load_engine(engine_path, self.device)

        if self.engine is None:
            from transformers import Swin2SRForImageSuperResolution

            self.model = Swin2SRForImageSuperResolution.from_pretrained(
                "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
                torch_dtype=self.dtype,
            ).to(device=self.device, dtype=self.dtype)
            self.model.eval()

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t = int(frames_256.shape[0])
        if t == 0:
            return []

        if frames_256.dtype == torch.uint8:
            raise RuntimeError("Swin2SR secondary expects float frames in [0, 1], got uint8")

        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []
        frames_256 = frames_256[ks:ke]
        t = int(frames_256.shape[0])

        out: list[torch.Tensor] = []
        bs = int(self.batch_size)
        for start in range(0, t, bs):
            end = min(start + bs, t)
            chunk = frames_256[start:end].to(device=self.device, dtype=self.dtype)
            n = int(end - start)
            chunk = pad_batch_with_last(chunk, batch_size=bs)

            with torch.inference_mode():
                if self.engine is not None:
                    reconstruction = self.engine(chunk)
                else:
                    outputs = self.model(pixel_values=chunk)
                    reconstruction = outputs.reconstruction

            reconstruction = reconstruction[:n].clamp(0, 1)
            out_u8 = reconstruction.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
            out.extend(list(torch.unbind(out_u8, 0)))

        return out

    def submit(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int, meta: list[object], track_id: int = 0) -> None:
        ks = max(0, int(keep_start))
        ke = min(int(frames_256.shape[0]), int(keep_end))
        expected = int(ke - ks)
        if expected != int(len(meta)):
            raise RuntimeError(f"Swin2SR submit meta length mismatch (expected {expected}, got {len(meta)})")
        out = self.restore(frames_256, keep_start=ks, keep_end=ke)
        for m, f in zip(meta, out):
            self._completed.put(_Completed(meta=m, frame_u8=f))

    def drain_completed(self, *, limit: int | None = None) -> list[_Completed]:
        out: list[_Completed] = []
        while True:
            if limit is not None and len(out) >= int(limit):
                break
            try:
                item = self._completed.get_nowait()
            except queue.Empty:
                break
            out.append(item)
        return out

    def flush(self, *, timeout_s: float = 300.0) -> None:
        del timeout_s

    def flush_track(self, track_id: int) -> None:
        pass

    def transfer_track(self, old_track_id: int, new_track_id: int) -> None:
        pass

