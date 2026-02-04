from __future__ import annotations

import torch

from jasna.tensor_utils import pad_batch_with_last


class Swin2srSecondaryRestorer:
    name = "swin2sr"

    def __init__(self, *, device: torch.device, fp16: bool, batch_size: int, use_tensorrt: bool) -> None:
        from pathlib import Path

        self.device = torch.device(device)
        self.dtype = torch.float16 if (bool(fp16) and self.device.type == "cuda") else torch.float32
        self.batch_size = int(batch_size)
        self.engine = None
        self.model = None

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
        del keep_start, keep_end

        t = int(frames_256.shape[0])
        if t == 0:
            return []

        if frames_256.dtype == torch.uint8:
            raise RuntimeError("Swin2SR secondary expects float frames in [0, 1], got uint8")

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

