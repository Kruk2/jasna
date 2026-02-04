from __future__ import annotations

import torch


class Swin2srSecondaryRestorer:
    name = "swin2sr"

    def __init__(self, *, device: torch.device, fp16: bool, batch_size: int) -> None:
        from transformers import Swin2SRForImageSuperResolution

        self.device = torch.device(device)
        self.dtype = torch.float16 if (bool(fp16) and self.device.type == "cuda") else torch.float32
        self.batch_size = int(batch_size)
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
            if end - start < bs:
                pad = chunk[-1:].expand(bs - (end - start), -1, -1, -1)
                chunk = torch.cat([chunk, pad], dim=0)

            with torch.inference_mode():
                outputs = self.model(pixel_values=chunk)
                reconstruction = outputs.reconstruction

            reconstruction = reconstruction[: end - start].clamp(0, 1)
            out_u8 = reconstruction.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
            out.extend(list(torch.unbind(out_u8, 0)))

        return out

