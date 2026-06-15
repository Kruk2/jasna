from __future__ import annotations

import json
import logging
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

from jasna._frozen import is_frozen
from jasna.engine_paths import SD15_CKPT_ENC_PATH, SD15_CKPT_PATH

logger = logging.getLogger(__name__)

DEFAULT_FREEU = {"s1": 0.9, "s2": 0.2, "b1": 1.2, "b2": 1.4}


def use_plaintext_sd15(model_dir: Path) -> bool:
    return (model_dir / SD15_CKPT_PATH.name).exists() and not is_frozen()


def _read_checkpoint(model_dir: Path) -> dict:
    """Return the (pruned) fine-tune checkpoint dict.

    Dev / source checkouts with a plaintext ``sd15-200000.ckpt`` load it
    directly (no license). Otherwise the ``.enc`` is decrypted with the
    license-derived master key (shared with unet-4x, distinct subkey).
    """
    if use_plaintext_sd15(model_dir):
        return torch.load(str(model_dir / SD15_CKPT_PATH.name), map_location="cpu", weights_only=True)

    from jasna.protection import protected_model

    enc_path = model_dir / SD15_CKPT_ENC_PATH.name
    with tempfile.NamedTemporaryFile(suffix=".cpkt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        protected_model.decrypt_model_to_path(Sd15InpaintRestorer.MODEL_ID, enc_path, tmp_path)
        return torch.load(str(tmp_path), map_location="cpu", weights_only=True)
    finally:
        tmp_path.unlink(missing_ok=True)


class Sd15InpaintRestorer:
    # AES-GCM subkey id — must match the id used by keytool encrypt-model that
    # produced sd15-200000.ckpt.enc (verified against the shipped file: "sd-15-jav").
    MODEL_ID = "sd-15-jav"
    LATENT_SCALE = 0.18215

    def __init__(self, model_dir: Path, device: torch.device, fp16: bool) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self._dtype = torch.float16 if self.fp16 else torch.float32

        unet_config = json.loads((self.model_dir / "unet" / "config.json").read_text())
        unet = UNet2DConditionModel.from_config(unet_config)

        ckpt = _read_checkpoint(self.model_dir)
        state_dict = ckpt.get("state_dict", ckpt)
        unet_state = {}
        for k, v in state_dict.items():
            if not k.startswith("unet."):
                continue
            unet_state[k.removeprefix("unet.").removeprefix("_orig_mod.")] = v
        unet.load_state_dict(unet_state)

        if "_ema_flat" in ckpt:
            flat = ckpt["_ema_flat"]
            offset = 0
            for p in (p for p in unet.parameters() if p.requires_grad):
                p.data.copy_(flat[offset:offset + p.numel()].reshape(p.shape))
                offset += p.numel()
            logger.info("Loaded EMA weights into SD15 UNet")

        self.unet = unet.to(self.device, dtype=self._dtype).eval()
        self.vae = AutoencoderKL.from_pretrained(self.model_dir / "vae", local_files_only=True).to(self.device).eval()

        ddpm_config = DDPMScheduler.from_pretrained(
            self.model_dir / "scheduler",
            prediction_type="v_prediction",
            rescale_betas_zero_snr=True,
        ).config
        self.scheduler = DPMSolverMultistepScheduler.from_config(
            ddpm_config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
        )

        self.null_embedding = torch.load(self.model_dir / "null_embedding.pt").to(self.device)
        logger.info("Sd15InpaintRestorer loaded from %s (fp16=%s)", self.model_dir, self.fp16)

    @torch.no_grad()
    def restore_crop(
        self,
        mosaic_01: torch.Tensor,
        mask_01: torch.Tensor,
        *,
        steps: int,
        strength: float,
        seed: int,
        freeu: dict | None = None,
    ) -> torch.Tensor:
        """SDEdit-inpaint a single 512² crop.

        ``mosaic_01``: ``(1, 3, 512, 512)`` in [0, 1]; ``mask_01``: ``(1, 1, 512,
        512)`` in [0, 1]. Returns a ``(3, 512, 512)`` uint8 RGB tensor.
        """
        if freeu:
            self.unet.enable_freeu(**freeu)
        else:
            self.unet.disable_freeu()

        # VAE stays fp32 (fp16 SD1.5 VAE is unstable); only the UNet runs fp16
        # under autocast. So feed the VAE fp32 and let autocast handle the UNet.
        mosaic_t = mosaic_01.to(self.device, dtype=torch.float32)
        mask_t = mask_01.to(self.device, dtype=torch.float32)

        mosaic_latents = self.vae.encode(mosaic_t * 2.0 - 1.0).latent_dist.mode() * self.LATENT_SCALE
        mask_latents = F.interpolate(mask_t, size=mosaic_latents.shape[2:], mode="nearest")
        encoder_hidden_states = self.null_embedding.expand(mosaic_t.shape[0], -1, -1)
        init_latents = mosaic_latents

        self.scheduler.set_timesteps(int(steps), device=mosaic_latents.device)
        start_step = int(len(self.scheduler.timesteps) * (1 - strength))
        t_start = self.scheduler.timesteps[start_step]

        torch.manual_seed(int(seed))
        noise = torch.randn_like(init_latents)
        latents = self.scheduler.add_noise(init_latents, noise, t_start.unsqueeze(0))

        ctx = torch.autocast("cuda", dtype=torch.float16) if (self.fp16 and self.device.type == "cuda") else nullcontext()
        with ctx:
            for t in self.scheduler.timesteps[start_step:]:
                unet_input = torch.cat([latents, mask_latents, mosaic_latents], dim=1)
                noise_pred = self.unet(unet_input, t.expand(latents.shape[0]), encoder_hidden_states).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        images = self.vae.decode(latents.float() / self.LATENT_SCALE).sample
        out01 = (images * 0.5 + 0.5).clamp(0, 1)
        return (out01[0] * 255.0).round().to(torch.uint8)

    def close(self) -> None:
        self.unet = None
        self.vae = None
        self.scheduler = None
        self.null_embedding = None
