from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn import functional as F

from jasna._frozen import is_frozen
from jasna.engine_paths import (
    UNET4X_ONNX_ENC_PATH,
    UNET4X_ONNX_PATH,
    get_unet4x_encrypted_engine_path,
    get_unet4x_engine_path,
)
from jasna.protection import ProtectionError
from jasna.trt.trt_runner import TrtRunner

logger = logging.getLogger(__name__)

UNET4X_INPUT_SIZE = 256
UNET4X_OUTPUT_SIZE = 1024
UNET4X_MODEL_ID = "unet-4x"

UNET4X_INPUT_SHAPES = {
    "lr_prev": (1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3),
    "lr_curr": (1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3),
    "hr_prev": (1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3),
}


def use_plaintext_unet4x() -> bool:
    return UNET4X_ONNX_PATH.exists() and not is_frozen()


def compile_unet4x_engine(
    device: torch.device,
    fp16: bool = True,
) -> Path:
    if use_plaintext_unet4x():
        from jasna.trt import compile_onnx_to_tensorrt_engine
        return compile_onnx_to_tensorrt_engine(
            UNET4X_ONNX_PATH, device, batch_size=None, fp16=bool(fp16), workspace_gb=20,
        )

    from jasna.protection import protected_model
    from jasna.trt import compile_onnx_bytes_to_encrypted_engine

    engine_path = get_unet4x_encrypted_engine_path(fp16=bool(fp16))
    if engine_path.exists():
        try:
            protected_model.decrypt_engine_bytes(UNET4X_MODEL_ID, engine_path.read_bytes())
        except ProtectionError:
            logger.warning("Discarding stale encrypted Unet4x engine: %s", engine_path)
            engine_path.unlink(missing_ok=True)
        else:
            return engine_path

    onnx_bytes = protected_model.decrypt_model_bytes(UNET4X_MODEL_ID, UNET4X_ONNX_ENC_PATH)
    try:
        return compile_onnx_bytes_to_encrypted_engine(
            onnx_bytes,
            UNET4X_MODEL_ID,
            engine_path,
            device,
            batch_size=None,
            fp16=bool(fp16),
            workspace_gb=20,
        )
    finally:
        del onnx_bytes


def encrypted_unet4x_engine_is_usable(fp16: bool = True) -> bool:
    if use_plaintext_unet4x():
        return get_unet4x_engine_path(UNET4X_ONNX_PATH, fp16=bool(fp16)).exists()
    engine_path = get_unet4x_encrypted_engine_path(fp16=bool(fp16))
    if not engine_path.exists():
        return False

    from jasna.protection import protected_model
    try:
        protected_model.decrypt_engine_bytes(UNET4X_MODEL_ID, engine_path.read_bytes())
    except ProtectionError:
        return False
    return True


class Unet4xSecondaryRestorer:
    name = "unet-4x"
    num_workers = 1
    preferred_queue_size = 2
    prefers_cpu_input = False

    def __init__(self, *, device: torch.device, fp16: bool = True) -> None:
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self._dtype = torch.float16 if self.fp16 else torch.float32

        if use_plaintext_unet4x():
            self.engine_path = get_unet4x_engine_path(UNET4X_ONNX_PATH, fp16=self.fp16)
            if not self.engine_path.exists():
                raise FileNotFoundError(
                    f"Unet4x engine not found: {self.engine_path}. "
                    "Run engine compilation first via ensure_engines_compiled()."
                )
            self.runner = TrtRunner(
                self.engine_path, input_shapes=UNET4X_INPUT_SHAPES, device=self.device,
            )
        else:
            self.engine_path = get_unet4x_encrypted_engine_path(fp16=self.fp16)
            if not self.engine_path.exists():
                raise FileNotFoundError(
                    f"Unet4x engine not found: {self.engine_path}. "
                    "Run engine compilation first via ensure_engines_compiled()."
                )
            from jasna.protection import protected_model
            try:
                engine_bytes = protected_model.decrypt_engine_bytes(UNET4X_MODEL_ID, self.engine_path.read_bytes())
            except ProtectionError:
                logger.warning("Encrypted Unet4x engine is stale; recompiling: %s", self.engine_path)
                self.engine_path.unlink(missing_ok=True)
                self.engine_path = compile_unet4x_engine(self.device, fp16=self.fp16)
                engine_bytes = protected_model.decrypt_engine_bytes(UNET4X_MODEL_ID, self.engine_path.read_bytes())
            try:
                self.runner = TrtRunner.from_engine_bytes(
                    engine_bytes, input_shapes=UNET4X_INPUT_SHAPES, device=self.device,
                    source=str(self.engine_path),
                )
            finally:
                del engine_bytes
        logger.info(
            "Unet4xSecondaryRestorer loaded: %s (%dx%d -> %dx%d)",
            self.engine_path,
            UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE,
            UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE,
        )

        self._stream = torch.cuda.current_stream(self.device)

        self._g_lr_prev = torch.zeros(1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3, dtype=self._dtype, device=self.device)
        self._g_lr_curr = torch.zeros(1, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE, 3, dtype=self._dtype, device=self.device)
        self._g_hr_prev = torch.zeros(1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=self._dtype, device=self.device)

        self.runner.context.set_tensor_address("lr_prev", self._g_lr_prev.data_ptr())
        self.runner.context.set_tensor_address("lr_curr", self._g_lr_curr.data_ptr())
        self.runner.context.set_tensor_address("hr_prev", self._g_hr_prev.data_ptr())

    def _to_nhwc(self, frames_nchw: torch.Tensor) -> torch.Tensor:
        return frames_nchw.permute(0, 2, 3, 1).contiguous()

    def _infer(self) -> None:
        self.runner.context.execute_async_v3(self._stream.cuda_stream)

    def _advance_temporal(self) -> None:
        self._infer()
        self._g_hr_prev.copy_(self.runner.outputs["out"])
        self._g_lr_prev.copy_(self._g_lr_curr)

    def _bootstrap(self, first_frame_nhwc: torch.Tensor) -> None:
        self._g_lr_prev.copy_(first_frame_nhwc.unsqueeze(0))
        self._g_lr_curr.copy_(first_frame_nhwc.unsqueeze(0))
        hr_init_nchw = F.interpolate(
            first_frame_nhwc.unsqueeze(0).permute(0, 3, 1, 2),
            size=(UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        self._g_hr_prev.copy_(hr_init_nchw.permute(0, 2, 3, 1))
        self._advance_temporal()

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        T = int(frames_256.shape[0])
        if T == 0:
            return []

        ks = max(0, int(keep_start))
        ke = min(T, int(keep_end))
        if ks >= ke:
            return []

        frames = frames_256.to(device=self.device, dtype=self._dtype)
        frames_nhwc = self._to_nhwc(frames)

        if ks > 0:
            ctx = ks - 1
            self._bootstrap(frames_nhwc[ctx])
            self._g_lr_curr.copy_(frames_nhwc[ctx : ctx + 1])
            self._advance_temporal()
        else:
            self._bootstrap(frames_nhwc[0])

        kept_count = ke - ks
        result = torch.empty(kept_count, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=self._dtype, device=self.device)

        for i in range(ks, ke):
            self._g_lr_curr.copy_(frames_nhwc[i : i + 1])
            if i == 0:
                self._g_lr_prev.copy_(frames_nhwc[0:1])

            self._advance_temporal()
            result[i - ks].copy_(self._g_hr_prev[0])

        kept_nchw = result.permute(0, 3, 1, 2).clamp_(0, 1).mul_(255.0).to(dtype=torch.uint8)
        return list(kept_nchw.unbind(0))

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None
