"""2x frame generation: one frame made by NVIDIA RTX Video Frame Generation between each
pair of finished frames, right before the encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter


class FrameGenerationWriter:
    """Writes each frame and, before the next one, a frame generated at the midpoint. PTS
    are doubled so the midpoints stay whole; the encoder runs with ``frame_generation`` and
    half the source time base. The last frame is written twice, the copy half its frame
    interval later, so ``N`` frames become ``2N`` and the output keeps the source duration."""

    def __init__(self, writer: FrameWriter, generate: RtxFrameGeneration) -> None:
        """Owns ``generate`` and closes it with ``writer``."""
        self._writer = writer
        self._generate = generate
        self._previous: tuple[torch.Tensor, int] | None = None
        self._interval = 0

    def write(self, frame: torch.Tensor, pts: int) -> None:
        if self._previous is not None:
            previous, previous_pts = self._previous
            self._writer.write(previous, 2 * previous_pts)
            self._writer.write(self._generate(previous, frame), previous_pts + pts)
            self._interval = pts - previous_pts
        self._previous = (frame, pts)

    def after_write(self, frames_written: int, stop_event) -> None:
        pass

    def finish(self) -> None:
        if self._previous is not None:
            frame, pts = self._previous
            self._writer.write(frame, 2 * pts)
            if self._interval > 0:
                self._writer.write(frame, 2 * pts + self._interval)
            self._previous = None
        self._writer.finish()

    def close(self) -> None:
        self._previous = None
        try:
            self._writer.close()
        finally:
            self._generate.close()


class RtxFrameGeneration:
    """nvvfx ``VideoFrameGeneration`` at the midpoint of two uint8 ``[3, height, width]`` GPU
    frames. Its automatic shot-change detection stays on, so a hard cut is not blended across."""

    def __init__(self, device: torch.device, width: int, height: int) -> None:
        import jasna.restorer.rtx_superres_secondary_restorer  # noqa: F401  pins pip TensorRT before nvvfx loads
        from nvvfx import VideoFrameGeneration

        self._device = device
        self._effect = VideoFrameGeneration(device=device.index or 0, automatic_shot_change_detection_enabled=True)
        self._effect.input_width = width
        self._effect.input_height = height
        try:
            with torch.cuda.stream(torch.cuda.default_stream(device)):
                self._effect.load()
        except Exception as exc:
            self._effect.close()
            raise RuntimeError(
                f"RTX frame generation could not start ({exc}). It needs a recent NVIDIA driver: "
                "580.82 or newer on Linux."
            ) from exc

    def __call__(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream(self._device)
        inputs = [frame.float().div_(255.0).contiguous() for frame in (previous, current)]
        # nvvfx effects fail on any stream but the default one they load with (see ltx/super_res.py).
        default = torch.cuda.default_stream(self._device)
        default.wait_stream(stream)
        with torch.cuda.stream(default):
            image = torch.from_dlpack(self._effect.run_at_timestep(*inputs, 0.5).image)
            # The effect reuses one output buffer, so this copy must be made before the next run.
            generated = image.clamp(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        stream.wait_stream(default)
        generated.record_stream(stream)
        return generated

    def close(self) -> None:
        """Frees the effect's GPU memory; safe to call again."""
        self._effect.close()
