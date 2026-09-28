"""Restore a whole video with the LTX model in four passes, one model on the GPU at a time.

1. scan: decode, detect mosaics and scene cuts, plan tracks, cameras and windows;
2. encode: crop every window's 121 frames onto its canvas and VAE-encode them;
3. denoise: run each track's windows through the transformer in lockstep;
4. compose: decode each window's latent when the video reaches it, composite, write.

The camera and fusion need a whole track before its first window can be denoised, so
the video is decoded three times (scan, encode, compose). Per-window latents (1-2 MB)
are kept in a temporary directory between passes, so memory does not grow with length.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from jasna.ltx.camera import WINDOW_FRAMES
from jasna.ltx.compose import Candidate, composite_frame, crop_to_canvas
from jasna.ltx.plan import Region, TrackPlan, Window, crossfade_weights, feather_pixels, plan_video
from jasna.ltx.regions import regions_per_frame
from jasna.ltx.sampler import LtxTransformer
from jasna.models.ltx_vae import decode_latent, load_video_decoder, load_video_encoder
from jasna.tracking.scene_detector import SceneCutDetector

logger = logging.getLogger(__name__)

FrameBatches = Iterator[tuple[torch.Tensor, list[int]]]
FrameSource = Callable[[], FrameBatches]


@dataclass(frozen=True)
class LtxModelFiles:
    transformer: Path
    vae: Path
    tuned_decoder: Path

    @classmethod
    def from_dir(cls, directory: Path) -> LtxModelFiles:
        files = cls(
            transformer=directory / "transformer.safetensors",
            vae=directory / "vae.safetensors",
            tuned_decoder=directory / "vae-decoder.safetensors",
        )
        missing = [str(path) for path in (files.transformer, files.vae, files.tuned_decoder) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"LTX model files missing: {', '.join(missing)}")
        return files


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class Progress:
    """One console bar per pass; ``frames`` sizes the frame-based passes."""

    frames: int
    disable: bool

    def bar(self, name: str, total: int, unit: str = "frame") -> tqdm:
        return tqdm(total=total, desc=f"LTX {name}", unit=unit, dynamic_ncols=True, disable=self.disable)


class _LatentStore:
    def __init__(self, directory: Path, kind: str) -> None:
        self._directory = directory
        self._kind = kind

    def _path(self, index: int) -> Path:
        return self._directory / f"{index:06d}.{self._kind}.pt"

    def put(self, index: int, latent: torch.Tensor) -> None:
        torch.save(latent.to("cpu", torch.bfloat16).contiguous(), self._path(index))

    def get(self, index: int) -> torch.Tensor:
        return torch.load(self._path(index), weights_only=True)

    def delete(self, index: int) -> None:
        self._path(index).unlink()


def _check(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def scan(
    frames: FrameSource,
    detector,
    *,
    frame_h: int,
    frame_w: int,
    large_canvas: bool,
    progress: Progress,
    cancel: threading.Event,
) -> list[TrackPlan]:
    regions: list[list[Region]] = []
    cuts: set[int] = set()
    scene = SceneCutDetector()
    with progress.bar("scan", progress.frames) as bar:
        for batch, _pts in frames():
            _check(cancel)
            cuts.update(len(regions) + offset for offset in scene.find_cuts(batch))
            regions.extend(regions_per_frame(detector(batch, target_hw=(frame_h, frame_w)), frame_h, frame_w))
            bar.update(len(batch))
    plans = plan_video(regions, cuts, frame_w=frame_w, frame_h=frame_h, large_canvas=large_canvas)
    logger.info(
        "LTX plan: %d frames, %d scene cuts, %d tracks, %d windows",
        len(regions),
        len(cuts),
        len(plans),
        sum(len(plan.windows) for plan in plans),
    )
    return plans


def _vae_pixels(canvases: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    stacked = torch.stack(canvases + [canvases[-1]] * (WINDOW_FRAMES - len(canvases)), dim=1)
    return stacked.to(device).float().div_(127.5).sub_(1.0).to(torch.bfloat16).unsqueeze(0)


def encode_references(
    frames: FrameSource,
    plans: list[TrackPlan],
    encoder,
    store: _LatentStore,
    *,
    device: torch.device,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    windows = sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start)
    pending = iter(windows)
    upcoming = next(pending, None)
    active: dict[int, tuple[Window, list[torch.Tensor]]] = {}
    frame_idx = 0
    bar = progress.bar("encode", progress.frames)
    for batch, _pts in frames():
        _check(cancel)
        bar.update(len(batch))
        for frame in batch:
            while upcoming is not None and upcoming.start == frame_idx:
                active[upcoming.index] = (upcoming, [])
                upcoming = next(pending, None)
            for index, (window, canvases) in list(active.items()):
                canvases.append(crop_to_canvas(frame, window.crops[frame_idx - window.start]).cpu())
                if len(canvases) == window.real_frames:
                    with torch.inference_mode():
                        store.put(index, encoder(_vae_pixels(canvases, device)))
                    del active[index]
            frame_idx += 1
    bar.close()
    if active or upcoming is not None:
        raise RuntimeError("the video ended before every planned window was filled")


def denoise(
    plans: list[TrackPlan],
    transformer: LtxTransformer,
    references: _LatentStore,
    finals: _LatentStore,
    *,
    seed: int,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    bar = progress.bar("denoise", sum(len(plan.windows) for plan in plans), unit="window")
    for plan in plans:
        _check(cancel)
        latents = transformer.denoise_chain(
            [references.get(w.index) for w in plan.windows], [seed + w.index for w in plan.windows]
        )
        for window, latent in zip(plan.windows, latents):
            finals.put(window.index, latent)
            references.delete(window.index)
        bar.update(len(plan.windows))
    bar.close()


def _decode_window(decoder, latent: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    """Decoded window as uint8 ``[F, 3, S, S]`` on ``device``."""
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        video = decode_latent(decoder, latent.to(device), free_bytes=free, generator=generator)
        video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)[0].float()
    return video.mul_(255.0).round_().to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def compose(
    frames: FrameSource,
    plans: list[TrackPlan],
    decoder,
    finals: _LatentStore,
    write: Callable[[torch.Tensor, int], None],
    *,
    frame_h: int,
    seed: int,
    device: torch.device,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    feather = feather_pixels(frame_h)
    weights: dict[int, torch.Tensor] = {}
    tracks: dict[int, TrackPlan] = {}
    for plan in plans:
        for window, weight in zip(plan.windows, crossfade_weights(plan.windows)):
            weights[window.index] = weight
            tracks[window.index] = plan
    pending = iter(sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start))
    upcoming = next(pending, None)
    covering: list[Window] = []
    decoded: dict[int, torch.Tensor] = {}
    frame_idx = 0
    bar = progress.bar("compose", progress.frames)
    for batch, pts in frames():
        _check(cancel)
        bar.update(len(batch))
        for frame, frame_pts in zip(batch, pts):
            while upcoming is not None and upcoming.start == frame_idx:
                covering.append(upcoming)
                upcoming = next(pending, None)
            for window in [w for w in covering if w.stop <= frame_idx]:
                covering.remove(window)
                decoded.pop(window.index, None)
            candidates = []
            for window in covering:
                if window.index not in decoded:
                    decoded[window.index] = _decode_window(decoder, finals.get(window.index), seed + window.index, device)
                local = frame_idx - window.start
                track = tracks[window.index]
                candidates.append(
                    Candidate(
                        canvas=decoded[window.index][local],
                        crop=window.crops[local],
                        weight=float(weights[window.index][local]),
                        polygons=track.polygons[frame_idx - track.start],
                    )
                )
            write(composite_frame(frame, candidates, feather=feather), frame_pts)
            frame_idx += 1
    bar.close()


def restore_video(
    frames: FrameSource,
    write: Callable[[torch.Tensor, int], None],
    *,
    detector,
    files: LtxModelFiles,
    frame_h: int,
    frame_w: int,
    large_canvas: bool,
    seed: int,
    device: torch.device,
    work_dir: Path,
    progress: Progress,
    cancel: threading.Event,
) -> None:
    """Restore every mosaic of the video ``frames`` yields and ``write`` each output frame
    with its pts. ``detector`` is only used by the scan pass; ``large_canvas`` allows 768 px
    windows for large mosaics; window ``i`` draws its noise from ``seed + i``."""
    plans = scan(frames, detector, frame_h=frame_h, frame_w=frame_w, large_canvas=large_canvas, progress=progress, cancel=cancel)
    with tempfile.TemporaryDirectory(dir=work_dir, prefix=".ltx-") as temp:
        references = _LatentStore(Path(temp), "reference")
        finals = _LatentStore(Path(temp), "final")
        if plans:
            encoder = load_video_encoder(files.vae, device)
            encode_references(frames, plans, encoder, references, device=device, progress=progress, cancel=cancel)
            del encoder
            torch.cuda.empty_cache()
            transformer = LtxTransformer(files.transformer, device)
            try:
                denoise(plans, transformer, references, finals, seed=seed, progress=progress, cancel=cancel)
            finally:
                transformer.close()
                del transformer
                torch.cuda.empty_cache()
        decoder = load_video_decoder(files.vae, files.tuned_decoder, device) if plans else None
        compose(
            frames,
            plans,
            decoder,
            finals,
            write,
            frame_h=frame_h,
            seed=seed,
            device=device,
            progress=progress,
            cancel=cancel,
        )
