"""Parent-owned checkpoints for offline LTX jobs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from jasna import __version__
from jasna.segments import SegmentRange, SegmentRestoration, job_restoration, normalize_segments, resolve_restorations
from jasna.session_config import SessionConfig

log = logging.getLogger(__name__)
FORMAT_VERSION = 3


class ResumeMismatchError(ValueError):
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        super().__init__(f"Saved LTX progress no longer matches the input, settings or app version: {directory}")

    def discard(self) -> None:
        shutil.rmtree(self.directory)


@contextmanager
def atomic_write(path: Path, mode: str):
    partial = path.with_name(path.name + ".part")
    with partial.open(mode) as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    commit_file(partial, path)


def commit_file(partial: Path, path: Path) -> None:
    partial.replace(path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json(path: Path, data) -> None:
    with atomic_write(path, "w") as stream:
        json.dump(data, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def resume_directory(config: SessionConfig, output: Path) -> Path:
    """Next to the output, or in the working directory with a short hash of the output path,
    since outputs of the same name in different folders share it."""
    if config.working_dir is None:
        return output.parent / f".{output.stem}.jasna-resume"
    output_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:8]
    return config.working_dir / f".{output.stem}.{output_key}.jasna-resume"


def _has_render_work(render: Path) -> bool:
    return any(any(render.glob(pattern)) for pattern in ("ltx/*.plans.npz", "ltx/*.pt", "segments/*-raw.nut"))


def has_saved_work(root: Path) -> bool:
    """Whether the checkpoint ``root`` holds finished work: an auto split, track plans,
    latents or fragments. A checkpoint without any is replaced without asking."""
    return (root / "split.json").is_file() or _has_render_work(root / "render")


def resume_directories(root: Path, output: Path) -> list[Path]:
    """Checkpoint folders in ``root`` that may belong to ``output`` or one of its auto-renamed names."""
    name = re.compile(rf"\.{re.escape(output.stem)}(?: \(\d+\))?(?:\.[0-9a-f]{{8}})?\.jasna-resume")
    candidates = [path for path in root.glob(".*.jasna-resume") if name.fullmatch(path.name)]
    return sorted(candidates, key=lambda path: (len(path.name), path.name))


def _model_identity(path: Path) -> list:
    resolved = str(path.resolve())
    try:
        stat = path.stat()
    except FileNotFoundError:
        return [resolved, None]
    return [resolved, stat.st_size, stat.st_mtime_ns]


LATENT_SETTINGS = (
    "device", "fp16", "batch_size", "detection_model_name", "detection_model_path", "detection_score_threshold",
    "restoration_model_name", "ltx_large_canvas", "ltx_window_hop", "ltx_seed", "ltx_fast", "ltx_model",
    "ltx_trial", "ltx_auto_split_percent", "vr_mode", "vr_projection", "retarget_high_fps",
)


def _digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def job_fingerprints(config: SessionConfig, source: Path, output: Path, metadata, segments) -> tuple[str, str]:
    """``(job, latents)``: ``job`` covers everything the output depends on; ``latents`` only
    what the saved auto split, track plans and latents depend on, so a change to the rest
    (codec, quality, LUT, sharpening, BasicVSR++ settings) keeps them."""
    from jasna.ltx.model_files import bundle_paths
    from jasna.session_factory import restoration_model_path_for

    stat = source.stat()
    settings = asdict(config)
    for key in ("disable_progress", "working_dir"):
        del settings[key]
    ranges = resolve_restorations(
        normalize_segments(tuple(segments or ())), job_restoration(config.restoration_model_name, config.ltx_seed)
    )
    models = {segment.restoration.model for segment in ranges} if ranges else {config.restoration_model_name}
    latent_models = {"detection": _model_identity(config.detection_model_path)}
    if "ltx" in models and not config.ltx_trial:
        latent_models["ltx"] = [
            _model_identity(path) for path in bundle_paths(
                restoration_model_path_for(config, "ltx"), config.ltx_model, fast=config.ltx_fast
            )
        ]
    latents = _digest({
        "format": FORMAT_VERSION,
        "version": __version__,
        "input": [str(source.resolve()), stat.st_size, stat.st_mtime_ns],
        "video": [metadata.num_frames, str(metadata.time_base), metadata.duration],
        "settings": {key: settings.pop(key) for key in LATENT_SETTINGS},
        "segments": [asdict(segment) for segment in ranges],
        "models": latent_models,
    })
    job = _digest({
        "latents": latents,
        "output": str(output.resolve()),
        "settings": settings,
        "basicvsrpp": (
            _model_identity(restoration_model_path_for(config, "basicvsrpp")) if "basicvsrpp" in models else None
        ),
        "lut": hashlib.sha256(Path(config.lut_path).read_bytes()).hexdigest() if config.lut_path else None,
    })
    return job, latents


def segments_from_json(data) -> tuple[SegmentRange, ...]:
    return normalize_segments(tuple(
        SegmentRange(item["start"], item["end"], SegmentRestoration(**item["restoration"])) for item in data
    ))


class ResumeJob:
    def __init__(self, config: SessionConfig, source: Path, output: Path, metadata, segments) -> None:
        self.root = resume_directory(config, output)
        self.directory = self.root / "render"
        self.fingerprint, self.latents = job_fingerprints(config, source, output, metadata, segments)
        self.output = output.resolve()
        self.source = source.resolve()

    def matches(self) -> bool:
        """Whether the saved LTX work can continue this job."""
        manifest = read_json(self.directory / "job.json")
        return manifest is not None and manifest.get("latents") == self.latents

    def prepare(self) -> None:
        manifest = read_json(self.directory / "job.json")
        if manifest is not None and manifest["fingerprint"] == self.fingerprint:
            log.info("Resuming saved LTX progress for %s", self.output.name)
            return
        if self.matches():
            log.info("Output settings changed for %s; reusing saved LTX work and redoing the rest", self.output.name)
            if (self.directory / "segments").exists():
                shutil.rmtree(self.directory / "segments")
        elif _has_render_work(self.directory):
            raise ResumeMismatchError(self.root)
        elif self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "job.json", {
            "fingerprint": self.fingerprint, "latents": self.latents, "input": str(self.source), "output": str(self.output),
        })

    def finish(self) -> None:
        shutil.rmtree(self.root)


def find_resume_output(config: SessionConfig, source: Path, output: Path, segments) -> Path | None:
    """Resume wins over GUI file conflicts (idea from #415), including an earlier auto-rename."""
    root = config.working_dir or output.parent
    candidates = resume_directories(root, output)
    if not candidates:
        return None
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import canonical_codec
    from jasna.session_factory import segment_job_config

    metadata = get_video_meta_data(str(source))
    for directory in candidates:
        manifest = read_json(directory / "render" / "job.json")
        if manifest is None:
            continue
        saved_output = Path(manifest["output"])
        if (
            saved_output.parent != output.parent.resolve() or saved_output.suffix.lower() != output.suffix.lower()
            or resume_directory(config, saved_output).resolve() != directory.resolve()
        ):
            continue
        effective_config, effective_segments = config, segments
        if not segments and config.ltx_auto_split_percent is not None and config.restoration_model_name == "ltx":
            split = read_json(directory / "split.json")
            key = job_fingerprints(config, source, saved_output, metadata, ())[1]
            if split is None or split["fingerprint"] != key:
                continue
            effective_segments = segments_from_json(split["segments"])
        if effective_segments:
            effective_config = segment_job_config(config, canonical_codec(metadata.codec_name), source)
        job = ResumeJob(effective_config, source, saved_output, metadata, effective_segments)
        if job.matches():
            return saved_output
    return None
