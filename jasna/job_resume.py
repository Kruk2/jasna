"""Parent-owned checkpoints for offline LTX jobs: their storage and fingerprints. Which job a
checkpoint belongs to is derived by ``jasna.session_factory``."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

from jasna import __version__
from jasna.os_utils import atomic_write
from jasna.segments import (
    SegmentRange,
    SegmentRestoration,
    job_models,
    job_restoration,
    normalize_segments,
    resolve_restorations,
)
from jasna.session_config import SessionConfig

if TYPE_CHECKING:
    from jasna.media.splice import SplicePlan

log = logging.getLogger(__name__)
FORMAT_VERSION = 3
T = TypeVar("T")


class ResumeMismatchError(ValueError):
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        super().__init__(f"Saved LTX progress no longer matches the input, settings or app version: {directory}")

    def discard(self) -> None:
        shutil.rmtree(self.directory)


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


def job_fingerprints(
    config: SessionConfig, source: Path, output: Path, metadata, splice_plan: SplicePlan | None
) -> tuple[str, str]:
    """``(job, latents)``: ``job`` covers everything the output depends on; ``latents`` only
    what the saved auto split, track plans and latents depend on, so a change to the rest
    (codec, quality, LUT, sharpening, BasicVSR++ settings) keeps them."""
    from jasna.ltx.model_files import bundle_paths
    from jasna.ltx.plan import track_max_gap

    stat = source.stat()
    settings = asdict(config)
    for key in ("disable_progress", "working_dir"):
        del settings[key]
    ranges = resolve_restorations(
        splice_plan.segments if splice_plan is not None else (),
        job_restoration(config.restoration_model_name, config.ltx_seed),
    )
    models = job_models(config.restoration_model_name, ranges)
    latent_models = {"detection": _model_identity(config.detection_model_path)}
    if "ltx" in models and not config.ltx_trial:
        latent_models["ltx"] = [
            _model_identity(path) for path in bundle_paths(
                config.model_path("ltx"), config.ltx_model, fast=config.ltx_fast
            )
        ]
    latents = _digest({
        "format": FORMAT_VERSION,
        "version": __version__,
        "input": [str(source.resolve()), stat.st_size, stat.st_mtime_ns],
        "video": [metadata.num_frames, str(metadata.time_base), metadata.duration],
        "settings": {key: settings.pop(key) for key in LATENT_SETTINGS},
        # LTX plans only change when --max-detection-gap exceeds MAX_GAP, so changes within
        # 0..MAX_GAP reuse saved latents; BasicVSR++ gap fill is still tracked in ``job``.
        "ltx_max_gap": track_max_gap(config.max_detection_gap),
        "segments": [asdict(segment) for segment in ranges],
        "models": latent_models,
    })
    job = _digest({
        "latents": latents,
        "output": str(output.resolve()),
        "settings": settings,
        "basicvsrpp": (
            _model_identity(config.model_path("basicvsrpp")) if "basicvsrpp" in models else None
        ),
        "lut": hashlib.sha256(Path(config.lut_path).read_bytes()).hexdigest() if config.lut_path else None,
        "plan": _digest(asdict(splice_plan)) if splice_plan is not None else None,
    })
    return job, latents


def segments_from_json(data) -> tuple[SegmentRange, ...]:
    return normalize_segments(tuple(
        SegmentRange(item["start"], item["end"], SegmentRestoration(**item["restoration"])) for item in data
    ))


class ResumeJob:
    """The checkpoint of one output: the restored work of a job, and the LTX auto split it
    was planned from. ``splice_plan`` None is a job that restores the whole video."""

    def __init__(self, config: SessionConfig, source: Path, output: Path, metadata, splice_plan: SplicePlan | None):
        self.root = resume_directory(config, output)
        self.directory = self.root / "render"
        self.fingerprint, self.latents = job_fingerprints(config, source, output, metadata, splice_plan)
        self.output = output.resolve()
        self.source = source.resolve()

    def matches(self) -> bool:
        """Whether the saved LTX work can continue this job."""
        manifest = read_json(self.directory / "job.json")
        return manifest is not None and manifest["latents"] == self.latents

    def prepare(self) -> None:
        manifest = read_json(self.directory / "job.json")
        if manifest is not None and manifest["fingerprint"] == self.fingerprint:
            log.info("Resuming saved LTX progress for %s", self.output.name)
            return
        if self.matches():
            log.info("Output settings changed for %s; reusing saved LTX work and redoing the rest", self.output.name)
            if (self.directory / "segments").exists():
                shutil.rmtree(self.directory / "segments")
        else:
            self._replace_unless(_has_render_work(self.directory))
            if self.directory.exists():
                shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "job.json", {
            "fingerprint": self.fingerprint, "latents": self.latents, "input": str(self.source), "output": str(self.output),
        })

    def saved_split(self) -> tuple[SegmentRange, ...] | None:
        """The LTX auto split saved for this whole-video job's settings; None when it still
        has to be scanned."""
        split = read_json(self.root / "split.json")
        if split is not None and split["fingerprint"] == self.latents:
            return segments_from_json(split["segments"])
        self._replace_unless(has_saved_work(self.root))
        return None

    def save_split(self, segments: tuple[SegmentRange, ...]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "split.json", {
            "fingerprint": self.latents, "segments": [asdict(segment) for segment in segments],
        })

    def _replace_unless(self, work_saved: bool) -> None:
        """Saved data for other settings is replaced, except saved work: only the user may
        discard that."""
        if work_saved:
            raise ResumeMismatchError(self.root)

    def finish(self) -> None:
        shutil.rmtree(self.root)


def saved_outputs(config: SessionConfig, output: Path) -> list[Path]:
    """Outputs whose saved work may continue a job writing ``output``: ``output`` itself or
    one of its auto-renamed names, in the same folder and format."""
    found = []
    for directory in resume_directories(config.working_dir or output.parent, output):
        manifest = read_json(directory / "render" / "job.json")
        if manifest is None:
            continue
        saved = Path(manifest["output"])
        if (
            saved.parent == output.parent.resolve() and saved.suffix.lower() == output.suffix.lower()
            and resume_directory(config, saved).resolve() == directory.resolve()
        ):
            found.append(saved)
    return found


def run_with_resume_prompt(run: Callable[[], T], ask: Callable[[ResumeMismatchError], bool]) -> T:
    """``run()``; when its saved progress no longer matches, ``ask`` whether to discard it and
    start over. A refusal keeps the progress and raises the mismatch."""
    while True:
        try:
            return run()
        except ResumeMismatchError as error:
            if not ask(error):
                raise
            error.discard()
