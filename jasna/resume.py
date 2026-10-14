"""Checkpoint/resume for smart-render jobs (断点续传).

A stopped job keeps its finished fragments in a checkpoint directory next to
the output video; the next run with the same input, segments and settings
reuses those fragments and renders only the missing ones. Once the whole job
finishes, the fragments are concatenated as usual and the checkpoint is
removed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_SUFFIX = ".jasna-checkpoint"
_STATE_NAME = "resume-state.json"


def resume_dir_for(output_video: Path) -> Path:
    output_video = Path(output_video)
    return output_video.parent / (output_video.stem + CHECKPOINT_SUFFIX)


def signature(payload) -> str:
    """Stable hash of everything that must match for a checkpoint to be reusable."""
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_state(resume_dir: Path) -> dict | None:
    state_file = Path(resume_dir) / _STATE_NAME
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != 1 or "signature" not in data:
        return None
    return data


def save_state(resume_dir: Path, state: dict) -> None:
    """Atomically persist the checkpoint (survives a crash mid-write)."""
    resume_dir = Path(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    state_file = resume_dir / _STATE_NAME
    tmp = resume_dir / (_STATE_NAME + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, state_file)


def completed_fragments(state: dict | None, work_dir: Path) -> set[int]:
    """Indices whose fragment files actually survived in ``work_dir``."""
    if not state:
        return set()
    done: set[int] = set()
    for entry in state.get("fragments", []):
        try:
            index = int(entry["index"])
        except (KeyError, TypeError, ValueError):
            continue
        fragment = Path(work_dir) / str(entry.get("file", ""))
        try:
            if fragment.is_file() and fragment.stat().st_size > 0:
                done.add(index)
        except OSError:
            continue
    return done


def clear_resume(resume_dir: Path) -> None:
    shutil.rmtree(Path(resume_dir), ignore_errors=True)
