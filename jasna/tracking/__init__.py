from __future__ import annotations

__all__ = ["ClipTracker", "TrackedClip", "EndedClip"]


def __getattr__(name: str):
    if name in {"ClipTracker", "TrackedClip", "EndedClip"}:
        from jasna.tracking import clip_tracker as _clip_tracker

        return getattr(_clip_tracker, name)
    raise AttributeError(name)
