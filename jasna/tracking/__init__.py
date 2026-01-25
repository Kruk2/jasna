from __future__ import annotations

__all__ = ["ClipTracker", "TrackedClip", "EndedClip", "FrameBuffer", "PendingFrame"]


def __getattr__(name: str):
    if name in {"ClipTracker", "TrackedClip", "EndedClip"}:
        from jasna.tracking import clip_tracker as _clip_tracker

        return getattr(_clip_tracker, name)
    if name in {"FrameBuffer", "PendingFrame"}:
        from jasna.tracking import frame_buffer as _frame_buffer

        return getattr(_frame_buffer, name)
    raise AttributeError(name)
