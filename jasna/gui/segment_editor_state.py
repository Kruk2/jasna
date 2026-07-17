from __future__ import annotations

from dataclasses import dataclass

from jasna.segments import SegmentRange, normalize_segments


@dataclass(frozen=True)
class SegmentEditResult:
    """Result of adding or replacing a range in the editor."""

    segment: SegmentRange
    merged_count: int = 0


@dataclass(frozen=True)
class _Snapshot:
    segments: tuple[SegmentRange, ...]
    selected_index: int | None


class SegmentEditorState:
    """UI-independent state and history for the segment editor."""

    def __init__(
        self,
        *,
        duration: float,
        fps: float,
        segments: tuple[SegmentRange, ...] = (),
    ) -> None:
        duration = float(duration)
        fps = float(fps)
        if duration <= 0:
            raise ValueError("video duration must be greater than zero")
        if fps <= 0:
            raise ValueError("video frame rate must be greater than zero")

        normalized = normalize_segments(segments, duration=duration)
        self.duration = duration
        self.fps = fps
        self.segments = normalized
        self.selected_index: int | None = 0 if normalized else None
        self.mark_in: float | None = None
        self.mark_out: float | None = None
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._initial = self._snapshot()

    @property
    def output_segments(self) -> tuple[SegmentRange, ...]:
        return self.segments

    @property
    def selected_segment(self) -> SegmentRange | None:
        if self.selected_index is None:
            return None
        if not 0 <= self.selected_index < len(self.segments):
            return None
        return self.segments[self.selected_index]

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def dirty(self) -> bool:
        return self.output_segments != self._initial.segments

    @property
    def selected_duration(self) -> float:
        return sum(segment.duration for segment in self.output_segments)

    def snap(self, seconds: float) -> float:
        """Clamp a timestamp to the video and snap it to the nearest frame."""

        value = min(self.duration, max(0.0, float(seconds)))
        snapped = round(value * self.fps) / self.fps
        return min(self.duration, max(0.0, snapped))

    def select(self, index: int | None) -> None:
        if index is None:
            self.selected_index = None
            return
        if not 0 <= int(index) < len(self.segments):
            raise IndexError("segment index out of range")
        self.selected_index = int(index)

    def set_mark_in(self, seconds: float) -> float:
        self.mark_in = self.snap(seconds)
        return self.mark_in

    def set_mark_out(self, seconds: float) -> float:
        self.mark_out = self.snap(seconds)
        return self.mark_out

    def clear_marks(self) -> None:
        self.mark_in = None
        self.mark_out = None

    def add(self, start: float, end: float) -> SegmentEditResult:
        start, end = self._validated_bounds(start, end)
        segment = SegmentRange(start, end)
        overlaps = sum(
            existing.start <= segment.end and segment.start <= existing.end
            for existing in self.segments
        )
        self._record_change()
        self.segments = normalize_segments((*self.segments, segment), duration=self.duration)
        self.selected_index = self._index_containing(segment)
        self.clear_marks()
        return SegmentEditResult(segment, max(0, overlaps))

    def replace_selected(self, start: float, end: float) -> SegmentEditResult:
        if self.selected_index is None:
            return self.add(start, end)
        start, end = self._validated_bounds(start, end)
        replacement = SegmentRange(start, end)
        original_index = self.selected_index
        remaining = [
            segment for index, segment in enumerate(self.segments)
            if index != original_index
        ]
        overlaps = sum(
            existing.start <= replacement.end and replacement.start <= existing.end
            for existing in remaining
        )
        self._record_change()
        self.segments = normalize_segments((*remaining, replacement), duration=self.duration)
        self.selected_index = self._index_containing(replacement)
        self.clear_marks()
        return SegmentEditResult(replacement, max(0, overlaps))

    def adjust_selected(self, start: float, end: float) -> SegmentEditResult:
        """Replace the selected range; used by timeline handle dragging."""

        return self.replace_selected(start, end)

    def add_many(self, ranges: tuple[SegmentRange, ...]) -> int:
        """Add several ranges as one undoable step; returns how many were new."""

        fresh = [
            candidate
            for candidate in ranges
            if not any(
                existing.start <= candidate.start and existing.end >= candidate.end
                for existing in self.segments
            )
        ]
        if not fresh:
            return 0
        self._record_change()
        self.segments = normalize_segments(
            (*self.segments, *fresh), duration=self.duration
        )
        self.selected_index = None
        self.clear_marks()
        return len(fresh)

    def delete_selected(self) -> bool:
        if self.selected_index is None:
            return False
        self._record_change()
        index = self.selected_index
        self.segments = tuple(
            segment for i, segment in enumerate(self.segments) if i != index
        )
        if self.segments:
            self.selected_index = min(index, len(self.segments) - 1)
        else:
            self.selected_index = None
        return True

    def clear(self) -> bool:
        if not self.segments:
            return False
        self._record_change()
        self.segments = ()
        self.selected_index = None
        self.clear_marks()
        return True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def _validated_bounds(self, start: float, end: float) -> tuple[float, float]:
        snapped_start = self.snap(start)
        snapped_end = self.snap(end)
        if snapped_end <= snapped_start:
            raise ValueError("segment end must be greater than start")
        return snapped_start, snapped_end

    def _index_containing(self, segment: SegmentRange) -> int:
        for index, candidate in enumerate(self.segments):
            if candidate.start <= segment.start and candidate.end >= segment.end:
                return index
        raise RuntimeError("normalized segment was not retained")

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(self.segments, self.selected_index)

    def _record_change(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _restore(self, snapshot: _Snapshot) -> None:
        self.segments = snapshot.segments
        self.selected_index = snapshot.selected_index
        self.clear_marks()
