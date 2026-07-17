from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

from jasna.gui.theme import Colors, Fonts
from jasna.segments import SegmentRange, format_timestamp


_PAD_X = 12
_MAIN_TOP = 30
_MAIN_BOTTOM = 58
_OVERVIEW_TOP = 78
_OVERVIEW_BOTTOM = 90
_HANDLE_HIT_PX = 8


def timeline_seconds_to_x(
    seconds: float,
    *,
    view_start: float,
    view_end: float,
    width: int,
    padding: int = _PAD_X,
) -> float:
    usable = max(1, int(width) - 2 * int(padding))
    span = max(1e-9, float(view_end) - float(view_start))
    fraction = (float(seconds) - float(view_start)) / span
    return float(padding) + fraction * usable


def timeline_x_to_seconds(
    x: float,
    *,
    view_start: float,
    view_end: float,
    width: int,
    padding: int = _PAD_X,
) -> float:
    usable = max(1, int(width) - 2 * int(padding))
    fraction = (float(x) - float(padding)) / usable
    value = float(view_start) + fraction * (float(view_end) - float(view_start))
    return min(float(view_end), max(float(view_start), value))


class SegmentTimeline(ctk.CTkFrame):
    """Interactive restoration-range timeline with a full-video overview."""

    def __init__(
        self,
        master,
        *,
        duration: float,
        fps: float,
        on_seek: Callable[[float], None],
        on_create: Callable[[float, float], None],
        on_select: Callable[[int], None],
        on_adjust: Callable[[int, float, float], None],
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=6, **kwargs)
        self.duration = max(1e-9, float(duration))
        self.fps = max(1.0, float(fps))
        self._on_seek = on_seek
        self._on_create = on_create
        self._on_select = on_select
        self._on_adjust = on_adjust
        self._segments: tuple[SegmentRange, ...] = ()
        self._detections: tuple[SegmentRange, ...] = ()
        self._selected_index: int | None = None
        self._playhead = 0.0
        self._view_start = 0.0
        self._view_end = self.duration
        self._drag_kind: str | None = None
        self._drag_index: int | None = None
        self._drag_anchor = 0.0
        self._drag_segment: SegmentRange | None = None
        self._draft: tuple[float, float] | None = None

        self._canvas = tk.Canvas(
            self,
            height=100,
            background=Colors.BG_CARD,
            borderwidth=0,
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(fill="x", expand=True, padx=4, pady=2)
        self._canvas.bind("<Configure>", lambda _event: self._redraw_timeline())
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._canvas.bind("<Button-4>", lambda event: self._on_wheel(event, delta=120))
        self._canvas.bind("<Button-5>", lambda event: self._on_wheel(event, delta=-120))

    @property
    def view_range(self) -> tuple[float, float]:
        return self._view_start, self._view_end

    def set_data(
        self,
        *,
        segments: tuple[SegmentRange, ...],
        selected_index: int | None,
        playhead: float,
    ) -> None:
        self._segments = tuple(segments)
        self._selected_index = selected_index
        self._playhead = min(self.duration, max(0.0, float(playhead)))
        self._redraw_timeline()

    def set_detections(self, runs: tuple[SegmentRange, ...]) -> None:
        self._detections = tuple(runs)
        self._redraw_timeline()

    def zoom_in(self) -> None:
        self._zoom(0.6, self._playhead)

    def zoom_out(self) -> None:
        self._zoom(1 / 0.6, self._playhead)

    def fit(self) -> None:
        self._view_start = 0.0
        self._view_end = self.duration
        self._redraw_timeline()

    def reveal(self, seconds: float) -> None:
        seconds = min(self.duration, max(0.0, float(seconds)))
        if self._view_start <= seconds <= self._view_end:
            return
        span = self._view_end - self._view_start
        self._set_view(seconds - span / 2, seconds + span / 2)

    def _redraw_timeline(self) -> None:
        canvas = self._canvas
        canvas.delete("all")
        width = max(2 * _PAD_X + 1, canvas.winfo_width())

        canvas.create_rectangle(
            _PAD_X,
            _MAIN_TOP,
            width - _PAD_X,
            _MAIN_BOTTOM,
            fill=Colors.BG_PANEL,
            outline=Colors.BORDER_LIGHT,
        )
        self._draw_ticks(width)

        for run in self._detections:
            self._draw_range(
                run,
                width,
                top=_MAIN_BOTTOM - 6,
                bottom=_MAIN_BOTTOM - 1,
                fill=Colors.STATUS_PAUSED,
            )

        for index, segment in enumerate(self._segments):
            self._draw_range(
                segment,
                width,
                top=_MAIN_TOP + 7,
                bottom=_MAIN_BOTTOM - 7,
                fill=Colors.PRIMARY,
                outline="#a5b4fc" if index == self._selected_index else "",
                outline_width=2 if index == self._selected_index else 0,
            )

        draft = self._draft
        if draft is not None:
            start, end = sorted(draft)
            self._draw_range(
                SegmentRange(max(0.0, start), max(start + 1e-9, end)),
                width,
                top=_MAIN_TOP + 5,
                bottom=_MAIN_BOTTOM - 5,
                fill=Colors.STATUS_PAUSED,
                stipple="gray50",
            )

        selected = self._selected_segment_for_draw()
        if selected is not None:
            for boundary in (selected.start, selected.end):
                x = self._time_to_x(boundary, width)
                canvas.create_rectangle(
                    x - 3,
                    _MAIN_TOP + 3,
                    x + 3,
                    _MAIN_BOTTOM - 3,
                    fill="#e0e7ff",
                    outline="",
                )

        playhead_x = self._time_to_x(self._playhead, width)
        if _PAD_X <= playhead_x <= width - _PAD_X:
            canvas.create_line(
                playhead_x,
                18,
                playhead_x,
                _MAIN_BOTTOM + 6,
                fill="#f8fafc",
                width=2,
            )
            canvas.create_polygon(
                playhead_x - 5,
                18,
                playhead_x + 5,
                18,
                playhead_x,
                24,
                fill="#f8fafc",
                outline="",
            )

        self._draw_overview(width)

    def _draw_ticks(self, width: int) -> None:
        for index in range(5):
            fraction = index / 4
            seconds = self._view_start + fraction * (self._view_end - self._view_start)
            x = _PAD_X + fraction * (width - 2 * _PAD_X)
            anchor = "w" if index == 0 else "e" if index == 4 else "center"
            self._canvas.create_text(
                x,
                12,
                text=format_timestamp(seconds, milliseconds=False),
                fill=Colors.TEXT_PRIMARY,
                font=(Fonts.FAMILY_MONO, 9),
                anchor=anchor,
            )

    def _draw_overview(self, width: int) -> None:
        self._canvas.create_rectangle(
            _PAD_X,
            _OVERVIEW_TOP,
            width - _PAD_X,
            _OVERVIEW_BOTTOM,
            fill=Colors.BG_PANEL,
            outline=Colors.BORDER_LIGHT,
        )
        usable = width - 2 * _PAD_X
        for segment in self._segments:
            x1 = _PAD_X + segment.start / self.duration * usable
            x2 = _PAD_X + segment.end / self.duration * usable
            self._canvas.create_rectangle(
                x1,
                _OVERVIEW_TOP + 2,
                max(x1 + 1, x2),
                _OVERVIEW_BOTTOM - 2,
                fill=Colors.PRIMARY,
                outline="",
            )
        for run in self._detections:
            x1 = _PAD_X + run.start / self.duration * usable
            x2 = _PAD_X + run.end / self.duration * usable
            self._canvas.create_rectangle(
                x1,
                _OVERVIEW_BOTTOM - 4,
                max(x1 + 1, x2),
                _OVERVIEW_BOTTOM - 1,
                fill=Colors.STATUS_PAUSED,
                outline="",
            )
        x1 = _PAD_X + self._view_start / self.duration * usable
        x2 = _PAD_X + self._view_end / self.duration * usable
        self._canvas.create_rectangle(
            x1,
            _OVERVIEW_TOP - 2,
            max(x1 + 2, x2),
            _OVERVIEW_BOTTOM + 2,
            outline="#e0e7ff",
            width=2,
        )

    def _draw_range(
        self,
        segment: SegmentRange,
        width: int,
        *,
        top: int,
        bottom: int,
        fill: str,
        outline: str = "",
        outline_width: int = 0,
        stipple: str = "",
    ) -> None:
        start = max(segment.start, self._view_start)
        end = min(segment.end, self._view_end)
        if end <= start:
            return
        x1 = self._time_to_x(start, width)
        x2 = self._time_to_x(end, width)
        self._canvas.create_rectangle(
            x1,
            top,
            max(x1 + 2, x2),
            bottom,
            fill=fill,
            outline=outline,
            width=outline_width,
            stipple=stipple,
        )

    def _on_press(self, event) -> None:
        width = self._canvas.winfo_width()
        if event.y >= _OVERVIEW_TOP - 4:
            seconds = self._overview_x_to_time(event.x, width)
            span = self._view_end - self._view_start
            self._set_view(seconds - span / 2, seconds + span / 2)
            self._on_seek(seconds)
            return
        seconds = self._x_to_time(event.x, width)
        selected = self._selected_segment_for_draw()
        if selected is not None and self._selected_index is not None:
            start_x = self._time_to_x(selected.start, width)
            end_x = self._time_to_x(selected.end, width)
            if abs(event.x - start_x) <= _HANDLE_HIT_PX:
                self._drag_kind = "start"
                self._drag_index = self._selected_index
                self._drag_segment = selected
                return
            if abs(event.x - end_x) <= _HANDLE_HIT_PX:
                self._drag_kind = "end"
                self._drag_index = self._selected_index
                self._drag_segment = selected
                return

        hit = self._hit_segment(seconds)
        if hit is not None:
            self._selected_index = hit
            self._on_select(hit)
            self._on_seek(seconds)
            self._redraw_timeline()
            return

        self._drag_kind = "create"
        self._drag_anchor = seconds
        self._draft = (seconds, seconds)
        self._redraw_timeline()

    def _on_motion(self, event) -> None:
        if self._drag_kind is None:
            return
        seconds = self._snap(self._x_to_time(event.x, self._canvas.winfo_width()))
        if self._drag_kind == "create":
            self._draft = (self._drag_anchor, seconds)
        elif self._drag_segment is not None:
            if self._drag_kind == "start":
                start = min(seconds, self._drag_segment.end - 1 / self.fps)
                self._drag_segment = SegmentRange(max(0.0, start), self._drag_segment.end)
            else:
                end = max(seconds, self._drag_segment.start + 1 / self.fps)
                self._drag_segment = SegmentRange(
                    self._drag_segment.start,
                    min(self.duration, end),
                )
        self._redraw_timeline()

    def _on_release(self, event) -> None:
        if self._drag_kind is None:
            return
        kind = self._drag_kind
        index = self._drag_index
        dragged = self._drag_segment
        draft = self._draft
        self._drag_kind = None
        self._drag_index = None
        self._drag_segment = None
        self._draft = None

        if kind == "create" and draft is not None:
            start, end = sorted((self._snap(draft[0]), self._snap(draft[1])))
            if end - start >= 1 / self.fps - 1e-9:
                self._on_create(start, end)
            else:
                self._on_seek(start)
        elif index is not None and dragged is not None:
            self._on_adjust(index, self._snap(dragged.start), self._snap(dragged.end))
        self._redraw_timeline()

    def _on_wheel(self, event, *, delta: int | None = None) -> str:
        wheel_delta = delta if delta is not None else int(getattr(event, "delta", 0))
        if not wheel_delta:
            return "break"
        if int(getattr(event, "state", 0)) & 0x0001:
            direction = -1 if wheel_delta > 0 else 1
            span = self._view_end - self._view_start
            self._set_view(
                self._view_start + direction * span * 0.12,
                self._view_end + direction * span * 0.12,
            )
        else:
            anchor = self._x_to_time(event.x, self._canvas.winfo_width())
            self._zoom(0.75 if wheel_delta > 0 else 1 / 0.75, anchor)
        return "break"

    def _zoom(self, factor: float, anchor: float) -> None:
        old_span = self._view_end - self._view_start
        minimum = max(2 / self.fps, min(2.0, self.duration))
        new_span = min(self.duration, max(minimum, old_span * float(factor)))
        if abs(new_span - old_span) < 1e-9:
            return
        relative = (anchor - self._view_start) / old_span
        start = anchor - relative * new_span
        self._set_view(start, start + new_span)

    def _set_view(self, start: float, end: float) -> None:
        span = min(self.duration, max(1e-9, float(end) - float(start)))
        start = float(start)
        if start < 0:
            start = 0.0
        if start + span > self.duration:
            start = self.duration - span
        self._view_start = max(0.0, start)
        self._view_end = min(self.duration, self._view_start + span)
        self._redraw_timeline()

    def _selected_segment_for_draw(self) -> SegmentRange | None:
        if self._drag_segment is not None:
            return self._drag_segment
        if self._selected_index is None or not 0 <= self._selected_index < len(self._segments):
            return None
        return self._segments[self._selected_index]

    def _hit_segment(self, seconds: float) -> int | None:
        for index in range(len(self._segments) - 1, -1, -1):
            segment = self._segments[index]
            if segment.start <= seconds <= segment.end:
                return index
        return None

    def _time_to_x(self, seconds: float, width: int) -> float:
        return timeline_seconds_to_x(
            seconds,
            view_start=self._view_start,
            view_end=self._view_end,
            width=width,
        )

    def _x_to_time(self, x: float, width: int) -> float:
        return timeline_x_to_seconds(
            x,
            view_start=self._view_start,
            view_end=self._view_end,
            width=width,
        )

    def _overview_x_to_time(self, x: float, width: int) -> float:
        return timeline_x_to_seconds(
            x,
            view_start=0.0,
            view_end=self.duration,
            width=width,
        )

    def _snap(self, seconds: float) -> float:
        return min(self.duration, max(0.0, round(seconds * self.fps) / self.fps))
