"""Restore preview of the segment editor: the restored frame at the playhead or a restored
playback window outside LTX ranges, and the seed panel's frames over them."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from jasna.gui.locales import t
from jasna.gui.models import AppSettings
from jasna.gui.restoration_preview import (
    RestorationClip,
    RestorationFailed,
    RestorationFrame,
    RestorationPreviewWorker,
    RestorationStatus,
    RestoredClipFrame,
)
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.gui.segment_preview_card import PreviewCard
from jasna.gui.segment_seed_panel import SeedPanel
from jasna.gui.theme import Colors
from jasna.media.probe import VideoMetadata

RESTORE_DELAY_MS = 400


def worker_status_text(message: str) -> str:
    if message == "loading_models":
        return t("segments_restore_loading_models")
    if message == "restoring":
        return t("segments_restore_restoring")
    return message


def restored_clip_covers(clip: tuple[RestoredClipFrame, ...], seconds: float, fps: float) -> bool:
    """Whether playback from ``seconds`` can start inside the restored ``clip``."""
    if len(clip) < 2:
        return False
    tolerance = 0.75 / fps
    return clip[0].seconds - tolerance <= seconds < clip[-1].seconds - tolerance


def clip_finished(clip: tuple[RestoredClipFrame, ...], seconds: float, fps: float) -> bool:
    """Whether playback at ``seconds`` reached the last frame of the restored ``clip``."""
    return seconds >= clip[-1].seconds - 0.5 / fps


def next_window_start(clip: tuple[RestoredClipFrame, ...], fps: float, duration: float) -> float | None:
    """Where the playback window after ``clip`` starts; None when the clip ends the video."""
    frame_duration = 1 / fps
    start = clip[-1].seconds + frame_duration
    return start if start < duration - frame_duration / 2 else None


class RestorePreview:
    """Drives the restore preview switch of ``card`` and owns its ``RestorationPreviewWorker``
    (BasicVSR++ with the editor's settings); over LTX ranges it shows what ``seeds`` rendered."""

    def __init__(
        self,
        card: PreviewCard,
        *,
        video_path: Path,
        metadata: VideoMetadata,
        state: SegmentEditorState,
        seeds: SeedPanel,
        closed: threading.Event,
        current_seconds: Callable[[], float],
        settings: Callable[[], AppSettings],
        projection: Callable[[], str],
        is_gpu_busy: Callable[[], bool],
        set_gpu_busy: Callable[[object, bool], None],
        set_playing: Callable[[bool], None],
        play_from: Callable[[float], None],
    ) -> None:
        self._card = card
        self._canvas = card.canvas
        self._video_path = video_path
        self._metadata = metadata
        self._state = state
        self._seeds = seeds
        self._closed = closed
        self._current_seconds = current_seconds
        self._settings = settings
        self._projection = projection
        self._is_gpu_busy = is_gpu_busy
        self._set_gpu_busy = set_gpu_busy
        self._set_playing = set_playing
        self._play_from = play_from
        self._active = False
        self._after: str | None = None
        self._toggle_blocked = False
        self._worker: RestorationPreviewWorker | None = None
        self._restored_source: Image.Image | None = None
        self._clip: tuple[RestoredClipFrame, ...] = ()
        self._play_pending = False
        self._generation = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def play_pending(self) -> bool:
        return self._play_pending

    @property
    def source(self) -> Image.Image | None:
        return self._restored_source

    def _clear(self) -> None:
        self._play_pending = False
        self._clip = ()
        self._restored_source = None

    def _cancel_scheduled(self) -> None:
        if self._after is not None:
            self._card.after_cancel(self._after)
            self._after = None

    def toggle(self) -> None:
        if self._active:
            self.deactivate()
            return
        if self._is_gpu_busy():
            self._card.restore_toggle.deselect()
            return
        self._active = True
        self._set_playing(False)
        self._card.restore_toggle.select()
        self._restored_source = None
        if self._seeds.unrendered_range() is not None and not self._seeds.render():
            self.deactivate()
            return
        self.request()

    def activate(self) -> None:
        """Turn the preview on without asking for a restoration."""
        if not self._active:
            self._active = True
            self._card.restore_toggle.select()

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._seeds.release()
            worker = RestorationPreviewWorker(
                self._video_path,
                self._metadata,
                on_stopped=lambda: self._set_gpu_busy(worker, False),
            )
            self._set_gpu_busy(worker, True)
            try:
                worker.start()
                self._worker = worker
            except Exception:
                self._set_gpu_busy(worker, False)
                self._active = False
                self._card.restore_toggle.deselect()
                self._canvas.redraw()
                raise

    def close_worker(self) -> None:
        """Stop the worker so another preview or a scan can take the GPU."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def close(self) -> None:
        """Stop for good; ``join`` then waits for the worker."""
        self._cancel_scheduled()
        if self._worker is not None:
            self._worker.close()

    def join(self, timeout: float) -> None:
        if self._worker is not None:
            self._worker.join(timeout)

    def deactivate(self) -> None:
        self._active = False
        if self._worker is not None:
            self._worker.cancel()
        self._set_playing(False)
        self._cancel_scheduled()
        self._clear()
        self._card.restore_toggle.deselect()
        self._seeds.refresh_render_button()
        self._canvas.redraw()

    def schedule(self) -> None:
        """Restore the playhead's frame once it stops moving."""
        self._generation = -1
        if self._after is not None:
            self._card.after_cancel(self._after)
        self._after = self._card.after(RESTORE_DELAY_MS, self.request)

    def after_seek(self) -> None:
        if not self._active:
            return
        self._clear()
        self._canvas.show_message(t("segments_restore_restoring"), Colors.STATUS_PENDING)
        self.schedule()

    def projection_changed(self) -> None:
        if not self._active:
            return
        self._set_playing(False)
        self._clear()
        self.schedule()

    def _playhead_uses_ltx(self) -> bool:
        return self._state.restoration_at(self._current_seconds()).model == "ltx"

    def request(self) -> None:
        """Show the restored frame at the playhead."""
        self._after = None
        if not self._active or self._closed.is_set():
            return
        self._play_pending = False
        self._clip = ()
        self._card.play_button.configure(text="▶")
        self._seeds.refresh_render_button()
        if self._playhead_uses_ltx():
            self._restored_source = None
            self.show_ltx_state()
            return
        if self._seeds.rendering_seed is not None:
            self._canvas.show_message(t("segments_preview_ltx_busy"), Colors.STATUS_PENDING)
            return
        self._ensure_worker()
        self._generation = self._worker.request(
            self._current_seconds(),
            self._settings(),
            projection=self._projection(),
        )
        if self._restored_source is None:
            self._canvas.show_message(t("segments_restore_restoring"), Colors.STATUS_PENDING)

    def show_ltx_state(self) -> None:
        """Preview over LTX: its rendered frames, else what it waits for."""
        segment = self._seeds.ltx_range_at_playhead()
        if segment is None:
            self._canvas.show_message(t("segments_restore_ltx_unavailable"), Colors.STATUS_WARNING)
        elif self._seeds.view_image() is not None:
            self._canvas.redraw()
        elif self._seeds.is_rendering(segment):
            self._canvas.show_message(
                t("segments_preview_ltx_rendering", seed=self._seeds.rendering_seed), Colors.STATUS_PENDING
            )
        else:
            self._canvas.show_message(
                t("segments_preview_ltx_not_rendered", seed=self._seeds.preview_seed(segment)), Colors.STATUS_WARNING
            )

    def play(self, seconds: float) -> None:
        """Play restored from ``seconds``: from the clip at hand, else once its window is restored."""
        if restored_clip_covers(self._clip, seconds, self._state.fps):
            self._start_playback(seconds)
        else:
            self._request_playback(seconds)

    def _request_playback(self, start_seconds: float) -> None:
        if not self._active or self._closed.is_set():
            return
        if self._state.restoration_at(start_seconds).model == "ltx" or self._seeds.rendering_seed is not None:
            self.request()
            return
        self._cancel_scheduled()
        self._set_playing(False)
        self._play_pending = True
        self._clip = ()
        self._restored_source = None
        self._card.play_button.configure(text="…")
        self._canvas.show_message(t("segments_restore_restoring"), Colors.STATUS_PENDING)
        self._ensure_worker()
        self._generation = self._worker.request(
            start_seconds,
            self._settings(),
            projection=self._projection(),
            playback=True,
        )

    def cancel_pending_play(self) -> None:
        self._play_pending = False
        self._card.play_button.configure(text="▶")
        self.request()

    def poll(self) -> None:
        if self._worker is not None:
            while not self._worker.events.empty():
                self.handle_event(self._worker.events.get_nowait())

    def handle_event(self, event) -> None:
        if not self._active or event.generation != self._generation:
            return
        if isinstance(event, RestorationStatus):
            if self._restored_source is None:
                self._canvas.show_message(worker_status_text(event.message), Colors.STATUS_PENDING)
        elif isinstance(event, RestorationFrame):
            self._clip = ()
            self._restored_source = event.image
            self._canvas.redraw()
        elif isinstance(event, RestorationClip):
            self._clip = event.frames
            if not event.frames:
                self._play_pending = False
                self._card.play_button.configure(text="▶")
                return
            self._restored_source = event.frames[0].image
            self._canvas.redraw()
            if self._play_pending:
                self._play_pending = False
                self._start_playback(event.frames[0].seconds)
        elif isinstance(event, RestorationFailed):
            self._clear()
            self._card.play_button.configure(text="▶")
            self._canvas.show_message(t("segments_restore_failed", message=event.message), Colors.STATUS_ERROR)

    def _start_playback(self, seconds: float) -> None:
        self.show_clip_frame(seconds)
        self._play_from(seconds)

    def playing_clip(self) -> bool:
        return self._active and bool(self._clip)

    def show_clip_frame(self, seconds: float) -> None:
        frame = min(self._clip, key=lambda item: abs(item.seconds - seconds))
        self._restored_source = frame.image
        self._canvas.redraw()

    def finish_clip_at(self, seconds: float) -> bool:
        """At the end of the restored clip, stop and restore the next window; False while
        playback should go on."""
        if not self.playing_clip() or not clip_finished(self._clip, seconds, self._state.fps):
            return False
        self._set_playing(False)
        start = next_window_start(self._clip, self._state.fps, self._state.duration)
        if start is not None:
            self._request_playback(start)
        return True

    def refresh_toggle(self) -> None:
        busy = bool(self._is_gpu_busy())
        if busy == self._toggle_blocked:
            return
        self._toggle_blocked = busy
        if busy and self._active:
            self.deactivate()
        self._card.restore_toggle.configure(state="disabled" if busy else "normal")
        self._card.restore_toggle_tooltip.set_text(
            t("segments_restore_gpu_busy") if busy else t("segments_restore_preview_hint")
        )
        self._seeds.refresh_render_button()
