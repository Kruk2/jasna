"""LTX seed rendering of the segment editor: renders the LTX range under the playhead with a
seed, keeps the frames of every tried seed and lets the user compare and pick one."""

from __future__ import annotations

import queue
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from jasna.gui.components import Tooltip, format_duration
from jasna.gui.locales import t
from jasna.gui.ltx_seed_preview import LtxSeedPreviewWorker, SeedFailed, SeedFrame, SeedProgress, SeedReady
from jasna.gui.models import AppSettings
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.gui.theme import Colors, Fonts
from jasna.media.probe import VideoMetadata
from jasna.media.splice import KeyframeIndex
from jasna.segments import SegmentRange

LTX_RENDER_ASK_SECONDS = 120.0

RangeKey = tuple[float, float]


def range_key(segment: SegmentRange) -> RangeKey:
    return (segment.start, segment.end)


def ltx_render_seconds(settings: AppSettings, metadata: VideoMetadata, segment: SegmentRange) -> float:
    """Estimated LTX render time of ``segment`` on this GPU."""
    import torch

    from jasna.ltx.model_files import bundle_names
    from jasna.ltx.plan import window_hop
    from jasna.ltx.speed import profile_path, segment_estimate_seconds

    return segment_estimate_seconds(
        profile_path(),
        torch.cuda.get_device_name(0),
        bundle_names(settings.ltx_model, fast=settings.ltx_fast)[0],
        frames=max(1, round(segment.duration * metadata.mean_fps)),
        fps=float(metadata.mean_fps),
        megapixels=metadata.video_width * metadata.video_height / 1e6,
        window_hop=window_hop(settings.ltx_overlap),
    )


class SeedPanel(ctk.CTkFrame):
    """Progress, status and tried-seed chips of the selected LTX range, gridded into the
    restoration inspector ``master``; also owns the inspector's cancel button and the render
    button packed into ``render_parent``."""

    def __init__(
        self,
        master: ctk.CTkFrame,
        *,
        render_parent: ctk.CTkFrame,
        video_path: Path,
        metadata: VideoMetadata,
        state: SegmentEditorState,
        ltx_blocked_reason: str | None,
        keyframe_index: Callable[[], KeyframeIndex | None],
        base_settings: Callable[[], AppSettings],
        video_settings: Callable[[], AppSettings],
        current_seconds: Callable[[], float],
        restore_active: Callable[[], bool],
        is_gpu_busy: Callable[[], bool],
        scanning: Callable[[], bool],
        claim_gpu: Callable[[], None],
        release_gpu: Callable[[], None],
        on_started: Callable[[], None],
        on_cancelled: Callable[[], None],
        on_show_frames: Callable[[RangeKey], None],
        on_use_seed: Callable[[int], None],
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._video_path = video_path
        self._metadata = metadata
        self._state = state
        self._ltx_blocked_reason = ltx_blocked_reason
        self._keyframe_index = keyframe_index
        self._base_settings = base_settings
        self._video_settings = video_settings
        self._current_seconds = current_seconds
        self._restore_active = restore_active
        self._is_gpu_busy = is_gpu_busy
        self._scanning = scanning
        self._claim_gpu = claim_gpu
        self._release_gpu = release_gpu
        self._on_started = on_started
        self._on_cancelled = on_cancelled
        self._on_show_frames = on_show_frames
        self._on_use_seed = on_use_seed
        self._worker: LtxSeedPreviewWorker | None = None
        self._generation = 0
        self._running: tuple[RangeKey, int] | None = None
        self._tried: dict[RangeKey, dict[int, tuple[SeedFrame, ...]]] = {}
        self._shown: tuple[RangeKey, int] | None = None
        self._image: tuple[SeedFrame, Image.Image] | None = None

        self.grid_columnconfigure(0, weight=1)
        self._cancel_btn = ctk.CTkButton(
            master,
            text=t("btn_cancel"),
            width=80,
            height=28,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._cancel,
        )
        Tooltip(self._cancel_btn, t("tip_segments_seed_cancel"))
        self._progress = ctk.CTkProgressBar(self, height=8, progress_color=Colors.MODEL_LTX)
        self._status = ctk.CTkLabel(
            self,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self._chips = ctk.CTkFrame(self, fg_color="transparent")
        self._chip_widgets: list = []
        self._render_btn = ctk.CTkButton(
            render_parent,
            text="",
            height=26,
            fg_color=Colors.MODEL_LTX,
            hover_color=Colors.PRIMARY_HOVER,
            command=self.render,
        )
        self._render_tooltip = Tooltip(self._render_btn, t("tip_segments_ltx_render"))

    @property
    def rendering_seed(self) -> int | None:
        return None if self._running is None else self._running[1]

    def lockable_widgets(self) -> tuple:
        return (self._render_btn,)

    def ltx_range_at_playhead(self) -> SegmentRange | None:
        segment = self._state.segment_at(self._current_seconds())
        return segment if segment is not None and segment.restoration.model == "ltx" else None

    def is_rendering(self, segment: SegmentRange) -> bool:
        return self._running is not None and self._running[0] == range_key(segment)

    def preview_seed(self, segment: SegmentRange) -> int:
        """The seed the preview shows for an LTX range: a tried seed picked from its chips,
        else the range's own."""
        shown = self._shown_seed_for(segment)
        return segment.restoration.ltx_seed if shown is None else shown

    def forget_shown_seed(self) -> None:
        self._shown = None

    def _shown_seed_for(self, segment: SegmentRange) -> int | None:
        if self._shown is None or self._shown[0] != range_key(segment):
            return None
        return self._shown[1]

    def unrendered_range(self) -> SegmentRange | None:
        """The LTX range under the playhead while the restore preview is on and its seed has
        no frames yet."""
        segment = self.ltx_range_at_playhead() if self._restore_active() else None
        if segment is None or self.is_rendering(segment):
            return None
        if self._tried.get(range_key(segment), {}).get(self.preview_seed(segment)):
            return None
        return segment

    def view_image(self) -> Image.Image | None:
        """The rendered frame nearest the playhead of the LTX range under it, while the restore
        preview is on."""
        segment = self.ltx_range_at_playhead() if self._restore_active() else None
        if segment is None:
            return None
        frames = self._tried.get(range_key(segment), {}).get(self.preview_seed(segment), ())
        if not frames:
            return None
        seconds = self._current_seconds()
        frame = min(frames, key=lambda item: abs(item.seconds - seconds))
        if self._image is None or self._image[0] != frame:
            with Image.open(frame.path) as image:
                self._image = (frame, image.convert("RGB"))
        return self._image[1]

    def hide(self) -> None:
        self._cancel_btn.grid_forget()
        self.grid_forget()

    def refresh(self) -> None:
        self.refresh_render_button()
        selected = self._state.selected_segment
        if selected is None or selected.restoration.model != "ltx":
            return
        if self._running is not None:
            self._cancel_btn.grid(row=1, column=3, sticky="e", padx=(0, 8), pady=(0, 6))
            self._progress.grid(row=0, column=0, sticky="ew", pady=(2, 0))
        else:
            self._cancel_btn.grid_forget()
            self._progress.grid_forget()
        if self._status.cget("text"):
            self._status.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        else:
            self._status.grid_forget()
        self._render_chips(selected)
        if self.grid_slaves():
            self.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 6))
        else:
            self.grid_forget()

    def _render_chips(self, selected: SegmentRange) -> None:
        for widget in self._chip_widgets:
            widget.destroy()
        self._chip_widgets = []
        tried = self._tried.get(range_key(selected), {})
        if not tried:
            self._chips.grid_forget()
            return
        self._chips.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        label = ctk.CTkLabel(
            self._chips,
            text=t("segments_seed_tried_trial" if self._video_settings().ltx_trial else "segments_seed_tried"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        )
        label.pack(side="left", padx=(0, 4))
        self._chip_widgets.append(label)
        shown = self._shown_seed_for(selected)
        for seed in tried:
            in_use = seed == selected.restoration.ltx_seed
            chip = ctk.CTkButton(
                self._chips,
                text=f"{seed} ✓" if in_use else str(seed),
                width=10,
                height=24,
                font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
                fg_color=Colors.MODEL_LTX if seed == shown else Colors.BG_CARD,
                hover_color=Colors.BORDER_LIGHT,
                command=lambda value=seed: self._show_tried_seed(value),
            )
            chip.pack(side="left", padx=(0, 4))
            Tooltip(chip, t("tip_segments_seed_chip"))
            self._chip_widgets.append(chip)
        if shown is not None and shown != selected.restoration.ltx_seed:
            use = ctk.CTkButton(
                self._chips,
                text=t("segments_seed_use"),
                height=24,
                font=(Fonts.FAMILY, Fonts.SIZE_TINY),
                command=lambda value=shown: self._on_use_seed(value),
            )
            use.pack(side="left", padx=(4, 0))
            Tooltip(use, t("tip_segments_seed_use"))
            self._chip_widgets.append(use)

    def _render_block_reason(self) -> str | None:
        """Locale key saying why an LTX preview render cannot start now, or None when it can."""
        if self._ltx_blocked_reason is not None:
            return self._ltx_blocked_reason
        if self._is_gpu_busy():
            return "segments_restore_gpu_busy"
        if self._keyframe_index() is None:
            return "segments_loading_preview"
        return None

    def refresh_render_button(self) -> None:
        segment = self.unrendered_range()
        if segment is None:
            self._render_btn.pack_forget()
            return
        reason = self._render_block_reason()
        self._render_btn.configure(
            text=t(
                "segments_ltx_render",
                seed=self.preview_seed(segment),
                duration=format_duration(ltx_render_seconds(self._video_settings(), self._metadata, segment)),
            ),
            state="disabled" if reason is not None or self._scanning() else "normal",
        )
        self._render_tooltip.set_text(t(reason) if reason else t("tip_segments_ltx_render"))
        if not self._render_btn.winfo_manager():
            self._render_btn.pack(side="left")

    def render(self) -> bool:
        """Render the LTX range under the playhead with its preview seed, after asking when it
        takes longer than ``LTX_RENDER_ASK_SECONDS``; returns whether it started."""
        segment = self.unrendered_range()
        if segment is None or self._render_block_reason() is not None:
            return False
        seed = self.preview_seed(segment)
        settings = self._video_settings()
        seconds = ltx_render_seconds(settings, self._metadata, segment)
        if seconds > LTX_RENDER_ASK_SECONDS and not messagebox.askyesno(
            t("segments_ltx_render_confirm_title"),
            t("segments_ltx_render_confirm", seed=seed, duration=format_duration(seconds)),
            parent=self.winfo_toplevel(),
        ):
            return False
        self._state.select(self._state.segments.index(segment))
        self._claim_gpu()
        if self._worker is None:
            base = self._base_settings()
            work_dir = Path(base.working_directory) if base.working_directory else Path(self._video_path).parent
            self._worker = LtxSeedPreviewWorker(
                self._video_path,
                self._metadata,
                self._keyframe_index(),
                work_dir,
                on_stopped=self._release_gpu,
            )
            self._worker.start()
        self._generation = self._worker.try_seed(segment, seed, self._video_settings())
        self._running = (range_key(segment), seed)
        self._shown = (range_key(segment), seed)
        self._progress.set(0.0)
        self._status.configure(text=t("segments_seed_starting", seed=seed), text_color=Colors.STATUS_PENDING)
        self._on_started()
        return True

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._running = None
        self._status.configure(text="")
        self.refresh()
        self._on_cancelled()

    def release(self) -> None:
        """Give the GPU back, stopping a running render."""
        if self._worker is None:
            return
        self._worker.release()
        if self._running is not None:
            self._running = None
            self._status.configure(text="")
            self.refresh()

    def poll(self) -> None:
        if self._worker is None:
            return
        try:
            while True:
                self._handle_event(self._worker.events.get_nowait())
        except queue.Empty:
            pass

    def _handle_event(self, event) -> None:
        if event.generation != self._generation or self._running is None:
            return
        if isinstance(event, SeedProgress):
            self._progress.set(event.fraction)
            stage = t(f"ltx_stage_{event.stage}")
            if self._video_settings().ltx_trial:
                stage = t("ltx_trial_stage", stage=stage)
            values = dict(seed=event.seed, stage=stage, percent=event.fraction * 100)
            text = (
                t("segments_seed_progress", eta=format_duration(event.eta_seconds), **values)
                if event.eta_seconds > 0
                else t("segments_seed_progress_no_eta", **values)
            )
            self._status.configure(text=text, text_color=Colors.STATUS_PENDING)
        elif isinstance(event, SeedReady):
            key, seed = self._running
            self._running = None
            self._tried.setdefault(key, {})[seed] = event.frames
            self._status.configure(text="")
            self._shown = (key, seed)
            self._on_show_frames(key)
            return
        elif isinstance(event, SeedFailed):
            self._running = None
            self._status.configure(
                text=t("segments_seed_failed", message=event.message), text_color=Colors.STATUS_ERROR
            )
        self.refresh()

    def _show_tried_seed(self, seed: int) -> None:
        selected = self._state.selected_segment
        if selected is None:
            return
        self._shown = (range_key(selected), seed)
        self._on_show_frames(range_key(selected))

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    def join(self, timeout: float) -> None:
        if self._worker is not None:
            self._worker.join(timeout)
