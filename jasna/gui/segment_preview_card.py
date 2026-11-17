"""Preview card of the segment editor: the preview canvas with its transport, zoom controls and
preview options (restore preview switch and VR projection choice)."""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk
from PIL import Image

from jasna.gui.components import Tooltip
from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.preview_canvas import PREVIEW_ZOOM_MAX, PREVIEW_ZOOM_STEP, PreviewCanvas
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.vr180 import VrModeResolution


class PreviewCard(ctk.CTkFrame):
    """Shows ``source()`` on ``canvas``; the seed panel packs its LTX render button into ``options``."""

    def __init__(
        self,
        master,
        *,
        source: Callable[[], Image.Image | None],
        vr_resolution: VrModeResolution,
        vr_projection: str,
        time_text: str,
        on_step: Callable[[int], None],
        on_toggle_play: Callable[[], None],
        on_suggest_mask: Callable[[], None],
        on_toggle_restore: Callable[[], None],
        on_projection_changed: Callable[[str], None],
    ) -> None:
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=Sizing.BORDER_RADIUS)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        preview_frame = ctk.CTkFrame(self, fg_color=Colors.BG_PANEL, corner_radius=Sizing.BORDER_RADIUS)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        self.canvas = PreviewCanvas(
            preview_frame,
            source=source,
            pan_button=1,
            on_view_changed=self._update_zoom_controls,
        )
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Double-Button-1>", self._reset_view)
        self.canvas.show_message(t("segments_loading_preview"), Colors.STATUS_PENDING)
        self._build_transport(time_text, on_step, on_toggle_play, on_suggest_mask)
        self._build_zoom_controls()
        self._build_options(vr_resolution, vr_projection, on_toggle_restore, on_projection_changed)

    def _build_transport(
        self,
        time_text: str,
        on_step: Callable[[int], None],
        on_toggle_play: Callable[[], None],
        on_suggest_mask: Callable[[], None],
    ) -> None:
        """Frame stepping, play, time and the mask suggestion button under the preview."""
        transport = ctk.CTkFrame(self, fg_color="transparent")
        transport.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 2))
        self._step_back = ctk.CTkButton(
            transport,
            text="|◀",
            width=44,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: on_step(-1),
        )
        self._step_back.pack(side="left")
        Tooltip(self._step_back, t("segments_previous_frame"))
        self.play_button = ctk.CTkButton(
            transport,
            text="▶",
            width=48,
            command=on_toggle_play,
        )
        self.play_button.pack(side="left", padx=6)
        self._step_forward = ctk.CTkButton(
            transport,
            text="▶|",
            width=44,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: on_step(1),
        )
        self._step_forward.pack(side="left")
        Tooltip(self._step_forward, t("segments_next_frame"))
        self.time_label = ctk.CTkLabel(
            transport,
            text=time_text,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self.time_label.pack(side="left", padx=12)
        self.suggest_button = ctk.CTkButton(
            transport,
            text=t("segments_suggest_mask"),
            height=28,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=on_suggest_mask,
        )
        self.suggest_button.pack(side="left", padx=(12, 0))
        Tooltip(self.suggest_button, t("segments_suggest_mask_hint"))

    def _build_zoom_controls(self) -> None:
        view_controls = ctk.CTkFrame(self, fg_color="transparent")
        view_controls.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        self._pan_zoom_hint = ctk.CTkLabel(
            view_controls,
            text=t("segments_preview_pan_zoom_hint"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        )
        self._pan_zoom_hint.pack(side="left")
        Tooltip(self._pan_zoom_hint, t("segments_preview_pan_zoom_hint"))
        zoom_controls = ctk.CTkFrame(view_controls, fg_color="transparent")
        zoom_controls.pack(side="right")
        self._zoom_out_btn = ctk.CTkButton(
            zoom_controls,
            text="−",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self.canvas.zoom_by(-PREVIEW_ZOOM_STEP),
        )
        self._zoom_out_btn.pack(side="left")
        Tooltip(self._zoom_out_btn, t("segments_preview_zoom_out"))
        self._zoom_label = ctk.CTkLabel(
            zoom_controls,
            text="100%",
            width=48,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._zoom_label.pack(side="left", padx=3)
        self._zoom_in_btn = ctk.CTkButton(
            zoom_controls,
            text="+",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self.canvas.zoom_by(PREVIEW_ZOOM_STEP),
        )
        self._zoom_in_btn.pack(side="left")
        Tooltip(self._zoom_in_btn, t("segments_preview_zoom_in"))
        self._reset_view_btn = ctk.CTkButton(
            zoom_controls,
            text=t("segments_preview_reset_view"),
            width=84,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self._reset_view,
        )
        Tooltip(
            self._reset_view_btn,
            t("segments_preview_reset_view_hint"),
        )
        self._update_zoom_controls()

    def _build_options(
        self,
        vr_resolution: VrModeResolution,
        vr_projection: str,
        on_toggle_restore: Callable[[], None],
        on_projection_changed: Callable[[str], None],
    ) -> None:
        """Restore preview switch and VR projection choice."""
        self.options = ctk.CTkFrame(self, fg_color="transparent")
        self.options.grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 8))
        restore_control = ctk.CTkFrame(self.options, fg_color="transparent")
        restore_control.pack(side="right")
        self.restore_toggle = CompactSwitch(
            restore_control,
            on_toggle_restore,
            Colors.BG_CARD,
        )
        self.restore_toggle.pack(side="right")
        ctk.CTkLabel(
            restore_control,
            text=t("segments_restore_preview"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
        ).pack(side="right", padx=(0, 6))
        self.restore_toggle_tooltip = Tooltip(self.restore_toggle, t("segments_restore_preview_hint"))
        self._build_projection_control(vr_resolution, vr_projection, on_projection_changed)

    def _build_projection_control(
        self,
        vr_resolution: VrModeResolution,
        vr_projection: str,
        on_projection_changed: Callable[[str], None],
    ) -> None:
        projection_names = {
            "raw": t("segments_vr_projection_raw"),
            "fisheye": t("segments_vr_projection_fisheye"),
            "gnomonic": t("segments_vr_projection_gnomonic"),
        }
        projection_control = ctk.CTkFrame(self.options, fg_color="transparent")
        projection_control.pack(side="right", padx=(0, 16))
        projection_label_text = t("segments_vr_projection")
        if vr_resolution.is_stereo:
            projection_label_text = t(
                "segments_vr_projection_resolved",
                projection=projection_names[vr_resolution.projection],
            )
        self._vr_projection_label = ctk.CTkLabel(
            projection_control,
            text=projection_label_text,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        )
        self._vr_projection_label.pack(side="left", padx=(0, 6))
        self._vr_projection_menu = ValueOptionMenu(
            projection_control,
            options={
                "auto": t("segments_vr_projection_auto"),
                **projection_names,
            },
            command=on_projection_changed,
            fg_color=Colors.BG_PANEL,
            button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=150,
            state="normal" if vr_resolution.is_stereo else "disabled",
        )
        self._vr_projection_menu.pack(side="left")
        self._vr_projection_menu.set_value(vr_projection)
        Tooltip(self._vr_projection_label, t("segments_vr_projection_hint"))
        Tooltip(self._vr_projection_menu, t("segments_vr_projection_hint"))

    def lockable_widgets(self) -> tuple:
        return (self._step_back, self._step_forward, self.play_button, self.restore_toggle, self.suggest_button)

    def _update_zoom_controls(self) -> None:
        view = self.canvas.view
        self._zoom_label.configure(text=f"{round(view.zoom * 100)}%")
        self._zoom_out_btn.configure(state="normal" if view.zoomed else "disabled")
        self._zoom_in_btn.configure(state="disabled" if view.zoom >= PREVIEW_ZOOM_MAX else "normal")
        reset_needed = not view.is_reset
        if reset_needed and not self._reset_view_btn.winfo_manager():
            self._reset_view_btn.pack(side="left", padx=(6, 0))
        elif not reset_needed and self._reset_view_btn.winfo_manager():
            self._reset_view_btn.pack_forget()
        self.canvas.configure(cursor="fleur" if view.zoomed else "")

    def _reset_view(self, event=None):
        self.canvas.reset_view()
        return "break" if event is not None else None
