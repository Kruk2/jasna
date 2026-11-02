"""Restoration processing settings section (clip length, overlap, precision, denoise)."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection
from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    create_slider_value_label,
    pack_rows,
)
from jasna.gui.theme import Colors, Sizing

# The sub-engines no longer depend on the clip length, so a long clip costs
# activation memory only — high values stay usable on a large GPU.
MIN_CLIP_SIZE = 10
MAX_CLIP_SIZE = 720
CLIP_SIZE_STEP = 10


class ProcessingSection:
    def __init__(self, parent, widgets: dict, on_modified, on_max_clip_size_change):
        self._widgets = widgets
        self._on_modified = on_modified
        self._on_max_clip_size_change = on_max_clip_size_change

        self.frame = CollapsibleSection(parent, t("section_processing"), expanded=True)
        content = self.frame.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        clip_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(clip_row, "max_clip_size")
        self._widgets["max_clip_size_val"] = create_slider_value_label(clip_row, "90", 4, Colors.BG_PANEL)
        self._widgets["max_clip_size_val"].pack(side="right")
        self._widgets["max_clip_size"] = ctk.CTkSlider(
            clip_row,
            from_=MIN_CLIP_SIZE,
            to=MAX_CLIP_SIZE,
            number_of_steps=(MAX_CLIP_SIZE - MIN_CLIP_SIZE) // CLIP_SIZE_STEP,
            command=self._on_max_clip_size_slider,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY, width=200,
        )
        self._widgets["max_clip_size"].pack(side="right", padx=(0, 8))
        self._widgets["max_clip_size"].set(90)

        overlap_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(overlap_row, "temporal_overlap")
        self._widgets["temporal_overlap_val"] = create_slider_value_label(overlap_row, "8", 3, Colors.BG_PANEL)
        self._widgets["temporal_overlap_val"].pack(side="right")
        self._widgets["temporal_overlap"] = ctk.CTkSlider(
            overlap_row, from_=0, to=30, number_of_steps=30,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=self._on_temporal_overlap_slider,
        )
        self._widgets["temporal_overlap"].pack(side="right", padx=(0, 8))
        self._widgets["temporal_overlap"].set(8)

        crossfade_row = ctk.CTkFrame(inner, fg_color="transparent")
        crossfade_frame = ctk.CTkFrame(crossfade_row, fg_color=Colors.BG_CARD, corner_radius=6)
        crossfade_frame.pack(fill="x")
        add_setting_label(crossfade_frame, "enable_crossfade", in_card=True)
        self._widgets["enable_crossfade"] = CompactSwitch(crossfade_frame, self._on_modified, Colors.BG_CARD)
        self._widgets["enable_crossfade"].pack(side="right", padx=12, pady=8)
        self._widgets["enable_crossfade"].select()

        toggles_row = ctk.CTkFrame(inner, fg_color="transparent")
        fp16_frame = ctk.CTkFrame(toggles_row, fg_color=Colors.BG_CARD, corner_radius=6)
        add_setting_label(fp16_frame, "fp16_mode", in_card=True)
        self._widgets["fp16_mode"] = CompactSwitch(fp16_frame, self._on_modified, Colors.BG_CARD)
        self._widgets["fp16_mode"].pack(side="right", padx=12, pady=8)
        self._widgets["fp16_mode"].select()

        compile_frame = ctk.CTkFrame(toggles_row, fg_color=Colors.BG_CARD, corner_radius=6)
        add_setting_label(compile_frame, "compile_basicvsrpp", in_card=True)
        self._widgets["compile_basicvsrpp"] = CompactSwitch(compile_frame, self._on_modified, Colors.BG_CARD)
        self._widgets["compile_basicvsrpp"].pack(side="right", padx=12, pady=8)
        self._widgets["compile_basicvsrpp"].select()

        denoise_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(denoise_row, "denoise_strength")
        self._widgets["denoise_strength"] = ValueOptionMenu(
            denoise_row,
            options={
                "none": t("denoise_none"),
                "low": t("denoise_low"),
                "medium": t("denoise_medium"),
                "high": t("denoise_high"),
            },
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=120,
        )
        self._widgets["denoise_strength"].pack(side="right")
        self._widgets["denoise_strength"].set_value("none")

        denoise_step_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(denoise_step_row, "denoise_step")
        self._widgets["denoise_step"] = ValueOptionMenu(
            denoise_step_row,
            options={
                "after_primary": t("after_primary"),
                "after_secondary": t("after_secondary"),
            },
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=140,
        )
        self._widgets["denoise_step"].pack(side="right")
        self._widgets["denoise_step"].set_value("after_primary")

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._rows = [
            (clip_row, row_gap), (overlap_row, row_gap), (crossfade_row, row_gap),
            (toggles_row, row_gap), (denoise_row, row_gap), (denoise_step_row, dict(fill="x")),
        ]
        self._toggles = [
            (fp16_frame, dict(side="left", fill="x", expand=True, padx=(0, 4))),
            (compile_frame, dict(side="right", fill="x", expand=True, padx=(4, 0))),
        ]
        self._standard_only = (clip_row, overlap_row, crossfade_row, compile_frame, denoise_row, denoise_step_row)
        self.set_ltx_only(False)

    def set_ltx_only(self, ltx_only: bool) -> None:
        hidden = self._standard_only if ltx_only else ()
        pack_rows(self._rows, hidden)
        pack_rows(self._toggles, hidden)

    def _on_max_clip_size_slider(self, value: float):
        max_clip_size = int(value)
        self._widgets["max_clip_size_val"].configure(text=str(max_clip_size))
        self._on_modified()
        self._on_max_clip_size_change(max_clip_size)

    def _on_temporal_overlap_slider(self, value: float):
        self._widgets["temporal_overlap_val"].configure(text=str(int(value)))
        self._on_modified()

    def apply(self, preset):
        self._widgets["max_clip_size"].set(preset.max_clip_size)
        self._widgets["max_clip_size_val"].configure(text=str(preset.max_clip_size))
        self._widgets["temporal_overlap"].set(preset.temporal_overlap)
        self._widgets["temporal_overlap_val"].configure(text=str(preset.temporal_overlap))
        for key in ("enable_crossfade", "fp16_mode", "compile_basicvsrpp"):
            if getattr(preset, key):
                self._widgets[key].select()
            else:
                self._widgets[key].deselect()
        self._widgets["denoise_strength"].set_value(preset.denoise_strength)
        self._widgets["denoise_step"].set_value(preset.denoise_step)

    def collect(self) -> dict:
        return {
            "max_clip_size": int(self._widgets["max_clip_size"].get()),
            "temporal_overlap": int(self._widgets["temporal_overlap"].get()),
            "enable_crossfade": self._widgets["enable_crossfade"].get() == 1,
            "fp16_mode": self._widgets["fp16_mode"].get() == 1,
            "compile_basicvsrpp": self._widgets["compile_basicvsrpp"].get() == 1,
            "denoise_strength": self._widgets["denoise_strength"].get_value(),
            "denoise_step": self._widgets["denoise_step"].get_value(),
        }
