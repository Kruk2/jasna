"""BasicVSR++ processing settings section (clip overlap and denoise)."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    create_slider_value_label,
)
from jasna.gui.theme import Colors, Sizing


class ProcessingSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified

        self.frame = CollapsibleSection(parent, t("section_processing"), expanded=True)
        content = self.frame.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        overlap_row = ctk.CTkFrame(inner, fg_color="transparent")
        overlap_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
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

        denoise_row = ctk.CTkFrame(inner, fg_color="transparent")
        denoise_row.pack(fill="x")
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

    def _on_temporal_overlap_slider(self, value: float):
        self._widgets["temporal_overlap_val"].configure(text=str(int(value)))
        self._on_modified()

    def apply(self, preset):
        self._widgets["temporal_overlap"].set(preset.temporal_overlap)
        self._widgets["temporal_overlap_val"].configure(text=str(preset.temporal_overlap))
        self._widgets["denoise_strength"].set_value(preset.denoise_strength)

    def collect(self) -> dict:
        return {
            "temporal_overlap": int(self._widgets["temporal_overlap"].get()),
            "denoise_strength": self._widgets["denoise_strength"].get_value(),
        }
