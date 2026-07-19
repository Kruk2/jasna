"""Advanced settings section."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import create_compact_switch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    ValueOptionMenu,
    create_slider_value_label,
    get_tooltip,
)
from jasna.gui.theme import Colors, Fonts, Sizing

TEMPORAL_FILTER_SLIDER_MAX = 10


class AdvancedSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified

        section = CollapsibleSection(parent, t("section_advanced"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        # Temporal Overlap row
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        overlap_label = ctk.CTkLabel(row1, text=t("temporal_overlap"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        overlap_label.pack(side="left")
        overlap_tooltip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        overlap_tooltip.pack(side="left", padx=4)
        Tooltip(overlap_tooltip, get_tooltip("temporal_overlap"))

        self._widgets["temporal_overlap_val"] = create_slider_value_label(
            row1, "8", 3, Colors.BG_PANEL
        )
        self._widgets["temporal_overlap_val"].pack(side="right")
        self._widgets["temporal_overlap"] = ctk.CTkSlider(
            row1, from_=0, to=30, number_of_steps=30,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=lambda v: self._on_slider_change("temporal_overlap", int(v))
        )
        self._widgets["temporal_overlap"].pack(side="right", padx=(0, 8))
        self._widgets["temporal_overlap"].set(8)

        # Max Detection Gap row
        gap_row = ctk.CTkFrame(inner, fg_color="transparent")
        gap_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        gap_label = ctk.CTkLabel(gap_row, text=t("max_detection_gap"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        gap_label.pack(side="left")
        gap_tooltip = ctk.CTkLabel(gap_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        gap_tooltip.pack(side="left", padx=4)
        Tooltip(gap_tooltip, get_tooltip("max_detection_gap"))

        self._widgets["max_detection_gap_val"] = create_slider_value_label(
            gap_row, "2", 3, Colors.BG_PANEL
        )
        self._widgets["max_detection_gap_val"].pack(side="right")
        self._widgets["max_detection_gap"] = ctk.CTkSlider(
            gap_row, from_=0, to=TEMPORAL_FILTER_SLIDER_MAX,
            number_of_steps=TEMPORAL_FILTER_SLIDER_MAX,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=lambda v: self._on_slider_change("max_detection_gap", int(v))
        )
        self._widgets["max_detection_gap"].pack(side="right", padx=(0, 8))
        self._widgets["max_detection_gap"].set(2)

        # Min Detection Duration row
        mindur_row = ctk.CTkFrame(inner, fg_color="transparent")
        mindur_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        mindur_label = ctk.CTkLabel(mindur_row, text=t("min_detection_duration"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        mindur_label.pack(side="left")
        mindur_tooltip = ctk.CTkLabel(mindur_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        mindur_tooltip.pack(side="left", padx=4)
        Tooltip(mindur_tooltip, get_tooltip("min_detection_duration"))

        self._widgets["min_detection_duration_val"] = create_slider_value_label(
            mindur_row, "2", 3, Colors.BG_PANEL
        )
        self._widgets["min_detection_duration_val"].pack(side="right")
        self._widgets["min_detection_duration"] = ctk.CTkSlider(
            mindur_row, from_=0, to=TEMPORAL_FILTER_SLIDER_MAX,
            number_of_steps=TEMPORAL_FILTER_SLIDER_MAX,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=lambda v: self._on_slider_change("min_detection_duration", int(v))
        )
        self._widgets["min_detection_duration"].pack(side="right", padx=(0, 8))
        self._widgets["min_detection_duration"].set(2)

        # Crossfade toggle
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        crossfade_frame = ctk.CTkFrame(row2, fg_color=Colors.BG_CARD, corner_radius=6)
        crossfade_frame.pack(fill="x")
        crossfade_label = ctk.CTkLabel(crossfade_frame, text=t("enable_crossfade"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        crossfade_label.pack(side="left", padx=12, pady=8)
        crossfade_tip = ctk.CTkLabel(crossfade_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        crossfade_tip.pack(side="left")
        Tooltip(crossfade_tip, get_tooltip("enable_crossfade"))
        self._widgets["enable_crossfade"] = create_compact_switch(
            crossfade_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["enable_crossfade"].pack(side="right", padx=12, pady=8)
        self._widgets["enable_crossfade"].select()

        row_vr = ctk.CTkFrame(inner, fg_color="transparent")
        row_vr.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        vr_label = ctk.CTkLabel(
            row_vr,
            text=t("vr_mode"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        vr_label.pack(side="left")
        vr_tip = ctk.CTkLabel(
            row_vr,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        vr_tip.pack(side="left", padx=4)
        Tooltip(vr_tip, get_tooltip("vr_mode"))
        self._widgets["vr_mode"] = ValueOptionMenu(
            row_vr,
            options={
                "auto": t("vr_mode_auto"),
                "off": t("vr_mode_off"),
                "sbs": t("vr_mode_sbs"),
                "sbs-fisheye": t("vr_mode_sbs_fisheye"),
            },
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=180,
        )
        self._widgets["vr_mode"].pack(side="right")
        self._widgets["vr_mode"].set_value("auto")

        # Denoising Strength
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        strength_label = ctk.CTkLabel(row3, text=t("denoise_strength"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        strength_label.pack(side="left")
        strength_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        strength_tip.pack(side="left", padx=4)
        Tooltip(strength_tip, get_tooltip("denoise_strength"))

        self._widgets["denoise_strength"] = ValueOptionMenu(
            row3,
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

        # Denoise Step
        row4 = ctk.CTkFrame(inner, fg_color="transparent")
        row4.pack(fill="x")

        step_label = ctk.CTkLabel(row4, text=t("denoise_step"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        step_label.pack(side="left")
        step_tip = ctk.CTkLabel(row4, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        step_tip.pack(side="left", padx=4)
        Tooltip(step_tip, get_tooltip("denoise_step"))

        self._widgets["denoise_step"] = ValueOptionMenu(
            row4,
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

    def _on_slider_change(self, key: str, value: int):
        self._widgets[f"{key}_val"].configure(text=str(value))
        self._on_modified()

    def apply(self, preset):
        self._widgets["temporal_overlap"].set(preset.temporal_overlap)
        self._widgets["temporal_overlap_val"].configure(text=str(preset.temporal_overlap))
        self._widgets["max_detection_gap"].set(preset.max_detection_gap)
        self._widgets["max_detection_gap_val"].configure(text=str(preset.max_detection_gap))
        self._widgets["min_detection_duration"].set(preset.min_detection_duration)
        self._widgets["min_detection_duration_val"].configure(text=str(preset.min_detection_duration))

        if preset.enable_crossfade:
            self._widgets["enable_crossfade"].select()
        else:
            self._widgets["enable_crossfade"].deselect()

        self._widgets["vr_mode"].set_value(preset.vr_mode)
        self._widgets["denoise_strength"].set_value(preset.denoise_strength)
        self._widgets["denoise_step"].set_value(preset.denoise_step)

    def collect(self) -> dict:
        return {
            "temporal_overlap": int(self._widgets["temporal_overlap"].get()),
            "max_detection_gap": int(self._widgets["max_detection_gap"].get()),
            "min_detection_duration": int(self._widgets["min_detection_duration"].get()),
            "enable_crossfade": self._widgets["enable_crossfade"].get() == 1,
            "vr_mode": self._widgets["vr_mode"].get_value(),
            "denoise_strength": self._widgets["denoise_strength"].get_value(),
            "denoise_step": self._widgets["denoise_step"].get_value(),
        }
