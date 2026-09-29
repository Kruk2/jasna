"""Advanced settings section."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    create_slider_value_label,
    get_tooltip,
    pack_rows,
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

        add_setting_label(row1, "temporal_overlap")

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

        add_setting_label(gap_row, "max_detection_gap")

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

        add_setting_label(mindur_row, "min_detection_duration")

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

        # Scene cut detection toggle
        scene_row = ctk.CTkFrame(inner, fg_color="transparent")

        scene_frame = ctk.CTkFrame(scene_row, fg_color=Colors.BG_CARD, corner_radius=6)
        scene_frame.pack(fill="x")
        add_setting_label(scene_frame, "scene_detection", in_card=True)
        self._widgets["scene_detection"] = CompactSwitch(
            scene_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["scene_detection"].pack(side="right", padx=12, pady=8)
        self._widgets["scene_detection"].select()

        # Crossfade toggle
        row2 = ctk.CTkFrame(inner, fg_color="transparent")

        crossfade_frame = ctk.CTkFrame(row2, fg_color=Colors.BG_CARD, corner_radius=6)
        crossfade_frame.pack(fill="x")
        add_setting_label(crossfade_frame, "enable_crossfade", in_card=True)
        self._widgets["enable_crossfade"] = CompactSwitch(
            crossfade_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["enable_crossfade"].pack(side="right", padx=12, pady=8)
        self._widgets["enable_crossfade"].select()

        row_vr = ctk.CTkFrame(inner, fg_color="transparent")
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

        add_setting_label(row3, "denoise_strength")

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

        add_setting_label(row4, "denoise_step")

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

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._rows = [
            (row1, row_gap), (gap_row, row_gap), (mindur_row, row_gap), (scene_row, row_gap),
            (row2, row_gap), (row_vr, row_gap), (row3, row_gap), (row4, dict(fill="x")),
        ]
        self._vr_row = row_vr
        self.set_model("basicvsrpp")

    def set_model(self, model: str) -> None:
        hidden = tuple(row for row, _options in self._rows if row is not self._vr_row) if model == "ltx" else ()
        pack_rows(self._rows, hidden)

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

        if preset.scene_detection:
            self._widgets["scene_detection"].select()
        else:
            self._widgets["scene_detection"].deselect()

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
            "scene_detection": self._widgets["scene_detection"].get() == 1,
            "enable_crossfade": self._widgets["enable_crossfade"].get() == 1,
            "vr_mode": self._widgets["vr_mode"].get_value(),
            "denoise_strength": self._widgets["denoise_strength"].get_value(),
            "denoise_step": self._widgets["denoise_step"].get_value(),
        }
