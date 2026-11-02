"""Mosaic detection settings section."""

import customtkinter as ctk

from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    Row,
    add_setting_label,
    ValueOptionMenu,
    SLIDER_WIDTH,
    create_slider_value_label,
    layout_rows,
)
from jasna.gui.theme import Colors, Sizing

TEMPORAL_FILTER_SLIDER_MAX = 10


class DetectionSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        model_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(model_row, "detection_model")

        from jasna.mosaic.detection_registry import detection_model_choices
        available_models = detection_model_choices()
        self._widgets["detection_model"] = ctk.CTkOptionMenu(
            model_row, values=available_models,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=160,
            command=self._on_detection_model_changed,
        )
        self._widgets["detection_model"].pack(side="right")
        self._widgets["detection_model"].set(available_models[0])

        threshold_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(threshold_row, "detection_threshold", "detection_score_threshold")
        self._widgets["detection_threshold_val"] = create_slider_value_label(
            threshold_row, "0.35", Colors.BG_PANEL
        )
        self._widgets["detection_threshold_val"].pack(side="right")
        self._widgets["detection_score_threshold"] = ctk.CTkSlider(
            threshold_row, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=lambda v: self._widgets["detection_threshold_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["detection_score_threshold"].pack(side="right", padx=(0, 8))
        self._widgets["detection_score_threshold"].set(0.35)

        vr_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(vr_row, "vr_mode")
        self._widgets["vr_mode"] = ValueOptionMenu(
            vr_row,
            options={
                "auto": t("vr_mode_auto"),
                "off": t("vr_mode_off"),
                "sbs": t("vr_mode_sbs"),
                "sbs-fisheye": t("vr_mode_sbs_fisheye"),
                "tb": t("vr_mode_tb"),
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

        gap_row = self._filter_slider_row(inner, "max_detection_gap")
        mindur_row = self._filter_slider_row(inner, "min_detection_duration")
        scene_row = ctk.CTkFrame(inner, fg_color="transparent")
        scene_frame = ctk.CTkFrame(scene_row, fg_color=Colors.BG_CARD, corner_radius=6)
        scene_frame.pack(fill="x")
        add_setting_label(scene_frame, "scene_detection", in_card=True)
        self._widgets["scene_detection"] = CompactSwitch(scene_frame, self._on_modified, Colors.BG_CARD)
        self._widgets["scene_detection"].pack(side="right", padx=12, pady=8)
        self._widgets["scene_detection"].select()

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._rows = [
            Row(model_row, row_gap, "advanced"),
            Row(threshold_row, row_gap, "advanced"),
            Row(vr_row, row_gap, "advanced"),
            Row(gap_row, row_gap, "expert"),
            Row(mindur_row, row_gap, "expert"),
            Row(scene_row, row_gap, "expert", "standard"),
        ]

    def _filter_slider_row(self, inner, key: str):
        row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(row, key)
        self._widgets[f"{key}_val"] = create_slider_value_label(row, "2", Colors.BG_PANEL)
        self._widgets[f"{key}_val"].pack(side="right")
        self._widgets[key] = ctk.CTkSlider(
            row, from_=0, to=TEMPORAL_FILTER_SLIDER_MAX,
            number_of_steps=TEMPORAL_FILTER_SLIDER_MAX,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=lambda v: self._on_slider_change(key, int(v))
        )
        self._widgets[key].pack(side="right", padx=(0, 8))
        self._widgets[key].set(2)
        return row

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        return layout_rows(self._rows, level, standard=standard, ltx=ltx)

    def _on_slider_change(self, key: str, value: int):
        self._widgets[f"{key}_val"].configure(text=str(value))
        self._on_modified()

    def _on_detection_model_changed(self, value: str):
        from jasna.mosaic.detection_registry import recommended_score_threshold

        threshold = recommended_score_threshold(value)
        self._widgets["detection_score_threshold"].set(threshold)
        self._widgets["detection_threshold_val"].configure(text=f"{threshold:.2f}")
        self._on_modified()

    def apply(self, preset):
        det_model = preset.detection_model
        det_threshold = preset.detection_score_threshold
        choices = self._widgets["detection_model"].cget("values")
        if det_model not in choices:
            from jasna.mosaic.detection_registry import (
                DEFAULT_DETECTION_MODEL_NAME,
                recommended_score_threshold,
            )
            det_model = DEFAULT_DETECTION_MODEL_NAME if DEFAULT_DETECTION_MODEL_NAME in choices else choices[0]
            det_threshold = recommended_score_threshold(det_model)
        self._widgets["detection_model"].set(det_model)
        self._widgets["detection_score_threshold"].set(det_threshold)
        self._widgets["detection_threshold_val"].configure(text=f"{det_threshold:.2f}")
        self._widgets["vr_mode"].set_value(preset.vr_mode)
        for key in ("max_detection_gap", "min_detection_duration"):
            self._widgets[key].set(getattr(preset, key))
            self._widgets[f"{key}_val"].configure(text=str(getattr(preset, key)))
        if preset.scene_detection:
            self._widgets["scene_detection"].select()
        else:
            self._widgets["scene_detection"].deselect()

    def collect(self) -> dict:
        return {
            "detection_model": self._widgets["detection_model"].get(),
            "detection_score_threshold": float(self._widgets["detection_score_threshold"].get()),
            "vr_mode": self._widgets["vr_mode"].get_value(),
            "max_detection_gap": int(self._widgets["max_detection_gap"].get()),
            "min_detection_duration": int(self._widgets["min_detection_duration"].get()),
            "scene_detection": self._widgets["scene_detection"].get() == 1,
        }
