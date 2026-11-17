"""Mosaic detection settings section."""

from typing import get_args

import customtkinter as ctk

from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    SLIDER_WIDTH,
    Row,
    add_setting_label,
    create_slider_value_label,
    layout_rows,
    option_menu,
    set_widgets_enabled,
)
from jasna.gui.theme import Colors, Sizing
from jasna.session_config import DEFAULT_DETECTION_MODEL_NAME, VrModeName

TEMPORAL_FILTER_SLIDER_MAX = 10
_TEMPORAL_FILTERS = ("max_detection_gap", "min_detection_duration")


class DetectionSection:
    def __init__(self, parent, on_modified):
        from jasna.mosaic.detection_registry import detection_model_choices

        self._widgets = {}
        self._on_modified = on_modified

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        model_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(model_row, "detection_model")
        self._models = detection_model_choices()
        self._widgets["detection_model"] = option_menu(
            model_row, {model: model for model in self._models}, self._on_detection_model_changed, width=160
        )
        self._widgets["detection_model"].pack(side="right")

        threshold_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(threshold_row, "detection_threshold", "detection_score_threshold")
        self._widgets["detection_threshold_val"] = create_slider_value_label(threshold_row, Colors.BG_PANEL)
        self._widgets["detection_threshold_val"].pack(side="right")
        self._widgets["detection_score_threshold"] = ctk.CTkSlider(
            threshold_row, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=self._on_threshold_slider,
        )
        self._widgets["detection_score_threshold"].pack(side="right", padx=(0, 8))

        vr_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(vr_row, "vr_mode")
        self._widgets["vr_mode"] = option_menu(
            vr_row,
            {mode: t(f"vr_mode_{mode.replace('-', '_')}") for mode in get_args(VrModeName)},
            lambda _value: self._on_modified(),
            width=180,
        )
        self._widgets["vr_mode"].pack(side="right")

        gap_row, mindur_row = (self._filter_slider_row(inner, key) for key in _TEMPORAL_FILTERS)
        scene_row = ctk.CTkFrame(inner, fg_color="transparent")
        scene_frame = ctk.CTkFrame(scene_row, fg_color=Colors.BG_CARD, corner_radius=6)
        scene_frame.pack(fill="x")
        add_setting_label(scene_frame, "scene_detection", in_card=True)
        self._widgets["scene_detection"] = CompactSwitch(scene_frame, self._on_modified, Colors.BG_CARD)
        self._widgets["scene_detection"].pack(side="right", padx=12, pady=8)

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
        self._widgets[f"{key}_val"] = create_slider_value_label(row, Colors.BG_PANEL)
        self._widgets[f"{key}_val"].pack(side="right")
        self._widgets[key] = ctk.CTkSlider(
            row, from_=0, to=TEMPORAL_FILTER_SLIDER_MAX,
            number_of_steps=TEMPORAL_FILTER_SLIDER_MAX,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=lambda v: self._on_filter_slider(key, int(v))
        )
        self._widgets[key].pack(side="right", padx=(0, 8))
        return row

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        return layout_rows(self._rows, level, standard=standard, ltx=ltx)

    def set_enabled(self, enabled: bool) -> None:
        set_widgets_enabled(
            (self._widgets[key] for key in (
                "detection_model", "detection_score_threshold", "vr_mode", *_TEMPORAL_FILTERS, "scene_detection"
            )),
            enabled,
        )

    def limit_to_clip_size(self, max_clip_size: int) -> None:
        """A detection gap or minimum duration cannot span a whole clip."""
        limit = max(0, min(TEMPORAL_FILTER_SLIDER_MAX, max_clip_size - 1))
        for key in _TEMPORAL_FILTERS:
            slider = self._widgets[key]
            slider.configure(to=limit, number_of_steps=max(1, limit))
            self._set_filter(key, min(int(slider.get()), limit))

    def _set_filter(self, key: str, value: int) -> None:
        self._widgets[key].set(value)
        self._widgets[f"{key}_val"].configure(text=str(value))

    def _on_filter_slider(self, key: str, value: int):
        self._widgets[f"{key}_val"].configure(text=str(value))
        self._on_modified()

    def _set_threshold(self, threshold: float) -> None:
        self._widgets["detection_score_threshold"].set(threshold)
        self._widgets["detection_threshold_val"].configure(text=f"{threshold:.2f}")

    def _on_threshold_slider(self, value: float) -> None:
        self._widgets["detection_threshold_val"].configure(text=f"{value:.2f}")
        self._on_modified()

    def _on_detection_model_changed(self, value: str):
        from jasna.mosaic.detection_registry import recommended_score_threshold

        self._set_threshold(recommended_score_threshold(value))
        self._on_modified()

    def apply(self, preset):
        det_model = preset.detection_model
        det_threshold = preset.detection_score_threshold
        if det_model not in self._models:
            from jasna.mosaic.detection_registry import recommended_score_threshold

            det_model = DEFAULT_DETECTION_MODEL_NAME if DEFAULT_DETECTION_MODEL_NAME in self._models else self._models[0]
            det_threshold = recommended_score_threshold(det_model)
        self._widgets["detection_model"].set_value(det_model)
        self._set_threshold(det_threshold)
        self._widgets["vr_mode"].set_value(preset.vr_mode)
        for key in _TEMPORAL_FILTERS:
            self._set_filter(key, getattr(preset, key))
        self._widgets["scene_detection"].set(preset.scene_detection)

    def collect(self) -> dict:
        return {
            "detection_model": self._widgets["detection_model"].get_value(),
            "detection_score_threshold": float(self._widgets["detection_score_threshold"].get()),
            "vr_mode": self._widgets["vr_mode"].get_value(),
            **{key: int(self._widgets[key].get()) for key in _TEMPORAL_FILTERS},
            "scene_detection": self._widgets["scene_detection"].get() == 1,
        }
