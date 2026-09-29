"""Basic settings section."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    create_slider_value_label,
    pack_rows,
)
from jasna.gui.theme import Colors, Fonts, Sizing

# The sub-engines no longer depend on the clip length, so a long clip costs
# activation memory only — high values stay usable on a large GPU.
MIN_CLIP_SIZE = 10
MAX_CLIP_SIZE = 720
CLIP_SIZE_STEP = 10


def build_max_clip_size_slider(slider_class, parent, on_change, **kwargs):
    return slider_class(
        parent,
        from_=MIN_CLIP_SIZE,
        to=MAX_CLIP_SIZE,
        number_of_steps=(MAX_CLIP_SIZE - MIN_CLIP_SIZE) // CLIP_SIZE_STEP,
        command=on_change,
        **kwargs,
    )


class BasicSection:
    def __init__(self, parent, widgets: dict, on_modified, on_max_clip_size_change):
        self._widgets = widgets
        self._on_modified = on_modified
        self._on_max_clip_size_change = on_max_clip_size_change

        section = CollapsibleSection(parent, t("section_basic"), expanded=True)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        # Max Clip Size slider
        row1 = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(row1, "max_clip_size")

        self._widgets["max_clip_size_val"] = create_slider_value_label(
            row1, "90", 4, Colors.BG_PANEL
        )
        self._widgets["max_clip_size_val"].pack(side="right")
        self._widgets["max_clip_size"] = build_max_clip_size_slider(
            ctk.CTkSlider, row1, self._on_max_clip_size_slider,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY, width=200,
        )
        self._widgets["max_clip_size"].pack(side="right", padx=(0, 8))
        self._widgets["max_clip_size"].set(90)

        # Detection Model
        row2 = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(row2, "detection_model")

        from jasna.mosaic.detection_registry import detection_model_choices
        available_models = detection_model_choices()
        self._widgets["detection_model"] = ctk.CTkOptionMenu(
            row2, values=available_models,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=160,
            command=self._on_detection_model_changed,
        )
        self._widgets["detection_model"].pack(side="right")
        self._widgets["detection_model"].set(available_models[0])

        # Detection Threshold
        row3 = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(row3, "detection_threshold", "detection_score_threshold")

        self._widgets["detection_threshold_val"] = create_slider_value_label(
            row3, "0.35", 4, Colors.BG_PANEL
        )
        self._widgets["detection_threshold_val"].pack(side="right")
        self._widgets["detection_score_threshold"] = ctk.CTkSlider(
            row3, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160, command=lambda v: self._widgets["detection_threshold_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["detection_score_threshold"].pack(side="right", padx=(0, 8))
        self._widgets["detection_score_threshold"].set(0.35)

        # Toggles row - FP16 Mode and Compile BasicVSR++
        row4 = ctk.CTkFrame(inner, fg_color="transparent")

        fp16_frame = ctk.CTkFrame(row4, fg_color=Colors.BG_CARD, corner_radius=6)
        add_setting_label(fp16_frame, "fp16_mode", in_card=True)
        self._widgets["fp16_mode"] = CompactSwitch(
            fp16_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["fp16_mode"].pack(side="right", padx=12, pady=8)
        self._widgets["fp16_mode"].select()

        compile_frame = ctk.CTkFrame(row4, fg_color=Colors.BG_CARD, corner_radius=6)
        add_setting_label(compile_frame, "compile_basicvsrpp", in_card=True)
        self._widgets["compile_basicvsrpp"] = CompactSwitch(
            compile_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["compile_basicvsrpp"].pack(side="right", padx=12, pady=8)
        self._widgets["compile_basicvsrpp"].select()

        # File Conflict dropdown
        row5 = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(row5, "file_conflict")

        # Warning icon for overwrite (hidden by default)
        self._widgets["conflict_warning"] = ctk.CTkLabel(
            row5, text="⚠️", text_color=Colors.STATUS_WARNING, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL)
        )

        self._widgets["file_conflict"] = ValueOptionMenu(
            row5,
            options={
                "auto_rename": t("file_conflict_auto_rename"),
                "overwrite": t("file_conflict_overwrite"),
                "skip": t("file_conflict_skip"),
            },
            command=self._on_file_conflict_changed,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=140,
        )
        self._widgets["file_conflict"].pack(side="right")
        self._widgets["file_conflict"].set_value("auto_rename")

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        toggle_gap = dict(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        self._rows = [(row1, row_gap), (row2, row_gap), (row3, row_gap), (row4, toggle_gap), (row5, toggle_gap)]
        self._toggles = [
            (fp16_frame, dict(side="left", fill="x", expand=True, padx=(0, 4))),
            (compile_frame, dict(side="right", fill="x", expand=True, padx=(4, 0))),
        ]
        self._standard_only = (row1, compile_frame)
        self.set_model("basicvsrpp")

    def set_model(self, model: str) -> None:
        hidden = self._standard_only if model == "ltx" else ()
        pack_rows(self._rows, hidden)
        pack_rows(self._toggles, hidden)

    def _on_max_clip_size_slider(self, value: float):
        max_clip_size = int(value)
        self._widgets["max_clip_size_val"].configure(text=str(max_clip_size))
        self._on_modified()
        self._on_max_clip_size_change(max_clip_size)

    def _on_detection_model_changed(self, value: str):
        from jasna.mosaic.detection_registry import recommended_score_threshold

        threshold = recommended_score_threshold(value)
        self._widgets["detection_score_threshold"].set(threshold)
        self._widgets["detection_threshold_val"].configure(text=f"{threshold:.2f}")
        self._on_modified()

    def _on_file_conflict_changed(self, value: str):
        if value == "overwrite":
            self._widgets["conflict_warning"].pack(side="right", padx=(0, 8))
            Tooltip(self._widgets["conflict_warning"], t("file_conflict_overwrite_warning"))
        else:
            self._widgets["conflict_warning"].pack_forget()
        self._on_modified()

    def apply(self, preset):
        self._widgets["max_clip_size"].set(preset.max_clip_size)
        self._widgets["max_clip_size_val"].configure(text=str(preset.max_clip_size))

        if preset.fp16_mode:
            self._widgets["fp16_mode"].select()
        else:
            self._widgets["fp16_mode"].deselect()

        if preset.compile_basicvsrpp:
            self._widgets["compile_basicvsrpp"].select()
        else:
            self._widgets["compile_basicvsrpp"].deselect()

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

        self._widgets["file_conflict"].set_value(preset.file_conflict)
        self._on_file_conflict_changed(self._widgets["file_conflict"].get_value())

    def collect(self) -> dict:
        return {
            "max_clip_size": int(self._widgets["max_clip_size"].get()),
            "fp16_mode": self._widgets["fp16_mode"].get() == 1,
            "detection_model": self._widgets["detection_model"].get(),
            "detection_score_threshold": float(self._widgets["detection_score_threshold"].get()),
            "compile_basicvsrpp": self._widgets["compile_basicvsrpp"].get() == 1,
            "file_conflict": self._widgets["file_conflict"].get_value(),
        }
