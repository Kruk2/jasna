"""Basic settings section."""

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

        # Max Clip Size slider (10-180, step 10)
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        clip_label = ctk.CTkLabel(row1, text=t("max_clip_size"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        clip_label.pack(side="left")
        clip_tooltip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        clip_tooltip.pack(side="left", padx=4)
        Tooltip(clip_tooltip, get_tooltip("max_clip_size"))

        self._widgets["max_clip_size_val"] = create_slider_value_label(
            row1, "90", 4, Colors.BG_PANEL
        )
        self._widgets["max_clip_size_val"].pack(side="right")
        self._widgets["max_clip_size"] = ctk.CTkSlider(
            row1, from_=10, to=180, number_of_steps=17,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=self._on_max_clip_size_slider
        )
        self._widgets["max_clip_size"].pack(side="right", padx=(0, 8))
        self._widgets["max_clip_size"].set(90)

        # Detection Model
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        model_label = ctk.CTkLabel(row2, text=t("detection_model"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        model_label.pack(side="left")
        model_tip = ctk.CTkLabel(row2, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        model_tip.pack(side="left", padx=4)
        Tooltip(model_tip, get_tooltip("detection_model"))

        from jasna.mosaic.detection_registry import detection_model_choices
        available_models = detection_model_choices()
        self._widgets["detection_model"] = ctk.CTkOptionMenu(
            row2, values=available_models,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=160,
            command=lambda _value: self._on_modified(),
        )
        self._widgets["detection_model"].pack(side="right")
        self._widgets["detection_model"].set(available_models[0])

        # Detection Threshold
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        thresh_label = ctk.CTkLabel(row3, text=t("detection_threshold"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        thresh_label.pack(side="left")
        thresh_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        thresh_tip.pack(side="left", padx=4)
        Tooltip(thresh_tip, get_tooltip("detection_score_threshold"))

        self._widgets["detection_threshold_val"] = create_slider_value_label(
            row3, "0.25", 4, Colors.BG_PANEL
        )
        self._widgets["detection_threshold_val"].pack(side="right")
        self._widgets["detection_score_threshold"] = ctk.CTkSlider(
            row3, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160, command=lambda v: self._widgets["detection_threshold_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["detection_score_threshold"].pack(side="right", padx=(0, 8))
        self._widgets["detection_score_threshold"].set(0.25)

        # Toggles row - FP16 Mode and Compile BasicVSR++
        row4 = ctk.CTkFrame(inner, fg_color="transparent")
        row4.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

        fp16_frame = ctk.CTkFrame(row4, fg_color=Colors.BG_CARD, corner_radius=6)
        fp16_frame.pack(side="left", fill="x", expand=True, padx=(0, 4))
        fp16_label = ctk.CTkLabel(fp16_frame, text=t("fp16_mode"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        fp16_label.pack(side="left", padx=12, pady=8)
        fp16_tip = ctk.CTkLabel(fp16_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        fp16_tip.pack(side="left")
        Tooltip(fp16_tip, get_tooltip("fp16_mode"))
        self._widgets["fp16_mode"] = create_compact_switch(
            fp16_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["fp16_mode"].pack(side="right", padx=12, pady=8)
        self._widgets["fp16_mode"].select()

        compile_frame = ctk.CTkFrame(row4, fg_color=Colors.BG_CARD, corner_radius=6)
        compile_frame.pack(side="right", fill="x", expand=True, padx=(4, 0))
        compile_label = ctk.CTkLabel(compile_frame, text=t("compile_basicvsrpp"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        compile_label.pack(side="left", padx=12, pady=8)
        compile_tip = ctk.CTkLabel(compile_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        compile_tip.pack(side="left")
        Tooltip(compile_tip, get_tooltip("compile_basicvsrpp"))
        self._widgets["compile_basicvsrpp"] = create_compact_switch(
            compile_frame,
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["compile_basicvsrpp"].pack(side="right", padx=12, pady=8)
        self._widgets["compile_basicvsrpp"].select()

        # File Conflict dropdown
        row5 = ctk.CTkFrame(inner, fg_color="transparent")
        row5.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

        conflict_label = ctk.CTkLabel(row5, text=t("file_conflict"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        conflict_label.pack(side="left")
        conflict_tip = ctk.CTkLabel(row5, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        conflict_tip.pack(side="left", padx=4)
        Tooltip(conflict_tip, get_tooltip("file_conflict"))

        # Warning icon for overwrite (hidden by default)
        self._widgets["conflict_warning"] = ctk.CTkLabel(
            row5, text="⚠️", text_color=Colors.STATUS_PAUSED, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL)
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

    def _on_max_clip_size_slider(self, value: float):
        max_clip_size = int(value)
        self._widgets["max_clip_size_val"].configure(text=str(max_clip_size))
        self._on_modified()
        self._on_max_clip_size_change(max_clip_size)

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
        if det_model not in self._widgets["detection_model"].cget("values"):
            from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
            det_model = "rfdetr-v5"
            det_threshold = max(det_threshold, RfDetrMosaicDetectionModel.DEFAULT_SCORE_THRESHOLD)
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
