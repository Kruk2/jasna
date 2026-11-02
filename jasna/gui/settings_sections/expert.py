"""Expert settings: performance, detection timing and file handling tweaks most users never need."""

from tkinter import filedialog

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch, create_icon
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    create_slider_value_label,
    get_tooltip,
    pack_rows,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.session_config import LTX_DEFAULT_OVERLAP, LTX_OVERLAP_CHOICES

# The sub-engines no longer depend on the clip length, so a long clip costs
# activation memory only — high values stay usable on a large GPU.
MIN_CLIP_SIZE = 10
MAX_CLIP_SIZE = 720
CLIP_SIZE_STEP = 10
TEMPORAL_FILTER_SLIDER_MAX = 10


class ExpertSection:
    def __init__(self, parent, widgets: dict, on_modified, on_max_clip_size_change):
        self._widgets = widgets
        self._on_modified = on_modified
        self._on_max_clip_size_change = on_max_clip_size_change

        self.frame = CollapsibleSection(parent, t("section_expert"), expanded=True)
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
        add_setting_label(overlap_row, "ltx_overlap")
        self._widgets["ltx_overlap_val"] = create_slider_value_label(
            overlap_row, str(LTX_DEFAULT_OVERLAP), 3, Colors.BG_PANEL
        )
        self._widgets["ltx_overlap_val"].pack(side="right")
        self._widgets["ltx_overlap"] = ctk.CTkSlider(
            overlap_row,
            from_=LTX_OVERLAP_CHOICES[0],
            to=LTX_OVERLAP_CHOICES[-1],
            number_of_steps=len(LTX_OVERLAP_CHOICES) - 1,
            fg_color=Colors.BG_CARD, progress_color=Colors.MODEL_LTX, button_color=Colors.MODEL_LTX,
            width=200, command=lambda v: self._on_slider_change("ltx_overlap", round(v)),
        )
        self._widgets["ltx_overlap"].pack(side="right", padx=(0, 8))
        self._widgets["ltx_overlap"].set(LTX_DEFAULT_OVERLAP)

        gap_row = self._filter_slider_row(inner, "max_detection_gap")
        mindur_row = self._filter_slider_row(inner, "min_detection_duration")

        toggles_row = ctk.CTkFrame(inner, fg_color="transparent")
        fp16_frame = self._switch_card(toggles_row, "fp16_mode")
        compile_frame = self._switch_card(toggles_row, "compile_basicvsrpp")
        scene_row = ctk.CTkFrame(inner, fg_color="transparent")
        scene_frame = self._switch_card(scene_row, "scene_detection")
        scene_frame.pack(fill="x")

        args_label_row = ctk.CTkFrame(inner, fg_color="transparent")
        args_label = ctk.CTkLabel(
            args_label_row, text=t("custom_args"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL)
        )
        args_label.pack(side="left", anchor="w")
        args_tip = ctk.CTkLabel(
            args_label_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2"
        )
        args_tip.pack(side="left", padx=4)
        Tooltip(args_tip, get_tooltip("encoder_custom_args"))
        self._widgets["encoder_custom_args"] = ctk.CTkEntry(
            inner, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("placeholder_encoder_args")
        )

        working_dir_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(working_dir_row, "working_directory")
        working_dir_input_row = ctk.CTkFrame(inner, fg_color="transparent")
        self._widgets["working_directory"] = ctk.CTkEntry(
            working_dir_input_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("working_directory_placeholder"),
        )
        self._widgets["working_directory"].pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(
            working_dir_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_working_directory,
        ).pack(side="right")

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        label_gap = dict(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        input_gap = dict(fill="x", pady=(4, 0))
        self._rows = [
            (clip_row, row_gap), (overlap_row, row_gap), (gap_row, row_gap), (mindur_row, row_gap),
            (toggles_row, row_gap), (scene_row, row_gap),
            (args_label_row, label_gap), (self._widgets["encoder_custom_args"], input_gap),
            (working_dir_row, label_gap), (working_dir_input_row, input_gap),
        ]
        self._toggles = [
            (fp16_frame, dict(side="left", fill="x", expand=True, padx=(0, 4))),
            (compile_frame, dict(side="right", fill="x", expand=True, padx=(4, 0))),
        ]
        self._standard_only = (clip_row, compile_frame, scene_row)
        self._ltx_only = (overlap_row,)
        self.set_models(standard=True, ltx=False)

    def _switch_card(self, master, key: str):
        card = ctk.CTkFrame(master, fg_color=Colors.BG_CARD, corner_radius=6)
        add_setting_label(card, key, in_card=True)
        self._widgets[key] = CompactSwitch(card, self._on_modified, Colors.BG_CARD)
        self._widgets[key].pack(side="right", padx=12, pady=8)
        self._widgets[key].select()
        return card

    def _filter_slider_row(self, inner, key: str):
        row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(row, key)
        self._widgets[f"{key}_val"] = create_slider_value_label(row, "2", 3, Colors.BG_PANEL)
        self._widgets[f"{key}_val"].pack(side="right")
        self._widgets[key] = ctk.CTkSlider(
            row, from_=0, to=TEMPORAL_FILTER_SLIDER_MAX,
            number_of_steps=TEMPORAL_FILTER_SLIDER_MAX,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=lambda v: self._on_slider_change(key, int(v))
        )
        self._widgets[key].pack(side="right", padx=(0, 8))
        self._widgets[key].set(2)
        return row

    def set_models(self, *, standard: bool, ltx: bool) -> None:
        """Show only the settings of the restoration models the job uses."""
        hidden = (() if standard else self._standard_only) + (() if ltx else self._ltx_only)
        pack_rows(self._rows, hidden)
        pack_rows(self._toggles, hidden)

    def _on_slider_change(self, key: str, value: int):
        self._widgets[f"{key}_val"].configure(text=str(value))
        self._on_modified()

    def _on_max_clip_size_slider(self, value: float):
        max_clip_size = int(value)
        self._widgets["max_clip_size_val"].configure(text=str(max_clip_size))
        self._on_modified()
        self._on_max_clip_size_change(max_clip_size)

    def _browse_working_directory(self):
        directory = filedialog.askdirectory(title=t("dialog_select_working_directory"))
        if directory:
            self._widgets["working_directory"].delete(0, "end")
            self._widgets["working_directory"].insert(0, directory)

    def apply(self, preset):
        for key in ("max_clip_size", "ltx_overlap", "max_detection_gap", "min_detection_duration"):
            value = getattr(preset, key)
            self._widgets[key].set(value)
            self._widgets[f"{key}_val"].configure(text=str(value))
        for key in ("fp16_mode", "compile_basicvsrpp", "scene_detection"):
            if getattr(preset, key):
                self._widgets[key].select()
            else:
                self._widgets[key].deselect()
        for key in ("encoder_custom_args", "working_directory"):
            self._widgets[key].delete(0, "end")
            self._widgets[key].insert(0, getattr(preset, key) or "")

    def collect(self) -> dict:
        return {
            "max_clip_size": int(self._widgets["max_clip_size"].get()),
            "ltx_overlap": round(self._widgets["ltx_overlap"].get()),
            "max_detection_gap": int(self._widgets["max_detection_gap"].get()),
            "min_detection_duration": int(self._widgets["min_detection_duration"].get()),
            "fp16_mode": self._widgets["fp16_mode"].get() == 1,
            "compile_basicvsrpp": self._widgets["compile_basicvsrpp"].get() == 1,
            "scene_detection": self._widgets["scene_detection"].get() == 1,
            "encoder_custom_args": self._widgets["encoder_custom_args"].get(),
            "working_directory": self._widgets["working_directory"].get().strip(),
        }
