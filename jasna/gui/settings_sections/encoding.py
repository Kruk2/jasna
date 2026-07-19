"""Encoding settings section."""

import customtkinter as ctk
from tkinter import filedialog

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import create_compact_switch, create_icon
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    ValueOptionMenu,
    create_slider_value_label,
    get_tooltip,
)
from jasna.gui.theme import Colors, Fonts, Sizing

# Display labels contain punctuation ("H.264 (AVC)"), so canonical values come
# from these maps, never from .lower() on the label.
CODEC_LABEL_TO_CANONICAL = {
    "HEVC (H.265)": "hevc",
    "H.264 (AVC)": "h264",
    "AV1": "av1",
}
CODEC_CANONICAL_TO_LABEL = {v: k for k, v in CODEC_LABEL_TO_CANONICAL.items()}

_AV1_CQ_OFFSET = 7
_CQ_MIN = 15
_CQ_MAX = 35


def translate_cq_for_codec(cq: int, old_codec: str, new_codec: str) -> int:
    """Keep roughly equal quality when the user visibly switches codec scales."""
    if old_codec == new_codec:
        return cq
    if old_codec == "av1":
        cq -= _AV1_CQ_OFFSET
    if new_codec == "av1":
        cq += _AV1_CQ_OFFSET
    return max(_CQ_MIN, min(_CQ_MAX, cq))


class EncodingSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified

        section = CollapsibleSection(parent, t("section_encoding"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        # Codec
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        codec_label = ctk.CTkLabel(row1, text=t("codec"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        codec_label.pack(side="left")
        codec_tip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        codec_tip.pack(side="left", padx=4)
        Tooltip(codec_tip, get_tooltip("codec"))
        self._widgets["codec"] = ValueOptionMenu(
            row1,
            options=CODEC_CANONICAL_TO_LABEL,
            command=self._on_codec_changed,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY, width=120,
        )
        self._widgets["codec"].pack(side="right")
        self._widgets["codec"].set_value("hevc")
        self._active_codec = "hevc"

        # Quality/CQ
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        cq_label = ctk.CTkLabel(row2, text=t("quality_cq"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        cq_label.pack(side="left")
        cq_tip = ctk.CTkLabel(row2, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        cq_tip.pack(side="left", padx=4)
        Tooltip(cq_tip, get_tooltip("encoder_cq"))

        self._widgets["encoder_cq_val"] = create_slider_value_label(
            row2, "22", 3, Colors.BG_PANEL
        )
        self._widgets["encoder_cq_val"].pack(side="right")
        self._widgets["encoder_cq"] = ctk.CTkSlider(
            row2, from_=15, to=35, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160, command=lambda v: self._widgets["encoder_cq_val"].configure(text=str(int(v)))
        )
        self._widgets["encoder_cq"].pack(side="right", padx=(0, 8))
        self._widgets["encoder_cq"].set(22)

        # Optional exact 60/59.94 -> 30/29.97 frame-rate retargeting.
        retarget_row = ctk.CTkFrame(inner, fg_color="transparent")
        retarget_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        retarget_label = ctk.CTkLabel(
            retarget_row,
            text=t("retarget_high_fps"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        retarget_label.pack(side="left")
        retarget_tip = ctk.CTkLabel(
            retarget_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        retarget_tip.pack(side="left", padx=4)
        Tooltip(retarget_tip, get_tooltip("retarget_high_fps"))
        self._widgets["retarget_high_fps"] = create_compact_switch(
            retarget_row,
            self._on_modified,
            Colors.BG_PANEL,
        )
        self._widgets["retarget_high_fps"].pack(side="right")

        # Custom args
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x")

        args_label = ctk.CTkLabel(row3, text=t("custom_args"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        args_label.pack(side="left", anchor="w")
        args_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        args_tip.pack(side="left", padx=4)
        Tooltip(args_tip, get_tooltip("encoder_custom_args"))

        args_row = ctk.CTkFrame(inner, fg_color="transparent")
        args_row.pack(fill="x", pady=(4, 0))
        self._widgets["encoder_custom_args"] = ctk.CTkEntry(
            args_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("placeholder_encoder_args")
        )
        self._widgets["encoder_custom_args"].pack(fill="x")

        # LUT (color correction)
        lut_row = ctk.CTkFrame(inner, fg_color="transparent")
        lut_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        lut_label = ctk.CTkLabel(lut_row, text=t("lut_path"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        lut_label.pack(side="left")
        lut_tip = ctk.CTkLabel(lut_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        lut_tip.pack(side="left", padx=4)
        Tooltip(lut_tip, get_tooltip("lut_path"))

        lut_input_row = ctk.CTkFrame(inner, fg_color="transparent")
        lut_input_row.pack(fill="x", pady=(4, 0))
        self._widgets["lut_path"] = ctk.CTkEntry(
            lut_input_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("lut_path_placeholder"),
        )
        self._widgets["lut_path"].pack(side="left", fill="x", expand=True, padx=(0, 4))

        lut_browse_btn = ctk.CTkButton(
            lut_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_lut_path,
        )
        lut_browse_btn.pack(side="right")

        working_dir_row = ctk.CTkFrame(inner, fg_color="transparent")
        working_dir_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        working_dir_label = ctk.CTkLabel(working_dir_row, text=t("working_directory"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        working_dir_label.pack(side="left")
        working_dir_tip = ctk.CTkLabel(working_dir_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        working_dir_tip.pack(side="left", padx=4)
        Tooltip(working_dir_tip, get_tooltip("working_directory"))

        working_dir_input_row = ctk.CTkFrame(inner, fg_color="transparent")
        working_dir_input_row.pack(fill="x", pady=(4, 0))
        self._widgets["working_directory"] = ctk.CTkEntry(
            working_dir_input_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("working_directory_placeholder"),
        )
        self._widgets["working_directory"].pack(side="left", fill="x", expand=True, padx=(0, 4))

        working_dir_browse_btn = ctk.CTkButton(
            working_dir_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_working_directory,
        )
        working_dir_browse_btn.pack(side="right")

    def _on_codec_changed(self, new_codec: str):
        cq = translate_cq_for_codec(
            int(self._widgets["encoder_cq"].get()), self._active_codec, new_codec
        )
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._active_codec = new_codec
        self._on_modified()

    def _browse_lut_path(self):
        filepath = filedialog.askopenfilename(
            title=t("dialog_select_lut"),
            filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")],
        )
        if filepath:
            self._widgets["lut_path"].delete(0, "end")
            self._widgets["lut_path"].insert(0, filepath)

    def _browse_working_directory(self):
        directory = filedialog.askdirectory(title=t("dialog_select_working_directory"))
        if directory:
            self._widgets["working_directory"].delete(0, "end")
            self._widgets["working_directory"].insert(0, directory)

    def apply(self, preset):
        self._widgets["codec"].set_value(preset.codec)
        self._active_codec = self._widgets["codec"].get_value()
        self._widgets["encoder_cq"].set(preset.encoder_cq)
        self._widgets["encoder_cq_val"].configure(text=str(preset.encoder_cq))
        self._widgets["encoder_custom_args"].delete(0, "end")
        self._widgets["encoder_custom_args"].insert(0, preset.encoder_custom_args)
        if preset.retarget_high_fps:
            self._widgets["retarget_high_fps"].select()
        else:
            self._widgets["retarget_high_fps"].deselect()

        self._widgets["lut_path"].delete(0, "end")
        self._widgets["lut_path"].insert(0, preset.lut_path or "")

        self._widgets["working_directory"].delete(0, "end")
        self._widgets["working_directory"].insert(0, preset.working_directory or "")

    def collect(self) -> dict:
        return {
            "codec": self._widgets["codec"].get_value(),
            "encoder_cq": int(self._widgets["encoder_cq"].get()),
            "encoder_custom_args": self._widgets["encoder_custom_args"].get(),
            "retarget_high_fps": self._widgets["retarget_high_fps"].get() == 1,
            "lut_path": self._widgets["lut_path"].get().strip(),
            "working_directory": self._widgets["working_directory"].get().strip(),
        }
