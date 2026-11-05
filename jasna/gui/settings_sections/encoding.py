"""Encoding settings section."""

import customtkinter as ctk
from tkinter import filedialog

from jasna.accelerator import vendor_for_device
from jasna.gui.components import Tooltip
from jasna.gui.icons import CompactSwitch, create_icon
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    SLIDER_WIDTH,
    create_slider_value_label,
    Row,
    get_tooltip,
    layout_rows,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media.encoder_settings import encoder_cq_spec, resolve_cq
from jasna.session_config import DEFAULT_CODEC

# Display labels contain punctuation ("H.264 (AVC)"), so canonical values come
# from these maps, never from .lower() on the label.
CODEC_LABEL_TO_CANONICAL = {
    "HEVC (H.265)": "hevc",
    "H.264 (AVC)": "h264",
    "AV1": "av1",
}
CODEC_CANONICAL_TO_LABEL = {v: k for k, v in CODEC_LABEL_TO_CANONICAL.items()}


class EncodingSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        conflict_row = ctk.CTkFrame(inner, fg_color="transparent")
        conflict_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        add_setting_label(conflict_row, "file_conflict")
        self._widgets["conflict_warning"] = ctk.CTkLabel(
            conflict_row, text="⚠️", text_color=Colors.STATUS_WARNING, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL)
        )
        Tooltip(self._widgets["conflict_warning"], t("file_conflict_overwrite_warning"))
        self._widgets["file_conflict"] = ValueOptionMenu(
            conflict_row,
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

        # Codec
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        add_setting_label(row1, "codec")
        self._widgets["codec"] = ValueOptionMenu(
            row1,
            options=CODEC_CANONICAL_TO_LABEL,
            command=self._on_codec_changed,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY, width=120,
        )
        self._widgets["codec"].pack(side="right")
        self._active_codec = DEFAULT_CODEC
        self._cq_vendor = vendor_for_device()
        self._cq_values = {
            codec: encoder_cq_spec(codec, self._cq_vendor).default
            for codec in CODEC_CANONICAL_TO_LABEL
        }

        # Quality/CQ
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        add_setting_label(row2, "quality_cq", "encoder_cq")

        initial_spec = encoder_cq_spec(self._active_codec, self._cq_vendor)
        self._widgets["encoder_cq_val"] = create_slider_value_label(row2, Colors.BG_PANEL)
        self._widgets["encoder_cq_val"].pack(side="right")
        self._widgets["encoder_cq"] = ctk.CTkSlider(
            row2,
            from_=initial_spec.minimum,
            to=initial_spec.maximum,
            number_of_steps=initial_spec.maximum - initial_spec.minimum,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH,
            command=self._on_cq_changed,
        )
        self._widgets["encoder_cq"].pack(side="right", padx=(0, 8))

        # Sharpening
        sharpen_row = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(sharpen_row, "sharpen_strength")

        self._widgets["sharpen_strength_val"] = create_slider_value_label(sharpen_row, Colors.BG_PANEL)
        self._widgets["sharpen_strength_val"].pack(side="right")
        self._widgets["sharpen_strength"] = ctk.CTkSlider(
            sharpen_row, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH,
            command=lambda v: self._widgets["sharpen_strength_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["sharpen_strength"].pack(side="right", padx=(0, 8))

        # Optional exact 60/59.94 -> 30/29.97 frame-rate retargeting.
        retarget_row = ctk.CTkFrame(inner, fg_color="transparent")
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
        self._widgets["retarget_high_fps"] = CompactSwitch(
            retarget_row,
            self._on_modified,
            Colors.BG_PANEL,
        )
        self._widgets["retarget_high_fps"].pack(side="right")

        # Fragmented MP4: output stays playable while it is written.
        fmp4_row = ctk.CTkFrame(inner, fg_color="transparent")
        fmp4_label = ctk.CTkLabel(
            fmp4_row,
            text=t("fmp4"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        fmp4_label.pack(side="left")
        fmp4_tip = ctk.CTkLabel(
            fmp4_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        fmp4_tip.pack(side="left", padx=4)
        Tooltip(fmp4_tip, get_tooltip("fmp4"))
        self._widgets["fmp4"] = CompactSwitch(
            fmp4_row,
            self._on_modified,
            Colors.BG_PANEL,
        )
        self._widgets["fmp4"].pack(side="right")

        # LUT (color correction)
        lut_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(lut_row, "lut_path")

        lut_input_row = ctk.CTkFrame(inner, fg_color="transparent")
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

        args_label_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(args_label_row, "custom_args", "encoder_custom_args")
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
        # File conflict, codec and quality above are always shown.
        self._rows = [
            Row(sharpen_row, row_gap, "advanced"),
            Row(retarget_row, row_gap, "advanced"),
            Row(fmp4_row, row_gap, "advanced"),
            Row(lut_row, label_gap, "advanced"),
            Row(lut_input_row, input_gap, "advanced"),
            Row(args_label_row, label_gap, "expert"),
            Row(self._widgets["encoder_custom_args"], input_gap, "expert"),
            Row(working_dir_row, label_gap, "expert"),
            Row(working_dir_input_row, input_gap, "expert"),
        ]

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        layout_rows(self._rows, level, standard=standard, ltx=ltx)
        return True

    def _browse_working_directory(self):
        directory = filedialog.askdirectory(title=t("dialog_select_working_directory"))
        if directory:
            self._widgets["working_directory"].delete(0, "end")
            self._widgets["working_directory"].insert(0, directory)

    def _on_file_conflict_changed(self, value: str):
        if value == "overwrite":
            self._widgets["conflict_warning"].pack(side="right", padx=(0, 8))
        else:
            self._widgets["conflict_warning"].pack_forget()
        self._on_modified()

    def _on_codec_changed(self, new_codec: str):
        self._cq_values[self._active_codec] = int(
            self._widgets["encoder_cq"].get()
        )
        self._active_codec = new_codec
        spec = encoder_cq_spec(new_codec, self._cq_vendor)
        self._widgets["encoder_cq"].configure(
            from_=spec.minimum,
            to=spec.maximum,
            number_of_steps=spec.maximum - spec.minimum,
        )
        cq = self._cq_values[new_codec]
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._on_modified()

    def _on_cq_changed(self, value: float):
        cq = int(value)
        self._cq_values[self._active_codec] = cq
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._on_modified()

    def _browse_lut_path(self):
        filepath = filedialog.askopenfilename(
            title=t("dialog_select_lut"),
            filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")],
        )
        if filepath:
            self._widgets["lut_path"].delete(0, "end")
            self._widgets["lut_path"].insert(0, filepath)

    def apply(self, preset):
        self._widgets["file_conflict"].set_value(preset.file_conflict)
        self._on_file_conflict_changed(self._widgets["file_conflict"].get_value())
        self._widgets["codec"].set_value(preset.codec)
        self._active_codec = self._widgets["codec"].get_value()
        self._cq_values = {
            codec: encoder_cq_spec(codec, self._cq_vendor).default
            for codec in CODEC_CANONICAL_TO_LABEL
        }
        spec = encoder_cq_spec(self._active_codec, self._cq_vendor)
        cq = resolve_cq(preset.encoder_cq, codec=self._active_codec, vendor=self._cq_vendor)
        if not isinstance(cq, int) or isinstance(cq, bool):
            raise ValueError(f"GUI CQ must be an integer (got {cq!r})")
        self._cq_values[self._active_codec] = cq
        self._widgets["encoder_cq"].configure(
            from_=spec.minimum,
            to=spec.maximum,
            number_of_steps=spec.maximum - spec.minimum,
        )
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        for key in ("encoder_custom_args", "working_directory"):
            self._widgets[key].delete(0, "end")
            self._widgets[key].insert(0, getattr(preset, key) or "")
        self._widgets["sharpen_strength"].set(preset.sharpen_strength)
        self._widgets["sharpen_strength_val"].configure(
            text=f"{preset.sharpen_strength:.2f}"
        )
        if preset.retarget_high_fps:
            self._widgets["retarget_high_fps"].select()
        else:
            self._widgets["retarget_high_fps"].deselect()
        if preset.fmp4:
            self._widgets["fmp4"].select()
        else:
            self._widgets["fmp4"].deselect()

        self._widgets["lut_path"].delete(0, "end")
        self._widgets["lut_path"].insert(0, preset.lut_path or "")

    def collect(self) -> dict:
        return {
            "file_conflict": self._widgets["file_conflict"].get_value(),
            "codec": self._widgets["codec"].get_value(),
            "encoder_cq": int(self._widgets["encoder_cq"].get()),
            "sharpen_strength": round(float(self._widgets["sharpen_strength"].get()), 2),
            "retarget_high_fps": self._widgets["retarget_high_fps"].get() == 1,
            "fmp4": self._widgets["fmp4"].get() == 1,
            "lut_path": self._widgets["lut_path"].get().strip(),
            "encoder_custom_args": self._widgets["encoder_custom_args"].get(),
            "working_directory": self._widgets["working_directory"].get().strip(),
        }
