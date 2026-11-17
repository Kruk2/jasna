"""Encoding settings section."""

import customtkinter as ctk
from tkinter import filedialog

from jasna.accelerator import vendor_for_device
from jasna.gui.components import Tooltip
from jasna.gui.icons import CompactSwitch, create_icon
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    SLIDER_WIDTH,
    Row,
    add_setting_label,
    bind_edits,
    create_slider_value_label,
    layout_rows,
    option_menu,
    set_entry_text,
    set_widgets_enabled,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media.encoder_settings import encoder_cq_spec, resolve_cq
from jasna.session_config import DEFAULT_CODEC

CODEC_CANONICAL_TO_LABEL = {"hevc": "HEVC (H.265)", "h264": "H.264 (AVC)", "av1": "AV1"}
_PATH_ENTRIES = ("lut_path", "encoder_custom_args", "working_directory")


class EncodingSection:
    def __init__(self, parent, on_modified):
        self._widgets = {}
        self._on_modified = on_modified

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        conflict_row = ctk.CTkFrame(inner, fg_color="transparent")
        conflict_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        add_setting_label(conflict_row, "file_conflict")
        self._widgets["conflict_warning"] = ctk.CTkLabel(
            conflict_row, text="⚠️", text_color=Colors.STATUS_WARNING, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL)
        )
        Tooltip(self._widgets["conflict_warning"], t("file_conflict_overwrite_warning"))
        self._widgets["file_conflict"] = option_menu(
            conflict_row,
            {value: t(f"file_conflict_{value}") for value in ("auto_rename", "overwrite", "skip")},
            self._on_file_conflict_changed,
            width=140,
        )
        self._widgets["file_conflict"].pack(side="right")

        progress_conflict_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(progress_conflict_row, "saved_progress_conflict")
        self._widgets["saved_progress_conflict"] = option_menu(
            progress_conflict_row,
            {value: t(f"saved_progress_conflict_{value}") for value in ("ask", "start_over", "skip")},
            lambda _value: self._on_modified(),
            width=140,
        )
        self._widgets["saved_progress_conflict"].pack(side="right")

        # Codec
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        add_setting_label(row1, "codec")
        self._widgets["codec"] = option_menu(row1, CODEC_CANONICAL_TO_LABEL, self._on_codec_changed, width=120)
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
            command=self._on_sharpen_slider,
        )
        self._widgets["sharpen_strength"].pack(side="right", padx=(0, 8))

        # Optional exact 60/59.94 -> 30/29.97 frame-rate retargeting.
        retarget_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(retarget_row, "retarget_high_fps")
        self._widgets["retarget_high_fps"] = CompactSwitch(
            retarget_row,
            self._on_modified,
            Colors.BG_PANEL,
        )
        self._widgets["retarget_high_fps"].pack(side="right")

        frame_generation_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(frame_generation_row, "frame_generation")
        self._widgets["frame_generation"] = option_menu(
            frame_generation_row, {"none": t("frame_generation_off"), "2x": "2x"}, lambda _value: self._on_modified(), width=120
        )
        self._widgets["frame_generation"].pack(side="right")
        self._frame_generation_supported = True
        self._enabled = True

        # Fragmented MP4: output stays playable while it is written.
        fmp4_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(fmp4_row, "fmp4")
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

        self._browse_button(lut_input_row, self._browse_lut_path)

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
        self._browse_button(working_dir_input_row, self._browse_working_directory)
        for key in _PATH_ENTRIES:
            bind_edits(self._widgets[key], self._on_modified)

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        label_gap = dict(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        input_gap = dict(fill="x", pady=(4, 0))
        # File conflict, codec and quality above are always shown.
        self._rows = [
            Row(progress_conflict_row, row_gap, "advanced"),
            Row(sharpen_row, row_gap, "advanced"),
            Row(retarget_row, row_gap, "advanced"),
            Row(frame_generation_row, row_gap, "advanced"),
            Row(fmp4_row, row_gap, "advanced"),
            Row(lut_row, label_gap, "advanced"),
            Row(lut_input_row, input_gap, "advanced"),
            Row(args_label_row, label_gap, "expert"),
            Row(self._widgets["encoder_custom_args"], input_gap, "expert"),
            Row(working_dir_row, label_gap, "expert"),
            Row(working_dir_input_row, input_gap, "expert"),
        ]

    @staticmethod
    def _browse_button(row, command) -> None:
        ctk.CTkButton(
            row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=command,
        ).pack(side="right")

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        layout_rows(self._rows, level, standard=standard, ltx=ltx)
        return True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        set_widgets_enabled(
            (self._widgets[key] for key in (
                "file_conflict", "saved_progress_conflict", "codec", "encoder_cq", "sharpen_strength",
                "retarget_high_fps", "fmp4", *_PATH_ENTRIES,
            )),
            enabled,
        )
        self._lock_frame_generation()

    def set_frame_generation_support(self, supported: bool) -> None:
        """Frame generation needs an RTX 40-series or newer GPU; its tooltip says so."""
        self._frame_generation_supported = supported
        if not supported:
            self._widgets["frame_generation"].set_value("none")
        self._lock_frame_generation()

    def _lock_frame_generation(self) -> None:
        unlocked = self._enabled and self._frame_generation_supported
        self._widgets["frame_generation"].configure(state="normal" if unlocked else "disabled")

    def _set_path(self, key: str, path: str) -> None:
        set_entry_text(self._widgets[key], path)
        self._on_modified()

    def _browse_working_directory(self):
        directory = filedialog.askdirectory(title=t("dialog_select_working_directory"))
        if directory:
            self._set_path("working_directory", directory)

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

    def _on_sharpen_slider(self, value: float) -> None:
        self._widgets["sharpen_strength_val"].configure(text=f"{value:.2f}")
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
            self._set_path("lut_path", filepath)

    def apply(self, preset):
        self._widgets["file_conflict"].set_value(preset.file_conflict)
        self._on_file_conflict_changed(self._widgets["file_conflict"].get_value())
        self._widgets["saved_progress_conflict"].set_value(preset.saved_progress_conflict)
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
        for key in _PATH_ENTRIES:
            set_entry_text(self._widgets[key], getattr(preset, key) or "")
        self._widgets["sharpen_strength"].set(preset.sharpen_strength)
        self._widgets["sharpen_strength_val"].configure(
            text=f"{preset.sharpen_strength:.2f}"
        )
        self._widgets["retarget_high_fps"].set(preset.retarget_high_fps)
        self._widgets["frame_generation"].set_value(
            preset.frame_generation if self._frame_generation_supported else "none"
        )
        self._widgets["fmp4"].set(preset.fmp4)

    def collect(self) -> dict:
        return {
            "file_conflict": self._widgets["file_conflict"].get_value(),
            "saved_progress_conflict": self._widgets["saved_progress_conflict"].get_value(),
            "codec": self._widgets["codec"].get_value(),
            "encoder_cq": int(self._widgets["encoder_cq"].get()),
            "sharpen_strength": round(float(self._widgets["sharpen_strength"].get()), 2),
            "retarget_high_fps": self._widgets["retarget_high_fps"].get() == 1,
            "frame_generation": self._widgets["frame_generation"].get_value(),
            "fmp4": self._widgets["fmp4"].get() == 1,
            "lut_path": self._widgets["lut_path"].get().strip(),
            "encoder_custom_args": self._widgets["encoder_custom_args"].get(),
            "working_directory": self._widgets["working_directory"].get().strip(),
        }
