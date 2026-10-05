"""Encoding settings section."""

import sys
import customtkinter as ctk
from tkinter import filedialog

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch, create_icon
from jasna.gui.locales import t
from jasna.gui.models import (
    ENCODER_RATE_MODE_AUTO_SOURCE as RATE_MODE_AUTO_SOURCE,
    ENCODER_RATE_MODE_MANUAL_CQ as RATE_MODE_MANUAL_CQ,
    ENCODER_RATE_MODES as RATE_MODES,
)
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    create_slider_value_label,
    get_tooltip,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media.encoder_settings import encoder_cq_spec, validate_encoder_cq

# Display labels contain punctuation ("H.264 (AVC)"), so canonical values come
# from these maps, never from .lower() on the label.
CODEC_LABEL_TO_CANONICAL = {
    "HEVC (H.265)": "hevc",
    "H.264 (AVC)": "h264",
    "AV1": "av1",
}
CODEC_CANONICAL_TO_LABEL = {v: k for k, v in CODEC_LABEL_TO_CANONICAL.items()}

def supports_auto_source_rate_gui(
    codec: str,
    vendor: AcceleratorVendor | str,
    *,
    platform: str | None = None,
) -> bool:
    """Return whether the validated source-rate encoding control applies."""

    return (
        (sys.platform if platform is None else platform) == "linux"
        and AcceleratorVendor(str(vendor)) is AcceleratorVendor.AMD
        and codec == "hevc"
    )


def supports_amd_dual_gop_gui(
    codec: str,
    vendor: AcceleratorVendor | str,
    *,
    platform: str | None = None,
) -> bool:
    """Return whether the proven Linux AMD dual-session control applies."""

    return supports_auto_source_rate_gui(codec, vendor, platform=platform)


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
        self._widgets["codec"].set_value("hevc")
        self._active_codec = "hevc"
        self._cq_vendor = vendor_for_device()
        self._cq_values = {
            codec: encoder_cq_spec(codec, self._cq_vendor).default
            for codec in CODEC_CANONICAL_TO_LABEL
        }

        # Linux AMD HEVC full/Smart Render rate-control policy. Other platforms
        # and codecs keep the established CQ control until their source-rate
        # route has been validated separately.
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        rate_mode_label = ctk.CTkLabel(
            row2,
            text=t("smart_rate_mode"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        rate_mode_label.pack(side="left")
        rate_mode_tip = ctk.CTkLabel(
            row2,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        rate_mode_tip.pack(side="left", padx=4)
        Tooltip(rate_mode_tip, get_tooltip("smart_rate_mode"))
        self._widgets["encoder_rate_mode"] = ValueOptionMenu(
            row2,
            options={
                RATE_MODE_AUTO_SOURCE: t("rate_mode_auto_source"),
                RATE_MODE_MANUAL_CQ: t("rate_mode_manual_cq"),
            },
            command=self._on_rate_mode_changed,
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=190,
        )
        self._widgets["encoder_rate_mode"].pack(side="right")
        self._widgets["encoder_rate_mode"].set_value(RATE_MODE_AUTO_SOURCE)
        self._rate_mode_row = row2

        # Experimental Linux AMD 8K Main10 HEVC temporal multi-session mode.
        dual_gop_row = ctk.CTkFrame(inner, fg_color="transparent")
        dual_gop_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        dual_gop_label = ctk.CTkLabel(
            dual_gop_row,
            text=t("amd_dual_gop_encode"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        dual_gop_label.pack(side="left")
        dual_gop_tip = ctk.CTkLabel(
            dual_gop_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        dual_gop_tip.pack(side="left", padx=4)
        Tooltip(dual_gop_tip, get_tooltip("amd_dual_gop_encode"))
        self._widgets["amd_dual_gop_encode"] = CompactSwitch(
            dual_gop_row,
            self._on_dual_gop_changed,
            Colors.BG_PANEL,
        )
        self._widgets["amd_dual_gop_encode"].pack(side="right")
        self._dual_gop_row = dual_gop_row

        # Quality/CQ
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        cq_label = ctk.CTkLabel(row3, text=t("quality_cq"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        cq_label.pack(side="left")
        cq_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        cq_tip.pack(side="left", padx=4)
        Tooltip(cq_tip, get_tooltip("encoder_cq"))

        initial_cq = self._cq_values[self._active_codec]
        initial_spec = encoder_cq_spec(self._active_codec, self._cq_vendor)
        self._widgets["encoder_cq_val"] = create_slider_value_label(
            row3, str(initial_cq), 3, Colors.BG_PANEL
        )
        self._widgets["encoder_cq_val"].pack(side="right")
        self._widgets["encoder_cq"] = ctk.CTkSlider(
            row3,
            from_=initial_spec.minimum,
            to=initial_spec.maximum,
            number_of_steps=initial_spec.maximum - initial_spec.minimum,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160,
            command=self._on_cq_changed,
        )
        self._widgets["encoder_cq"].pack(side="right", padx=(0, 8))
        self._widgets["encoder_cq"].set(initial_cq)
        self._cq_row = row3

        # Sharpening
        sharpen_row = ctk.CTkFrame(inner, fg_color="transparent")
        sharpen_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._sharpen_row = sharpen_row
        self._sync_rate_control_visibility()

        add_setting_label(sharpen_row, "sharpen_strength")

        self._widgets["sharpen_strength_val"] = create_slider_value_label(
            sharpen_row, "0.00", 4, Colors.BG_PANEL
        )
        self._widgets["sharpen_strength_val"].pack(side="right")
        self._widgets["sharpen_strength"] = ctk.CTkSlider(
            sharpen_row, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160,
            command=lambda v: self._widgets["sharpen_strength_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["sharpen_strength"].pack(side="right", padx=(0, 8))
        self._widgets["sharpen_strength"].set(0.0)

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
        self._widgets["retarget_high_fps"] = CompactSwitch(
            retarget_row,
            lambda: self._on_incompatible_export_toggle("retarget_high_fps"),
            Colors.BG_PANEL,
        )
        self._widgets["retarget_high_fps"].pack(side="right")

        # Fragmented MP4: output stays playable while it is written.
        fmp4_row = ctk.CTkFrame(inner, fg_color="transparent")
        fmp4_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
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
            lambda: self._on_incompatible_export_toggle("fmp4"),
            Colors.BG_PANEL,
        )
        self._widgets["fmp4"].pack(side="right")

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
        add_setting_label(lut_row, "lut_path")

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
        add_setting_label(working_dir_row, "working_directory")

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
        self._sync_rate_control_visibility()
        self._on_modified()

    def _on_rate_mode_changed(self, _new_mode: str):
        if (
            _new_mode != RATE_MODE_AUTO_SOURCE
            and self._widgets["amd_dual_gop_encode"].get() == 1
        ):
            self._widgets["amd_dual_gop_encode"].deselect()
        self._sync_rate_control_visibility()
        self._on_modified()

    def _on_dual_gop_changed(self):
        if self._widgets["amd_dual_gop_encode"].get() == 1:
            self._widgets["encoder_rate_mode"].set_value(
                RATE_MODE_AUTO_SOURCE
            )
            self._widgets["retarget_high_fps"].deselect()
            self._widgets["fmp4"].deselect()
        self._sync_rate_control_visibility()
        self._on_modified()

    def _on_incompatible_export_toggle(self, key: str):
        if (
            self._widgets[key].get() == 1
            and self._widgets["amd_dual_gop_encode"].get() == 1
        ):
            self._widgets["amd_dual_gop_encode"].deselect()
        self._on_modified()

    def _sync_rate_control_visibility(self):
        supports_auto = supports_auto_source_rate_gui(
            self._active_codec,
            self._cq_vendor,
        )
        if supports_auto:
            if not self._rate_mode_row.winfo_manager():
                self._rate_mode_row.pack(
                    fill="x",
                    pady=(0, Sizing.PADDING_SMALL),
                    before=self._sharpen_row,
                )
        else:
            self._rate_mode_row.pack_forget()

        show_cq = (
            not supports_auto
            or self._widgets["encoder_rate_mode"].get_value()
            == RATE_MODE_MANUAL_CQ
        )
        if show_cq:
            if not self._cq_row.winfo_manager():
                self._cq_row.pack(
                    fill="x",
                    pady=(0, Sizing.PADDING_SMALL),
                    before=self._sharpen_row,
                )
        else:
            self._cq_row.pack_forget()

        dual_row = getattr(self, "_dual_gop_row", None)
        dual_widget = self._widgets.get("amd_dual_gop_encode")
        if dual_row is None or dual_widget is None:
            return
        if supports_amd_dual_gop_gui(
            self._active_codec,
            self._cq_vendor,
        ):
            if not dual_row.winfo_manager():
                dual_row.pack(
                    fill="x",
                    pady=(0, Sizing.PADDING_SMALL),
                    before=self._cq_row,
                )
        else:
            dual_row.pack_forget()
            dual_widget.deselect()

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

    def _browse_working_directory(self):
        directory = filedialog.askdirectory(title=t("dialog_select_working_directory"))
        if directory:
            self._widgets["working_directory"].delete(0, "end")
            self._widgets["working_directory"].insert(0, directory)

    def apply(self, preset):
        self._widgets["codec"].set_value(preset.codec)
        self._active_codec = self._widgets["codec"].get_value()
        self._cq_values = {
            codec: encoder_cq_spec(codec, self._cq_vendor).default
            for codec in CODEC_CANONICAL_TO_LABEL
        }
        spec = encoder_cq_spec(self._active_codec, self._cq_vendor)
        cq = spec.default if preset.encoder_cq is None else preset.encoder_cq
        validate_encoder_cq(
            cq,
            codec=self._active_codec,
            vendor=self._cq_vendor,
        )
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
        rate_mode = getattr(preset, "encoder_rate_mode", RATE_MODE_AUTO_SOURCE)
        if rate_mode not in RATE_MODES:
            rate_mode = RATE_MODE_AUTO_SOURCE
        self._widgets["encoder_rate_mode"].set_value(rate_mode)
        self._sync_rate_control_visibility()
        self._widgets["encoder_custom_args"].delete(0, "end")
        self._widgets["encoder_custom_args"].insert(0, preset.encoder_custom_args)
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
        if getattr(preset, "amd_dual_gop_encode", False):
            self._widgets["amd_dual_gop_encode"].select()
        else:
            self._widgets["amd_dual_gop_encode"].deselect()
        self._sync_rate_control_visibility()

        self._widgets["lut_path"].delete(0, "end")
        self._widgets["lut_path"].insert(0, preset.lut_path or "")

        self._widgets["working_directory"].delete(0, "end")
        self._widgets["working_directory"].insert(0, preset.working_directory or "")

    def collect(self) -> dict:
        return {
            "codec": self._widgets["codec"].get_value(),
            "encoder_rate_mode": self._widgets["encoder_rate_mode"].get_value(),
            "encoder_cq": int(self._widgets["encoder_cq"].get()),
            "encoder_custom_args": self._widgets["encoder_custom_args"].get(),
            "amd_dual_gop_encode": (
                self._widgets["amd_dual_gop_encode"].get() == 1
            ),
            "sharpen_strength": round(float(self._widgets["sharpen_strength"].get()), 2),
            "retarget_high_fps": self._widgets["retarget_high_fps"].get() == 1,
            "fmp4": self._widgets["fmp4"].get() == 1,
            "lut_path": self._widgets["lut_path"].get().strip(),
            "working_directory": self._widgets["working_directory"].get().strip(),
        }
