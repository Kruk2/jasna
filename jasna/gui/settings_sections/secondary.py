"""Secondary restoration settings section."""

from pathlib import PureWindowsPath
from typing import get_args

import customtkinter as ctk
from tkinter import filedialog

from jasna.accelerator import AcceleratorVendor
from jasna.gui.components import Tooltip
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    Row,
    add_info_icon,
    add_setting_label,
    bind_edits,
    create_slider_value_label,
    folder_button,
    get_tooltip,
    layout_rows,
    option_menu,
    set_entry_text,
    set_widgets_enabled,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.session_config import LTX_SECONDARY_RESTORATIONS, TVAI_DEFAULT_FFMPEG_PATH, RtxLevelName, RtxQualityName
from jasna.session_factory import secondary_restorations_for

_TVAI_MODELS = ("iris-2", "iris-3", "prob-4", "nyx-1")


def _scales(*scales: int) -> dict[str, str]:
    return {str(scale): f"{scale}x" for scale in scales}


def _rtx_levels(levels) -> dict[str, str]:
    return {level: t(f"rtx_level_{level}") for level in levels}


class SecondarySection:
    def __init__(self, parent, on_modified):
        self._widgets = {}
        self._on_modified = on_modified
        self._vendor = AcceleratorVendor.NVIDIA
        self._enabled = True
        self._ltx_only = False

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(
            inner, text=t("upscale_restored_areas_heading"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"), anchor="w",
        ).pack(fill="x", pady=(Sizing.PADDING_SMALL, Sizing.PADDING_SMALL))

        self._engine = ctk.StringVar(value="none")
        engines_frame = ctk.CTkFrame(inner, fg_color="transparent")
        engines_frame.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        add_info_icon(engines_frame, get_tooltip("secondary_restoration")).pack(side="right")

        from jasna.engine_paths import UNET4X_ONNX_ENC_PATH, UNET4X_ONNX_PATH

        self._unet4x_available = UNET4X_ONNX_PATH.exists() or UNET4X_ONNX_ENC_PATH.exists()
        self._engine_radios = {}
        for value, label, tooltip_key in (
            ("none", t("secondary_none"), None),
            ("unet-4x", f"{t('secondary_unet_4x')} ({t('secondary_unet_4x_hint')})", "secondary_unet_4x"),
            ("tvai", f"{t('secondary_tvai')} ({t('secondary_tvai_hint')})", "secondary_tvai"),
            ("rtx-super-res", f"{t('secondary_rtx_super_res')} ({t('secondary_rtx_hint')})", "secondary_rtx"),
        ):
            radio = ctk.CTkRadioButton(
                engines_frame, text=label, variable=self._engine, value=value,
                fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
                command=self._on_engine_picked,
            )
            radio.pack(side="left", padx=(0, 0 if value == "rtx-super-res" else 16))
            if tooltip_key is not None:
                Tooltip(radio, get_tooltip(tooltip_key))
            self._engine_radios[value] = radio

        self._engines_frame = engines_frame
        self._ltx_only_note = ctk.CTkLabel(
            inner, text=t("secondary_ltx_only_note"), text_color=Colors.STATUS_PENDING,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), anchor="w", justify="left",
        )

        self._tvai_frame = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)
        tvai_inner = ctk.CTkFrame(self._tvai_frame, fg_color="transparent")
        tvai_inner.pack(fill="x", padx=12, pady=12)

        tvai_path_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        add_setting_label(tvai_path_row, "ffmpeg_path", "tvai_ffmpeg_path")
        tvai_path_input_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        self._widgets["tvai_ffmpeg_path"] = ctk.CTkEntry(
            tvai_path_input_row, fg_color=Colors.BG_PANEL, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY
        )
        self._widgets["tvai_ffmpeg_path"].pack(side="left", fill="x", expand=True, padx=(0, 4))
        bind_edits(self._widgets["tvai_ffmpeg_path"], self._on_modified)
        folder_button(tvai_path_input_row, self._browse_tvai_ffmpeg, background=Colors.BG_PANEL, height=28).pack(
            side="right"
        )

        tvai_model_row = self._card_menu_row(
            tvai_inner, "tvai_model", "model", {model: model for model in _TVAI_MODELS}, width=100
        )
        tvai_scale_row = self._card_menu_row(tvai_inner, "tvai_scale", "scale", _scales(1, 2, 4), width=80)

        tvai_denoise_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        add_setting_label(tvai_denoise_row, "tvai_denoise")
        self._tvai_denoise_switch = ctk.CTkSwitch(
            tvai_denoise_row,
            text="",
            fg_color=Colors.BG_PANEL,
            progress_color=Colors.PRIMARY,
            button_color=Colors.TEXT_PRIMARY,
            width=42,
            command=self._on_modified,
        )
        self._tvai_denoise_switch.pack(side="right")

        tvai_workers_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        add_setting_label(tvai_workers_row, "workers", "tvai_workers")
        self._widgets["tvai_workers_val"] = create_slider_value_label(tvai_workers_row, Colors.BG_CARD)
        self._widgets["tvai_workers_val"].pack(side="right")
        self._widgets["tvai_workers"] = ctk.CTkSlider(
            tvai_workers_row, from_=1, to=8, number_of_steps=7,
            fg_color=Colors.BG_PANEL, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=120, command=self._on_workers_slider,
        )
        self._widgets["tvai_workers"].pack(side="right", padx=(0, 8))

        gap = dict(fill="x", pady=(0, 8))
        self._tvai_rows = [
            Row(tvai_path_row, gap, "simple"),
            Row(tvai_path_input_row, gap, "simple"),
            Row(tvai_model_row, gap, "expert"),
            Row(tvai_scale_row, gap, "simple"),
            Row(tvai_denoise_row, gap, "simple"),
            Row(tvai_workers_row, dict(fill="x"), "expert"),
        ]

        self._rtx_frame = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)
        rtx_inner = ctk.CTkFrame(self._rtx_frame, fg_color="transparent")
        rtx_inner.pack(fill="x", padx=12, pady=12)
        rtx_rows = (
            self._card_menu_row(rtx_inner, "rtx_scale", "rtx_scale", _scales(2, 4), width=80),
            self._card_menu_row(rtx_inner, "rtx_quality", "rtx_quality", _rtx_levels(get_args(RtxQualityName)), width=100),
            self._card_menu_row(rtx_inner, "rtx_denoise", "rtx_denoise", _rtx_levels(get_args(RtxLevelName)), width=100),
            self._card_menu_row(rtx_inner, "rtx_deblur", "rtx_deblur", _rtx_levels(get_args(RtxLevelName)), width=100),
        )
        for row in rtx_rows[:-1]:
            row.pack(fill="x", pady=(0, 8))
        rtx_rows[-1].pack(fill="x")

    def _card_menu_row(self, master, key: str, label_key: str, options: dict[str, str], *, width: int):
        row = ctk.CTkFrame(master, fg_color="transparent")
        add_setting_label(row, label_key, key)
        self._widgets[key] = option_menu(
            row, options, lambda _value: self._on_modified(), width=width, background=Colors.BG_PANEL
        )
        self._widgets[key].pack(side="right")
        return row

    def _browse_tvai_ffmpeg(self):
        filepath = filedialog.askopenfilename(
            title=t("dialog_select_tvai_ffmpeg"),
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            initialdir=str(PureWindowsPath(TVAI_DEFAULT_FFMPEG_PATH).parent),
        )
        if filepath:
            set_entry_text(self._widgets["tvai_ffmpeg_path"], filepath)
            self._on_modified()

    def _on_workers_slider(self, value: float) -> None:
        self._widgets["tvai_workers_val"].configure(text=str(int(value)))
        self._on_modified()

    def _on_engine_picked(self) -> None:
        self._show_engine_options()
        self._on_modified()

    def _show_engine_options(self):
        engine = self._engine.get()
        self._tvai_frame.pack_forget()
        self._rtx_frame.pack_forget()
        if engine == "tvai":
            self._tvai_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        elif engine == "rtx-super-res":
            self._rtx_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

    def _available(self, engine: str) -> bool:
        if engine not in secondary_restorations_for(self._vendor):
            return False
        if engine == "unet-4x" and not self._unet4x_available:
            return False
        return not self._ltx_only or engine in LTX_SECONDARY_RESTORATIONS

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        state = "normal" if enabled else "disabled"
        for engine, radio in self._engine_radios.items():
            available = self._available(engine)
            radio.configure(state=state if available else "disabled")
            if not available and self._engine.get() == engine:
                self._engine.set("none")
                self._show_engine_options()
        self._tvai_denoise_switch.configure(state=state)
        set_widgets_enabled(
            (self._widgets[key] for key in (
                "tvai_ffmpeg_path", "tvai_model", "tvai_scale", "tvai_workers", "rtx_quality", "rtx_denoise", "rtx_deblur"
            )),
            enabled,
        )
        self._widgets["rtx_scale"].configure(state="disabled" if self._ltx_only else state)

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        """The TVAI model and worker count are expert settings."""
        layout_rows(self._tvai_rows, level, standard=standard, ltx=ltx)
        return True

    def set_gpu_support(self, vendor: AcceleratorVendor) -> None:
        self._vendor = vendor
        self.set_enabled(self._enabled)

    def set_ltx_only(self, ltx_only: bool) -> None:
        """LTX without auto split: only RTX Super Res, always 2x."""
        self._ltx_only = ltx_only
        if ltx_only:
            self._ltx_only_note.pack(fill="x", after=self._engines_frame)
        else:
            self._ltx_only_note.pack_forget()
        self.set_enabled(self._enabled)

    def apply(self, preset):
        self._engine.set(preset.secondary_restoration)
        set_entry_text(self._widgets["tvai_ffmpeg_path"], preset.tvai_ffmpeg_path)
        self._widgets["tvai_model"].set_value(preset.tvai_model)
        self._widgets["tvai_scale"].set_value(str(preset.tvai_scale))
        self._widgets["tvai_workers"].set(preset.tvai_workers)
        self._widgets["tvai_workers_val"].configure(text=str(preset.tvai_workers))
        if preset.tvai_denoise:
            self._tvai_denoise_switch.select()
        else:
            self._tvai_denoise_switch.deselect()
        self._widgets["rtx_scale"].set_value(str(preset.rtx_scale))
        for key in ("rtx_quality", "rtx_denoise", "rtx_deblur"):
            self._widgets[key].set_value(getattr(preset, key))
        self._show_engine_options()
        self.set_enabled(self._enabled)

    def collect(self) -> dict:
        return {
            "secondary_restoration": self._engine.get(),
            "tvai_ffmpeg_path": self._widgets["tvai_ffmpeg_path"].get(),
            "tvai_model": self._widgets["tvai_model"].get_value(),
            "tvai_scale": int(self._widgets["tvai_scale"].get_value()),
            "tvai_workers": int(self._widgets["tvai_workers"].get()),
            "tvai_denoise": self._tvai_denoise_switch.get() == 1,
            "rtx_scale": int(self._widgets["rtx_scale"].get_value()),
            **{key: self._widgets[key].get_value() for key in ("rtx_quality", "rtx_denoise", "rtx_deblur")},
        }
