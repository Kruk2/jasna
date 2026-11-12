"""Whole-video upscaling settings section."""

import sys
from typing import get_args

import customtkinter as ctk

from jasna.accelerator import AcceleratorVendor
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    SLIDER_WIDTH,
    Row,
    ValueOptionMenu,
    add_setting_label,
    create_slider_value_label,
    layout_rows,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.session_config import UPSCALE_HEIGHTS, Dlss5PresetName, Dlss5StyleName

_RTX_LEVELS = ("low", "medium", "high", "ultra")


def offered_upscale_models(vendor: AcceleratorVendor) -> tuple[str, ...]:
    """The AI networks run on NVIDIA and AMD; RTX Super Res only on NVIDIA, DLSS 5 only on
    NVIDIA under Windows."""
    if vendor is AcceleratorVendor.NVIDIA:
        dlss5 = ("dlss5",) if sys.platform == "win32" else ()
        return ("none", "span", "realplksr", "rtx-super-res", *dlss5)
    if vendor is AcceleratorVendor.AMD:
        return ("none", "span", "realplksr")
    return ("none",)


def _menu(row, options: dict[str, str], command, width: int) -> ValueOptionMenu:
    menu = ValueOptionMenu(
        row, options=options, command=command,
        fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
        button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
        dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY, width=width,
    )
    menu.pack(side="right")
    return menu


class FrameUpscaleSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified
        self._vendor = AcceleratorVendor.NVIDIA
        self._level = "simple"

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")
        heading = ctk.CTkLabel(
            inner, text=t("upscale_whole_video_heading"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"), anchor="w",
        )

        model_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(model_row, "upscale_model")
        self._widgets["upscale_model"] = _menu(
            model_row,
            {
                "none": t("upscale_off"),
                "span": t("upscale_span"),
                "realplksr": t("upscale_realplksr"),
                "rtx-super-res": t("upscale_rtx"),
                "dlss5": t("upscale_dlss5"),
            },
            self._on_model_changed,
            width=220,
        )

        self._card = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)
        card = ctk.CTkFrame(self._card, fg_color="transparent")
        card.pack(fill="x", padx=12, pady=12)

        height_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(height_row, "upscale_height")
        self._widgets["upscale_height"] = _menu(
            height_row,
            {str(height): t(f"upscale_height_{height}") for height in UPSCALE_HEIGHTS},
            lambda _value: self._on_modified(),
            width=140,
        )

        strength_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(strength_row, "upscale_strength")
        self._widgets["upscale_strength_val"] = create_slider_value_label(strength_row, Colors.BG_CARD)
        self._widgets["upscale_strength_val"].pack(side="right")
        self._widgets["upscale_strength"] = ctk.CTkSlider(
            strength_row, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_PANEL, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=self._on_strength_changed,
        )
        self._widgets["upscale_strength"].pack(side="right", padx=(0, 8))

        levels = {level: t(f"rtx_level_{level}") for level in _RTX_LEVELS}
        rtx_rows = []
        for key, options in (
            ("upscale_rtx_quality", levels),
            ("upscale_rtx_denoise", {"none": t("rtx_level_none"), **levels}),
            ("upscale_rtx_deblur", {"none": t("rtx_level_none"), **levels}),
        ):
            row = ctk.CTkFrame(card, fg_color="transparent")
            add_setting_label(row, key)
            self._widgets[key] = _menu(row, options, lambda _value: self._on_modified(), width=140)
            rtx_rows.append(row)
        self._rtx_rows = tuple(rtx_rows)

        dlss5_preset_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(dlss5_preset_row, "upscale_dlss5_preset")
        self._widgets["upscale_dlss5_preset"] = _menu(
            dlss5_preset_row,
            {preset: t("dlss5_preset_default") if preset == "default" else preset.upper()
             for preset in get_args(Dlss5PresetName)},
            lambda _value: self._on_modified(),
            width=140,
        )
        dlss5_style_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(dlss5_style_row, "upscale_dlss5_style")
        self._widgets["upscale_dlss5_style"] = _menu(
            dlss5_style_row,
            {style: t(f"dlss5_style_{style}") for style in get_args(Dlss5StyleName)},
            lambda _value: self._on_modified(),
            width=140,
        )
        dlss5_intensity_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(dlss5_intensity_row, "upscale_dlss5_intensity")
        self._widgets["upscale_dlss5_intensity_val"] = create_slider_value_label(dlss5_intensity_row, Colors.BG_CARD)
        self._widgets["upscale_dlss5_intensity_val"].pack(side="right")
        self._widgets["upscale_dlss5_intensity"] = ctk.CTkSlider(
            dlss5_intensity_row, from_=0.0, to=2.0, number_of_steps=40,
            fg_color=Colors.BG_PANEL, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=self._on_dlss5_intensity_changed,
        )
        self._widgets["upscale_dlss5_intensity"].pack(side="right", padx=(0, 8))
        self._dlss5_rows = (dlss5_preset_row, dlss5_style_row, dlss5_intensity_row)

        gap = dict(fill="x", pady=(0, 8))
        self._card_rows = [
            Row(height_row, gap, "simple"),
            Row(strength_row, gap, "advanced"),
            *(Row(row, gap, "advanced") for row in rtx_rows),
            *(Row(row, gap, "advanced") for row in self._dlss5_rows),
        ]
        self._rows = [
            Row(heading, dict(fill="x", pady=(Sizing.PADDING_SMALL, Sizing.PADDING_SMALL)), "simple"),
            Row(model_row, dict(fill="x", pady=(0, Sizing.PADDING_SMALL)), "simple"),
        ]

    def _on_model_changed(self, _value: str) -> None:
        self._layout_card()
        self._on_modified()

    def _on_strength_changed(self, value: float) -> None:
        self._widgets["upscale_strength_val"].configure(text=f"{value:.2f}")
        self._on_modified()

    def _on_dlss5_intensity_changed(self, value: float) -> None:
        self._widgets["upscale_dlss5_intensity_val"].configure(text=f"{value:.2f}")
        self._on_modified()

    def _layout_card(self) -> None:
        model = self._widgets["upscale_model"].get_value()
        if model == "none":
            self._card.pack_forget()
            return
        hidden = (
            *(() if model == "rtx-super-res" else self._rtx_rows),
            *(() if model == "dlss5" else self._dlss5_rows),
        )
        layout_rows(self._card_rows, self._level, standard=True, ltx=True, hidden=hidden)
        self._card.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        self._level = level
        self._card.pack_forget()
        layout_rows(self._rows, level, standard=standard, ltx=ltx)
        self._layout_card()
        return True

    def set_gpu_support(self, vendor: AcceleratorVendor) -> None:
        self._vendor = vendor
        menu = self._widgets["upscale_model"]
        offered = offered_upscale_models(vendor)
        menu.offer(offered)
        if menu.get_value() not in offered:
            menu.set_value("none")
            self._layout_card()

    def apply(self, preset) -> None:
        model = preset.upscale_model if preset.upscale_model in offered_upscale_models(self._vendor) else "none"
        self._widgets["upscale_model"].set_value(model)
        self._widgets["upscale_height"].set_value(str(preset.upscale_height))
        self._widgets["upscale_strength"].set(preset.upscale_strength)
        self._widgets["upscale_strength_val"].configure(text=f"{preset.upscale_strength:.2f}")
        self._widgets["upscale_rtx_quality"].set_value(preset.upscale_rtx_quality)
        self._widgets["upscale_rtx_denoise"].set_value(preset.upscale_rtx_denoise)
        self._widgets["upscale_rtx_deblur"].set_value(preset.upscale_rtx_deblur)
        self._widgets["upscale_dlss5_preset"].set_value(preset.upscale_dlss5_preset)
        self._widgets["upscale_dlss5_style"].set_value(preset.upscale_dlss5_style)
        self._widgets["upscale_dlss5_intensity"].set(preset.upscale_dlss5_intensity)
        self._widgets["upscale_dlss5_intensity_val"].configure(text=f"{preset.upscale_dlss5_intensity:.2f}")
        self._layout_card()

    def collect(self) -> dict:
        return {
            "upscale_model": self._widgets["upscale_model"].get_value(),
            "upscale_height": int(self._widgets["upscale_height"].get_value()),
            "upscale_strength": round(float(self._widgets["upscale_strength"].get()), 2),
            "upscale_rtx_quality": self._widgets["upscale_rtx_quality"].get_value(),
            "upscale_rtx_denoise": self._widgets["upscale_rtx_denoise"].get_value(),
            "upscale_rtx_deblur": self._widgets["upscale_rtx_deblur"].get_value(),
            "upscale_dlss5_preset": self._widgets["upscale_dlss5_preset"].get_value(),
            "upscale_dlss5_style": self._widgets["upscale_dlss5_style"].get_value(),
            "upscale_dlss5_intensity": round(float(self._widgets["upscale_dlss5_intensity"].get()), 2),
        }
