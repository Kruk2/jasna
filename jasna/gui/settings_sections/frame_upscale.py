"""Whole-video upscaling settings section."""

import sys
from typing import get_args

import customtkinter as ctk

from jasna.accelerator import AcceleratorVendor
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
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.session_config import UPSCALE_HEIGHTS, Dlss5PresetName, Dlss5StyleName, RtxLevelName, RtxQualityName
from jasna.session_factory import upscale_models_for

_RTX_KEYS = ("upscale_rtx_quality", "upscale_rtx_denoise", "upscale_rtx_deblur")


class FrameUpscaleSection:
    def __init__(self, parent, on_modified):
        self._widgets = {}
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
        self._widgets["upscale_model"] = self._menu(
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
        self._widgets["upscale_height"] = self._menu(
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

        rtx_rows = []
        for key, levels in zip(_RTX_KEYS, (get_args(RtxQualityName), get_args(RtxLevelName), get_args(RtxLevelName))):
            row = ctk.CTkFrame(card, fg_color="transparent")
            add_setting_label(row, key)
            self._widgets[key] = self._menu(
                row, {level: t(f"rtx_level_{level}") for level in levels}, lambda _value: self._on_modified(), width=140
            )
            rtx_rows.append(row)
        self._rtx_rows = tuple(rtx_rows)

        dlss5_preset_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(dlss5_preset_row, "upscale_dlss5_preset")
        self._widgets["upscale_dlss5_preset"] = self._menu(
            dlss5_preset_row,
            {preset: t("dlss5_preset_default") if preset == "default" else preset.upper()
             for preset in get_args(Dlss5PresetName)},
            lambda _value: self._on_modified(),
            width=140,
        )
        dlss5_style_row = ctk.CTkFrame(card, fg_color="transparent")
        add_setting_label(dlss5_style_row, "upscale_dlss5_style")
        self._widgets["upscale_dlss5_style"] = self._menu(
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

    @staticmethod
    def _menu(row, options: dict[str, str], command, *, width: int):
        menu = option_menu(row, options, command, width=width, background=Colors.BG_PANEL)
        menu.pack(side="right")
        return menu

    def _offered_models(self) -> tuple[str, ...]:
        return upscale_models_for(self._vendor, sys.platform)

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

    def set_enabled(self, enabled: bool) -> None:
        set_widgets_enabled(
            (self._widgets[key] for key in (
                "upscale_model", "upscale_height", "upscale_strength", *_RTX_KEYS,
                "upscale_dlss5_preset", "upscale_dlss5_style", "upscale_dlss5_intensity",
            )),
            enabled,
        )

    def set_gpu_support(self, vendor: AcceleratorVendor) -> None:
        self._vendor = vendor
        menu = self._widgets["upscale_model"]
        offered = self._offered_models()
        menu.offer(offered)
        if menu.get_value() not in offered:
            menu.set_value("none")
            self._layout_card()

    def apply(self, preset) -> None:
        model = preset.upscale_model if preset.upscale_model in self._offered_models() else "none"
        self._widgets["upscale_model"].set_value(model)
        self._widgets["upscale_height"].set_value(str(preset.upscale_height))
        self._widgets["upscale_strength"].set(preset.upscale_strength)
        self._widgets["upscale_strength_val"].configure(text=f"{preset.upscale_strength:.2f}")
        for key in _RTX_KEYS:
            self._widgets[key].set_value(getattr(preset, key))
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
            **{key: self._widgets[key].get_value() for key in _RTX_KEYS},
            "upscale_dlss5_preset": self._widgets["upscale_dlss5_preset"].get_value(),
            "upscale_dlss5_style": self._widgets["upscale_dlss5_style"].get_value(),
            "upscale_dlss5_intensity": round(float(self._widgets["upscale_dlss5_intensity"].get()), 2),
        }
