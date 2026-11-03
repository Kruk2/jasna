"""Processing settings section: clip length, overlap, denoise, video decoder and GPU precision."""

from typing import get_args

import customtkinter as ctk

from jasna.accelerator import AcceleratorVendor
from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    Row,
    add_setting_label,
    ValueOptionMenu,
    SLIDER_WIDTH,
    create_slider_value_label,
    layout_rows,
)
from jasna.gui.theme import Colors, Sizing
from jasna.session_config import DecodeBackendName

# The sub-engines no longer depend on the clip length, so a long clip costs
# activation memory only — high values stay usable on a large GPU.
MIN_CLIP_SIZE = 10
MAX_CLIP_SIZE = 720
CLIP_SIZE_STEP = 10


def decode_backends(vendor: AcceleratorVendor | None) -> tuple[DecodeBackendName, ...]:
    """The decoders this GPU can use (all until it is known): VALI is NVIDIA only, and on AMD the
    FFmpeg hardware decoder is what Auto uses anyway."""
    if vendor in (None, AcceleratorVendor.NVIDIA):
        return get_args(DecodeBackendName)
    return ("auto", "pyav-sw")


class ProcessingSection:
    def __init__(self, parent, widgets: dict, on_modified, on_max_clip_size_change):
        self._widgets = widgets
        self._on_modified = on_modified
        self._on_max_clip_size_change = on_max_clip_size_change

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        overlap_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(overlap_row, "temporal_overlap")
        self._widgets["temporal_overlap_val"] = create_slider_value_label(overlap_row, "8", Colors.BG_PANEL)
        self._widgets["temporal_overlap_val"].pack(side="right")
        self._widgets["temporal_overlap"] = ctk.CTkSlider(
            overlap_row, from_=0, to=30, number_of_steps=30,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=SLIDER_WIDTH, command=self._on_temporal_overlap_slider,
        )
        self._widgets["temporal_overlap"].pack(side="right", padx=(0, 8))
        self._widgets["temporal_overlap"].set(8)

        denoise_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(denoise_row, "denoise_strength")
        self._widgets["denoise_strength"] = ValueOptionMenu(
            denoise_row,
            options={
                "none": t("denoise_none"),
                "low": t("denoise_low"),
                "medium": t("denoise_medium"),
                "high": t("denoise_high"),
            },
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=120,
        )
        self._widgets["denoise_strength"].pack(side="right")
        self._widgets["denoise_strength"].set_value("none")

        clip_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(clip_row, "max_clip_size")
        self._widgets["max_clip_size_val"] = create_slider_value_label(clip_row, "90", Colors.BG_PANEL)
        self._widgets["max_clip_size_val"].pack(side="right")
        self._widgets["max_clip_size"] = ctk.CTkSlider(
            clip_row,
            from_=MIN_CLIP_SIZE,
            to=MAX_CLIP_SIZE,
            number_of_steps=(MAX_CLIP_SIZE - MIN_CLIP_SIZE) // CLIP_SIZE_STEP,
            command=self._on_max_clip_size_slider,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY, width=SLIDER_WIDTH,
        )
        self._widgets["max_clip_size"].pack(side="right", padx=(0, 8))
        self._widgets["max_clip_size"].set(90)

        decode_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(decode_row, "decode_backend")
        self._decode_backends = decode_backends(None)
        self._widgets["decode_backend"] = ValueOptionMenu(
            decode_row,
            options={backend: t(f"decode_backend_{backend.replace('-', '_')}") for backend in self._decode_backends},
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=140,
        )
        self._widgets["decode_backend"].pack(side="right")
        self._widgets["decode_backend"].set_value("auto")

        self._toggles_row = ctk.CTkFrame(inner, fg_color="transparent")
        fp16_frame = self._switch_card(self._toggles_row, "fp16_mode")
        compile_frame = self._switch_card(self._toggles_row, "compile_basicvsrpp")

        row_gap = dict(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._rows = [
            Row(overlap_row, row_gap, "advanced", "standard"),
            Row(denoise_row, row_gap, "advanced", "standard"),
            Row(clip_row, row_gap, "advanced", "standard"),
            Row(decode_row, row_gap, "expert"),
            Row(self._toggles_row, row_gap, "expert"),
        ]
        self._toggles = [
            Row(fp16_frame, dict(side="left", fill="x", expand=True, padx=(0, 4)), "expert"),
            Row(compile_frame, dict(side="right", fill="x", expand=True, padx=(4, 0)), "expert", "standard"),
        ]

    def _switch_card(self, master, key: str):
        card = ctk.CTkFrame(master, fg_color=Colors.BG_CARD, corner_radius=6)
        add_setting_label(card, key, in_card=True)
        self._widgets[key] = CompactSwitch(card, self._on_modified, Colors.BG_CARD)
        self._widgets[key].pack(side="right", padx=12, pady=8)
        self._widgets[key].select()
        return card

    def _set_decode_backend(self, backend: str) -> None:
        self._widgets["decode_backend"].set_value(backend if backend in self._decode_backends else "auto")

    def set_gpu_support(self, vendor: AcceleratorVendor) -> None:
        self._decode_backends = decode_backends(vendor)
        self._widgets["decode_backend"].offer(self._decode_backends)
        self._set_decode_backend(self._widgets["decode_backend"].get_value())

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        any_toggle = layout_rows(self._toggles, level, standard=standard, ltx=ltx)
        hidden = () if any_toggle else (self._toggles_row,)
        return layout_rows(self._rows, level, standard=standard, ltx=ltx, hidden=hidden)

    def _on_temporal_overlap_slider(self, value: float):
        self._widgets["temporal_overlap_val"].configure(text=str(int(value)))
        self._on_modified()

    def _on_max_clip_size_slider(self, value: float):
        max_clip_size = int(value)
        self._widgets["max_clip_size_val"].configure(text=str(max_clip_size))
        self._on_modified()
        self._on_max_clip_size_change(max_clip_size)

    def apply(self, preset):
        self._widgets["temporal_overlap"].set(preset.temporal_overlap)
        self._widgets["temporal_overlap_val"].configure(text=str(preset.temporal_overlap))
        self._widgets["denoise_strength"].set_value(preset.denoise_strength)
        self._widgets["max_clip_size"].set(preset.max_clip_size)
        self._widgets["max_clip_size_val"].configure(text=str(preset.max_clip_size))
        self._set_decode_backend(preset.decode_backend)
        for key in ("fp16_mode", "compile_basicvsrpp"):
            if getattr(preset, key):
                self._widgets[key].select()
            else:
                self._widgets[key].deselect()

    def collect(self) -> dict:
        return {
            "temporal_overlap": int(self._widgets["temporal_overlap"].get()),
            "denoise_strength": self._widgets["denoise_strength"].get_value(),
            "max_clip_size": int(self._widgets["max_clip_size"].get()),
            "decode_backend": self._widgets["decode_backend"].get_value(),
            "fp16_mode": self._widgets["fp16_mode"].get() == 1,
            "compile_basicvsrpp": self._widgets["compile_basicvsrpp"].get() == 1,
        }
