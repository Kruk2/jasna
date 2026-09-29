"""Restoration model choice (BasicVSR++ or LTX) and the LTX options."""

import random
from typing import Callable

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch, NativeIconButton
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import add_setting_label, get_tooltip
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.session_config import LTX_DEFAULT_SEED

_MODELS = ("basicvsrpp", "ltx")


def ltx_unavailable_reason(*, installed: bool, nvidia: bool | None) -> str | None:
    """Locale key saying why LTX cannot be picked; ``nvidia`` is None until the GPU is known."""
    if nvidia is False:
        return "model_ltx_needs_nvidia"
    if not installed:
        return "model_ltx_not_installed"
    return None


def parse_seed(text: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return LTX_DEFAULT_SEED


class _ModelCard(ctk.CTkFrame):
    def __init__(self, master, variable: ctk.StringVar, value: str, on_select: Callable[[str], None]):
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=8, border_width=2, border_color=Colors.BG_CARD)
        self._value = value
        self._on_select = on_select
        self._enabled = True
        self.radio = ctk.CTkRadioButton(
            self,
            text=t(f"model_{value}"),
            variable=variable,
            value=value,
            command=lambda: on_select(value),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
        )
        self.radio.pack(anchor="w", padx=12, pady=(10, 2))
        self.description = ctk.CTkLabel(
            self,
            text=t(f"model_{value}_description"),
            text_color=Colors.STATUS_PENDING,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            anchor="w",
            justify="left",
            wraplength=260,
        )
        self.description.pack(fill="x", padx=(40, 12), pady=(0, 10))
        for widget in (self, self.description):
            widget.bind("<Button-1>", self._clicked)

    def _clicked(self, _event) -> None:
        if self._enabled:
            self.radio.invoke()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.radio.configure(state="normal" if enabled else "disabled")

    def set_selected(self, selected: bool) -> None:
        self.configure(border_color=Colors.PRIMARY if selected else Colors.BG_CARD)


class RestorationModelSection:
    def __init__(self, parent, widgets: dict, on_modified, on_model_changed, *, ltx_installed: bool):
        self._widgets = widgets
        self._on_modified = on_modified
        self._on_model_changed = on_model_changed
        self._ltx_installed = ltx_installed
        self._nvidia: bool | None = None
        self._enabled = True

        section = CollapsibleSection(parent, t("section_restoration_model"), expanded=True)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        self._model = self._widgets["restoration_model"] = ctk.StringVar(value="basicvsrpp")
        cards = ctk.CTkFrame(inner, fg_color="transparent")
        cards.pack(fill="x")
        cards.grid_columnconfigure((0, 1), weight=1, uniform="model")
        self._cards = {}
        for column, value in enumerate(_MODELS):
            card = _ModelCard(cards, self._model, value, self._on_card_selected)
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
            self._cards[value] = card

        self._ltx_options = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)
        seed_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        seed_row.pack(fill="x")
        add_setting_label(seed_row, "ltx_seed", in_card=True)
        self._new_seed_btn = NativeIconButton(
            seed_row,
            "reset",
            16,
            Colors.TEXT_PRIMARY,
            Colors.BG_CARD,
            Colors.BORDER_LIGHT,
            Colors.BORDER_LIGHT,
            self._on_new_seed,
            28,
            28,
        )
        self._new_seed_btn.pack(side="right", padx=(4, 12), pady=8)
        Tooltip(self._new_seed_btn, get_tooltip("ltx_new_seed"))
        self._widgets["ltx_seed"] = ctk.CTkEntry(
            seed_row, width=120, fg_color=Colors.BG_PANEL, text_color=Colors.TEXT_PRIMARY, border_color=Colors.BORDER_LIGHT,
        )
        self._widgets["ltx_seed"].pack(side="right", pady=8)
        self._widgets["ltx_seed"].bind("<KeyRelease>", lambda _event: self._on_modified())

        self._fast_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        add_setting_label(self._fast_row, "ltx_fast", in_card=True)
        self._widgets["ltx_fast"] = CompactSwitch(self._fast_row, self._on_modified, Colors.BG_CARD)
        self._widgets["ltx_fast"].pack(side="right", padx=12, pady=8)

        large_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        large_row.pack(fill="x")
        add_setting_label(large_row, "ltx_large_canvas", in_card=True)
        self._widgets["ltx_large_canvas"] = CompactSwitch(large_row, self._on_modified, Colors.BG_CARD)
        self._widgets["ltx_large_canvas"].pack(side="right", padx=12, pady=8)
        self._large_row = large_row

        self._refresh_ltx_availability()

    def _on_card_selected(self, value: str) -> None:
        self._show_model(value)
        self._on_modified()

    def _show_model(self, value: str) -> None:
        self._model.set(value)
        for name, card in self._cards.items():
            card.set_selected(name == value)
        if value == "ltx":
            self._ltx_options.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        else:
            self._ltx_options.pack_forget()
        self._on_model_changed(value)

    def _on_new_seed(self) -> None:
        self._set_seed(random.randrange(1, 2**31))
        self._on_modified()

    def _set_seed(self, seed: int) -> None:
        self._widgets["ltx_seed"].delete(0, "end")
        self._widgets["ltx_seed"].insert(0, str(seed))

    def unavailable_reason(self) -> str | None:
        return ltx_unavailable_reason(installed=self._ltx_installed, nvidia=self._nvidia)

    def _refresh_ltx_availability(self) -> None:
        reason = self.unavailable_reason()
        card = self._cards["ltx"]
        card.set_enabled(self._enabled and reason is None)
        card.description.configure(
            text=t(reason) if reason else t("model_ltx_description"),
            text_color=Colors.STATUS_WARNING if reason else Colors.STATUS_PENDING,
        )
        if reason is not None and self._model.get() == "ltx":
            self._show_model("basicvsrpp")

    def set_gpu_support(self, *, nvidia: bool, blackwell: bool) -> None:
        self._nvidia = nvidia
        if blackwell:
            self._fast_row.pack(fill="x", before=self._large_row)
        else:
            self._widgets["ltx_fast"].deselect()
            self._fast_row.pack_forget()
        self._refresh_ltx_availability()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._cards["basicvsrpp"].set_enabled(enabled)
        self._new_seed_btn.configure(state="normal" if enabled else "disabled")
        self._refresh_ltx_availability()

    def apply(self, preset) -> None:
        self._set_seed(preset.ltx_seed)
        for key, selected in (("ltx_fast", preset.ltx_fast), ("ltx_large_canvas", preset.ltx_large_canvas)):
            if selected:
                self._widgets[key].select()
            else:
                self._widgets[key].deselect()
        self._show_model(preset.restoration_model)
        self._refresh_ltx_availability()

    def collect(self) -> dict:
        return {
            "restoration_model": self._widgets["restoration_model"].get(),
            "ltx_seed": parse_seed(self._widgets["ltx_seed"].get()),
            "ltx_fast": self._widgets["ltx_fast"].get() == 1,
            "ltx_large_canvas": self._widgets["ltx_large_canvas"].get() == 1,
        }
