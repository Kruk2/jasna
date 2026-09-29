"""Restoration model choice (BasicVSR++ or LTX) and the LTX options."""

import random
from typing import Callable

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch, NativeIconButton
from jasna.gui.locales import t
from jasna.gui.ltx_models import LtxModels, card_unavailable_reason, run_unavailable_reason, trial_only
from jasna.gui.settings_sections.widgets import ValueOptionMenu, add_setting_label, get_tooltip
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.ltx.model_files import LTX_MODELS
from jasna.session_config import LTX_DEFAULT_MODEL, LTX_DEFAULT_SEED

_MODELS = ("basicvsrpp", "ltx")


def parse_seed(text: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return LTX_DEFAULT_SEED


class _ModelCard(ctk.CTkFrame):
    def __init__(self, master, variable: ctk.StringVar, value: str, on_select: Callable[[str], None]):
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=8, border_width=2, border_color=Colors.BG_CARD)
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
            wraplength=190,
        )
        self.description.pack(fill="x", padx=(40, 12), pady=(0, 10))
        for widget in (self, self.description):
            widget.bind("<Button-1>", self._clicked)
        for widget in (self.radio, self.description):
            Tooltip(widget, get_tooltip(f"model_{value}"))

    def _clicked(self, _event) -> None:
        if self._enabled:
            self.radio.invoke()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.radio.configure(state="normal" if enabled else "disabled")

    def set_selected(self, selected: bool) -> None:
        self.configure(border_color=Colors.PRIMARY if selected else Colors.BG_CARD)


class RestorationModelSection:
    def __init__(self, parent, widgets: dict, on_modified, on_model_changed, *, ltx_models: LtxModels):
        self._widgets = widgets
        self._on_modified = on_modified
        self._on_model_changed = on_model_changed
        self._ltx_models = ltx_models
        self._nvidia: bool | None = None
        self._enabled = True

        section = CollapsibleSection(parent, t("section_restoration_model"), expanded=True)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        self._model = self._widgets["restoration_model"] = ctk.StringVar(value="basicvsrpp")
        self._cards_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self._cards_frame.pack(fill="x")
        self._cards_frame.grid_columnconfigure((0, 1), weight=1, uniform="model")
        self._cards = {}
        for column, value in enumerate(_MODELS):
            card = _ModelCard(self._cards_frame, self._model, value, self._on_card_selected)
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
            self._cards[value] = card

        self._download_row = ctk.CTkFrame(inner, fg_color="transparent")
        self._download_label = ctk.CTkLabel(
            self._download_row, text="", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_SMALL), anchor="w"
        )
        self._download_label.pack(fill="x")
        self._download_bar = ctk.CTkProgressBar(self._download_row, progress_color=Colors.PRIMARY, fg_color=Colors.BG_CARD)
        self._download_bar.pack(fill="x", pady=(2, 0))
        for widget in (self._download_label, self._download_bar):
            Tooltip(widget, get_tooltip("ltx_download"))

        self._ltx_options = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)
        model_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        model_row.pack(fill="x")
        add_setting_label(model_row, "ltx_model", in_card=True)
        self._ltx_model = self._widgets["ltx_model"] = ValueOptionMenu(
            model_row,
            options={model: t(f"ltx_model_{model}") for model in LTX_MODELS},
            command=self._on_ltx_model_selected,
            fg_color=Colors.BG_PANEL,
            button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=260,
        )
        self._ltx_model.pack(side="right", padx=12, pady=8)
        Tooltip(self._ltx_model, get_tooltip("ltx_model"))
        self._set_ltx_model(LTX_DEFAULT_MODEL)
        self._model_row = model_row
        self._install_status = ctk.CTkLabel(
            self._ltx_options, text="", text_color=Colors.STATUS_WARNING, font=(Fonts.FAMILY, Fonts.SIZE_SMALL), anchor="e"
        )
        Tooltip(self._install_status, get_tooltip("ltx_download"))

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
        Tooltip(self._widgets["ltx_seed"], get_tooltip("ltx_seed"))

        self._fast_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        add_setting_label(self._fast_row, "ltx_fast", in_card=True)
        self._widgets["ltx_fast"] = CompactSwitch(self._fast_row, self._on_fast_toggled, Colors.BG_CARD)
        self._widgets["ltx_fast"].pack(side="right", padx=12, pady=8)
        Tooltip(self._widgets["ltx_fast"], get_tooltip("ltx_fast"))

        large_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        large_row.pack(fill="x")
        add_setting_label(large_row, "ltx_large_canvas", in_card=True)
        self._widgets["ltx_large_canvas"] = CompactSwitch(large_row, self._on_modified, Colors.BG_CARD)
        self._widgets["ltx_large_canvas"].pack(side="right", padx=12, pady=8)
        Tooltip(self._widgets["ltx_large_canvas"], get_tooltip("ltx_large_canvas"))
        self._large_row = large_row

        trial_row = ctk.CTkFrame(self._ltx_options, fg_color="transparent")
        trial_row.pack(fill="x")
        add_setting_label(trial_row, "ltx_trial", in_card=True)
        self._widgets["ltx_trial"] = CompactSwitch(trial_row, self._on_trial_toggled, Colors.BG_CARD)
        self._widgets["ltx_trial"].pack(side="right", padx=12, pady=8)
        Tooltip(self._widgets["ltx_trial"], get_tooltip("ltx_trial"))
        self._trial_status = ctk.CTkLabel(
            self._ltx_options, text="", text_color=Colors.STATUS_WARNING, font=(Fonts.FAMILY, Fonts.SIZE_SMALL), anchor="w"
        )
        self._trial_row = trial_row

        self._notice = ctk.CTkLabel(
            self._ltx_options, text="", text_color=Colors.STATUS_WARNING, font=(Fonts.FAMILY, Fonts.SIZE_SMALL), anchor="w"
        )

        self._refresh_widgets()

    def _fast(self) -> bool:
        return self._widgets["ltx_fast"].get() == 1

    def _set_fast(self, fast: bool) -> None:
        if fast:
            self._widgets["ltx_fast"].select()
        else:
            self._widgets["ltx_fast"].deselect()

    def _trial(self) -> bool:
        return self._widgets["ltx_trial"].get() == 1

    def set_trial(self, trial: bool) -> None:
        if trial:
            self._widgets["ltx_trial"].select()
        else:
            self._widgets["ltx_trial"].deselect()
        self._refresh_widgets()

    def _set_ltx_model(self, model: str) -> None:
        self._ltx_model.set_value(model)
        self._accepted_ltx_model = self._ltx_model.get_value()

    def _show_notice(self, key: str | None) -> None:
        if key:
            self._notice.configure(text=t(key))
            self._notice.pack(fill="x", padx=12, pady=(0, 8))
        else:
            self._notice.pack_forget()

    def _ensure_variant(self, model: str, fast: bool) -> bool:
        """Make ``(model, fast)`` usable: download it when the user agrees, else switch to an
        installed variant with a notice. False when the user declined or nothing is installed."""
        self._show_notice(None)
        state = self._ltx_models.state
        if trial_only(state):
            self.set_trial(True)
        if self._trial() or state.installed(model, fast):
            return True
        if state.downloadable:
            return self._ltx_models.ensure(model, fast, on_ready=lambda: None)
        others = [(model, not fast)] + [(m, f) for m in LTX_MODELS if m != model for f in (fast, not fast)]
        for other_model, other_fast in others:
            if state.installed(other_model, other_fast):
                self._set_ltx_model(other_model)
                self._set_fast(other_fast)
                if other_model != model:
                    self._show_notice("model_ltx_variant_not_installed")
                else:
                    self._show_notice("model_ltx_fast_not_installed" if fast else "model_ltx_quality_not_installed")
                return True
        return False

    def _on_card_selected(self, value: str) -> None:
        if value == "ltx" and not self._ensure_variant(self._ltx_model.get_value(), self._fast()):
            self._show_model("basicvsrpp")
            return
        self._show_model(value)
        self._on_modified()

    def _on_ltx_model_selected(self, model: str) -> None:
        if self._ensure_variant(model, self._fast()):
            self._accepted_ltx_model = self._ltx_model.get_value()
        else:
            self._ltx_model.set_value(self._accepted_ltx_model)
        self._refresh_widgets()
        self._on_modified()

    def _on_fast_toggled(self) -> None:
        if not self._ensure_variant(self._ltx_model.get_value(), self._fast()):
            self._set_fast(not self._fast())
        self._refresh_widgets()
        self._on_modified()

    def _on_trial_toggled(self) -> None:
        if not self._trial() and not self._ensure_variant(self._ltx_model.get_value(), self._fast()):
            self.set_trial(True)
        self._refresh_widgets()
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
        self._refresh_widgets()

    def _on_new_seed(self) -> None:
        self._set_seed(random.randrange(1, 2**31))
        self._on_modified()

    def _set_seed(self, seed: int) -> None:
        self._widgets["ltx_seed"].delete(0, "end")
        self._widgets["ltx_seed"].insert(0, str(seed))

    def unavailable_reason(self) -> str | None:
        """Why the chosen LTX model cannot restore right now (None when it can)."""
        return run_unavailable_reason(
            self._ltx_models.state, self._ltx_model.get_value(), self._fast(), nvidia=self._nvidia, trial=self._trial()
        )

    def _refresh_widgets(self) -> None:
        state = self._ltx_models.state
        editable = self._enabled and not self._ltx_models.downloading
        reason = card_unavailable_reason(nvidia=self._nvidia)
        self._cards["basicvsrpp"].set_enabled(editable)
        ltx_card = self._cards["ltx"]
        ltx_card.set_enabled(editable and reason is None)
        ltx_card.description.configure(
            text=t(reason) if reason else t("model_ltx_description"),
            text_color=Colors.STATUS_WARNING if reason else Colors.STATUS_PENDING,
        )
        self._ltx_model.configure(state="normal" if editable else "disabled")
        self._new_seed_btn.configure(state="normal" if self._enabled else "disabled")
        locked = trial_only(state)
        if locked:
            self._widgets["ltx_trial"].select()
        self._widgets["ltx_trial"].configure(state="normal" if editable and not locked else "disabled")
        if self._trial():
            self._trial_status.configure(text=t("model_ltx_trial_only" if locked else "ltx_trial_notice"))
            self._trial_status.pack(fill="x", padx=12, pady=(0, 8), after=self._trial_row)
        else:
            self._trial_status.pack_forget()
        model, fast = self._ltx_model.get_value(), self._fast()
        if not self._trial() and not state.installed(model, fast) and state.downloadable:
            self._install_status.configure(text=t("model_ltx_download_needed", size=state.download_size(model, fast)))
            self._install_status.pack(fill="x", padx=12, pady=(0, 4), after=self._model_row)
        else:
            self._install_status.pack_forget()

    def _usable_model(self) -> str:
        """The chosen model, or BasicVSR++ when LTX cannot run; switches to the installed
        precision when only that one is there."""
        state = self._ltx_models.state
        if self._model.get() != "ltx" or card_unavailable_reason(nvidia=self._nvidia) is not None:
            return "basicvsrpp"
        model, fast = self._ltx_model.get_value(), self._fast()
        if not self._trial() and not state.usable(model, fast) and state.installed(model, not fast):
            self._set_fast(not fast)
        return "ltx"

    def _refresh_download(self) -> None:
        percent = self._ltx_models.percent
        if percent is None:
            self._download_row.pack_forget()
        else:
            self._download_label.configure(text=t("ltx_downloading", percent=percent))
            self._download_bar.set(percent / 100)
            self._download_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0), after=self._cards_frame)

    def refresh(self) -> None:
        """Show the current install state and download, falling back to a usable choice."""
        model = self._usable_model()
        if model != self._model.get():
            self._show_model(model)
        else:
            self._refresh_widgets()
        self._refresh_download()

    def set_gpu_support(self, *, nvidia: bool, blackwell: bool) -> None:
        self._nvidia = nvidia
        if blackwell:
            self._fast_row.pack(fill="x", before=self._large_row)
        else:
            self._set_fast(False)
            self._fast_row.pack_forget()
        self.refresh()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._refresh_widgets()

    def apply(self, preset) -> None:
        self._set_seed(preset.ltx_seed)
        self._set_fast(preset.ltx_fast)
        if preset.ltx_large_canvas:
            self._widgets["ltx_large_canvas"].select()
        else:
            self._widgets["ltx_large_canvas"].deselect()
        if preset.ltx_trial:
            self._widgets["ltx_trial"].select()
        else:
            self._widgets["ltx_trial"].deselect()
        self._set_ltx_model(preset.ltx_model)
        self._model.set(preset.restoration_model if preset.restoration_model in _MODELS else "basicvsrpp")
        self._show_notice(None)
        self._show_model(self._usable_model())
        self._refresh_download()

    def collect(self) -> dict:
        return {
            "restoration_model": self._widgets["restoration_model"].get(),
            "ltx_model": self._widgets["ltx_model"].get_value(),
            "ltx_seed": parse_seed(self._widgets["ltx_seed"].get()),
            "ltx_fast": self._widgets["ltx_fast"].get() == 1,
            "ltx_large_canvas": self._widgets["ltx_large_canvas"].get() == 1,
            "ltx_trial": self._widgets["ltx_trial"].get() == 1,
        }
