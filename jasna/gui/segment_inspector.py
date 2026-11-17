"""Restoration inspector of the segment editor: the model and LTX seed of the selected range,
with the seed panel that renders and compares seeds."""

from __future__ import annotations

import random
from collections.abc import Callable

import customtkinter as ctk

from jasna.gui.components import Tooltip
from jasna.gui.icons import NativeIconButton
from jasna.gui.locales import t
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.gui.segment_seed_panel import SeedPanel
from jasna.gui.settings_sections.restoration_model import parse_seed
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.segments import SegmentRestoration


class RestorationInspector(ctk.CTkFrame):
    """Grids itself into ``master`` while a range is selected. ``seed_panel(inspector, on_use_seed)``
    builds the seed panel into it; ``on_edited`` follows every change of the ranges and
    ``on_selected_changed`` a change of the selected range's restoration."""

    def __init__(
        self,
        master,
        *,
        state: SegmentEditorState,
        job_ltx_seed: int,
        ltx_key: str,
        ltx_label: str,
        ltx_blocked_reason: str | None,
        scanning: Callable[[], bool],
        seed_panel: Callable[[ctk.CTkFrame, Callable[[int], None]], SeedPanel],
        on_edited: Callable[[], None],
        on_selected_changed: Callable[[], None],
    ) -> None:
        super().__init__(master, fg_color=Colors.BG_PANEL, corner_radius=Sizing.BORDER_RADIUS)
        self._state = state
        self._job_ltx_seed = job_ltx_seed
        self._ltx_key = ltx_key
        self._ltx_blocked_reason = ltx_blocked_reason
        self._scanning = scanning
        self._on_edited = on_edited
        self._on_selected_changed = on_selected_changed
        self.grid_columnconfigure(1, weight=1)
        self._restore_with_label = ctk.CTkLabel(
            self,
            text=t("segments_restore_with"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
        )
        self._restore_with_label.grid(row=0, column=0, sticky="w", padx=(8, 6), pady=4)
        Tooltip(self._restore_with_label, t("tip_segments_restore_with"))
        models = ctk.CTkFrame(self, fg_color="transparent")
        models.grid(row=0, column=1, sticky="w", pady=4)
        self._range_model = ctk.StringVar(value="basicvsrpp")
        self._model_radios = {}
        for model in ("basicvsrpp", "ltx"):
            radio = ctk.CTkRadioButton(
                models,
                text=ltx_label if model == "ltx" else t("model_basicvsrpp"),
                variable=self._range_model,
                value=model,
                command=self._on_range_model_changed,
                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                fg_color=Colors.MODEL_LTX if model == "ltx" else Colors.PRIMARY,
                text_color=Colors.TEXT_PRIMARY,
                radiobutton_width=16,
                radiobutton_height=16,
            )
            radio.pack(side="left", padx=(0, 10))
            self._model_radios[model] = radio
        Tooltip(self._model_radios["basicvsrpp"], t("tip_model_basicvsrpp"))
        self._ltx_radio_tooltip = Tooltip(self._model_radios["ltx"], "")
        self._seed_label = ctk.CTkLabel(
            self,
            text=t("ltx_seed"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
        )
        self._seed_entry = ctk.CTkEntry(self, font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL))
        self._seed_entry.bind("<Return>", self._commit_seed_entry)
        self._seed_entry.bind("<FocusOut>", self._commit_seed_entry)
        for widget in (self._seed_label, self._seed_entry):
            Tooltip(widget, t("tip_ltx_seed"))
        self._new_seed_btn = NativeIconButton(
            self,
            "reset",
            14,
            Colors.TEXT_PRIMARY,
            Colors.BG_PANEL,
            Colors.BORDER_LIGHT,
            Colors.BORDER_LIGHT,
            self._new_range_seed,
            28,
            28,
        )
        Tooltip(self._new_seed_btn, t("tip_ltx_new_seed"))
        self._use_for_all_btn = ctk.CTkButton(
            self,
            text=t("segments_use_for_all"),
            width=10,
            height=24,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            command=self._use_restoration_for_all,
        )
        self._use_for_all_btn.grid(row=0, column=2, columnspan=2, sticky="e", padx=(4, 8), pady=4)
        Tooltip(self._use_for_all_btn, t("tip_segments_use_for_all"))
        self.seeds = seed_panel(self, lambda seed: self.apply(SegmentRestoration("ltx", seed)))

    def lockable_widgets(self) -> tuple:
        return (
            *self._model_radios.values(),
            self._seed_entry,
            self._new_seed_btn,
            self._use_for_all_btn,
            *self.seeds.lockable_widgets(),
        )

    def _selected_restoration(self) -> SegmentRestoration:
        model = self._range_model.get()
        if model != "ltx":
            return SegmentRestoration(model, None)
        selected = self._state.selected_segment
        current = None if selected is None else selected.restoration.ltx_seed
        seed = parse_seed(self._seed_entry.get()) if self._seed_entry.get().strip() else current
        return SegmentRestoration("ltx", self._job_ltx_seed if seed is None else seed)

    def apply(self, restoration: SegmentRestoration) -> None:
        """Give the selected range ``restoration``."""
        if self._state.set_selected_restoration(restoration):
            self.seeds.forget_shown_seed()
            self._on_edited()
            self._on_selected_changed()
        else:
            self.refresh()

    def _on_range_model_changed(self) -> None:
        self._seed_entry.delete(0, "end")
        self.apply(self._selected_restoration())

    def _commit_seed_entry(self, _event=None) -> None:
        if self._range_model.get() == "ltx" and self._state.selected_segment is not None:
            self.apply(self._selected_restoration())

    def _new_range_seed(self) -> None:
        self.apply(SegmentRestoration("ltx", random.randrange(1, 2**31)))

    def _use_restoration_for_all(self) -> None:
        selected = self._state.selected_segment
        if selected is not None and self._state.set_all_restoration(selected.restoration):
            self._on_edited()

    def refresh(self) -> None:
        selected = self._state.selected_segment
        if selected is None:
            self.grid_forget()
            return
        self.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        restoration = selected.restoration
        self._range_model.set(restoration.model)
        blocked = self._ltx_blocked_reason
        locked = self._scanning()
        self._model_radios["basicvsrpp"].configure(state="disabled" if locked else "normal")
        self._model_radios["ltx"].configure(
            state="disabled" if locked or (blocked is not None and restoration.model != "ltx") else "normal"
        )
        self._ltx_radio_tooltip.set_text(t(blocked) if blocked else t(f"tip_model_{self._ltx_key}"))
        if restoration.model == "ltx":
            self._seed_label.grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
            self._seed_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))
            self._new_seed_btn.grid(row=1, column=2, sticky="w", padx=4, pady=(0, 6))
            self._seed_entry.delete(0, "end")
            self._seed_entry.insert(0, str(restoration.ltx_seed))
        else:
            for widget in (self._seed_label, self._seed_entry, self._new_seed_btn):
                widget.grid_forget()
            self.seeds.hide()
        state = "disabled" if locked else "normal"
        for widget in (self._seed_entry, self._new_seed_btn, self._use_for_all_btn):
            widget.configure(state=state)
        self.seeds.refresh()
