"""Preset choice above the settings: pick, create, save, delete and reset presets, and mark
the preset modified once the settings differ from it."""

from collections.abc import Callable
from dataclasses import replace

import customtkinter as ctk

from jasna.gui.components import ConfirmDialog, PresetDialog, Tooltip
from jasna.gui.icons import NativeIconButton
from jasna.gui.locales import t
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.theme import Colors, Fonts, Sizing

_FACTORY_MARK = "🔒 "


class PresetBar(ctk.CTkFrame):
    def __init__(
        self,
        master,
        preset_manager: PresetManager,
        *,
        get_settings: Callable[[], AppSettings],
        apply: Callable[[AppSettings], None],
        show_toast: Callable[[str, str], None],
    ):
        super().__init__(master, fg_color="transparent", height=48)
        self.pack_propagate(False)
        self._presets = preset_manager
        self._get_settings = get_settings
        self._apply = apply
        self._show_toast = show_toast
        self._name = preset_manager.get_last_selected()
        self._saved: AppSettings | None = None
        self._loading = False

        ctk.CTkLabel(
            self, text=t("preset"), font=(Fonts.FAMILY, Fonts.SIZE_NORMAL), text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, 8))
        self._dropdown = ctk.CTkOptionMenu(
            self,
            values=self._dropdown_values(),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=180,
            height=Sizing.BUTTON_HEIGHT,
            command=self._on_picked,
        )
        self._dropdown.pack(side="left")

        self._reset_btn, self._delete_btn, self._save_btn, self._create_btn = (
            self._icon_button(icon, command, tooltip_key)
            for icon, command, tooltip_key in (
                ("reset", self._on_reset, "tip_preset_reset"),
                ("delete", self._on_delete, "tip_preset_delete"),
                ("save", self._on_save, "tip_preset_save"),
                ("create", self._on_create, "tip_preset_create"),
            )
        )
        self._reset_btn.pack(side="right")
        for button in (self._delete_btn, self._save_btn, self._create_btn):
            button.pack(side="right", padx=(0, 4))

    def _icon_button(self, icon: str, command, tooltip_key: str) -> NativeIconButton:
        button = NativeIconButton(
            self, icon, 18, Colors.TEXT_PRIMARY, Colors.BG_PANEL, Colors.BG_CARD, Colors.BORDER_LIGHT, command, 32, 32
        )
        Tooltip(button, t(tooltip_key))
        return button

    def _display_name(self, name: str) -> str:
        return f"{_FACTORY_MARK}{name}" if self._presets.is_factory_preset(name) else name

    def _dropdown_values(self) -> list[str]:
        factory, user = self._presets.get_all_preset_names()
        return [self._display_name(name) for name in factory + user]

    def _on_picked(self, display_name: str) -> None:
        self.load(display_name.removeprefix(_FACTORY_MARK))

    def load(self, name: str) -> None:
        """Show preset ``name`` (Default when it is gone) in the settings."""
        self._loading = True
        self._name, preset = self._presets.resolve(name)
        self._presets.set_last_selected(self._name)
        self._apply(preset)
        # A preset without a CQ uses this GPU's default, so compare against the value shown.
        self._saved = preset if preset.encoder_cq is not None else replace(
            preset, encoder_cq=self._get_settings().encoder_cq
        )
        self._show_name()
        self._loading = False

    def _show_name(self) -> None:
        factory = self._presets.is_factory_preset(self._name)
        self._save_btn.configure(state="disabled" if factory else "normal")
        if factory:
            self._delete_btn.pack_forget()
        else:
            self._delete_btn.pack(side="right", padx=(0, 4), after=self._reset_btn)
        self._dropdown.set(self._display_name(self._name))

    def refresh_modified(self) -> None:
        """Mark the preset modified when the settings shown differ from it."""
        if self._loading:
            return
        name = self._display_name(self._name)
        modified = self._get_settings() != self._saved
        self._dropdown.set(t("preset_modified", name=name) if modified else name)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self._dropdown, self._create_btn, self._save_btn, self._reset_btn, self._delete_btn):
            widget.configure(state=state)

    def _on_reset(self) -> None:
        self.load(self._name)
        self._show_toast(t("toast_settings_reset"), "info")

    def _on_save(self) -> None:
        settings = self._get_settings()
        if self._presets.update_preset(self._name, settings):
            self._saved = settings
            self._dropdown.set(self._name)
            self._show_toast(t("toast_preset_saved", name=self._name), "success")

    def _on_create(self) -> None:
        factory, user = self._presets.get_all_preset_names()

        def create(name: str) -> None:
            settings = self._get_settings()
            if self._presets.create_preset(name, settings):
                self._dropdown.configure(values=self._dropdown_values())
                self._name = name
                self._saved = settings
                self._presets.set_last_selected(name)
                self._show_name()
                self._show_toast(t("toast_preset_created", name=name), "success")

        PresetDialog(self.winfo_toplevel(), create, factory + user)

    def _on_delete(self) -> None:
        def delete() -> None:
            name = self._name
            if self._presets.delete_preset(name):
                self._dropdown.configure(values=self._dropdown_values())
                self.load("Default")
                self._show_toast(t("toast_preset_deleted", name=name), "success")

        ConfirmDialog(self.winfo_toplevel(), t("dialog_delete_preset"), t("confirm_delete", name=self._name), delete)
