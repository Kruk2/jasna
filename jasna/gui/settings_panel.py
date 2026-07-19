"""Settings panel - right side configuration options."""

import logging

import customtkinter as ctk
from dataclasses import asdict

from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.components import ConfirmDialog, PresetDialog, Toast, Tooltip
from jasna.gui.icons import NativeIconButton
from jasna.gui.locales import t
from jasna.gui.settings_sections.advanced import (
    TEMPORAL_FILTER_SLIDER_MAX,
    AdvancedSection,
)
from jasna.gui.settings_sections.basic import BasicSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.image_restoration import ImageRestorationSection
from jasna.gui.settings_sections.post_export import PostExportSection
from jasna.gui.settings_sections.secondary import SecondarySection

logger = logging.getLogger(__name__)


class SettingsPanel(ctk.CTkFrame):
    """Right panel composing the settings sections; widgets live in the sections."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
            **kwargs
        )

        self._preset_manager = PresetManager()
        self._current_preset = self._preset_manager.get_last_selected()
        self._saved_preset_settings: AppSettings | None = None  # Snapshot of preset when loaded
        self._is_modified = False
        self._applying_preset = False  # Flag to prevent modification tracking during apply
        self._widgets: dict = {}
        self._on_interactive_image_restore: callable | None = None

        self._build_preset_bar()
        self._build_scrollable()
        self._build_sections()
        self._apply_preset(self._current_preset)

    def _build_preset_bar(self):
        bar = ctk.CTkFrame(self, fg_color="transparent", height=48)
        bar.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        bar.pack_propagate(False)

        preset_label = ctk.CTkLabel(
            bar,
            text=t("preset"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        )
        preset_label.pack(side="left", padx=(0, 8))

        # Build dropdown values with sections
        self._update_dropdown_values()

        self._preset_dropdown = ctk.CTkOptionMenu(
            bar,
            values=self._dropdown_values,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=180,
            height=Sizing.BUTTON_HEIGHT,
            command=self._on_preset_changed,
        )
        self._preset_dropdown.pack(side="left")

        # Action buttons (right-aligned): Reset, Delete, Save, Create
        self._reset_btn = NativeIconButton(
            bar,
            "reset",
            18,
            Colors.TEXT_PRIMARY,
            Colors.BG_PANEL,
            Colors.BG_CARD,
            Colors.BORDER_LIGHT,
            self._on_reset,
            32,
            32,
        )
        self._reset_btn.pack(side="right")
        Tooltip(self._reset_btn, t("tip_preset_reset"))

        self._delete_btn = NativeIconButton(
            bar,
            "delete",
            18,
            Colors.TEXT_PRIMARY,
            Colors.BG_PANEL,
            Colors.BG_CARD,
            Colors.BORDER_LIGHT,
            self._on_delete_preset,
            32,
            32,
        )
        self._delete_btn.pack(side="right", padx=(0, 4))
        Tooltip(self._delete_btn, t("tip_preset_delete"))

        self._save_btn = NativeIconButton(
            bar,
            "save",
            18,
            Colors.TEXT_PRIMARY,
            Colors.BG_PANEL,
            Colors.BG_CARD,
            Colors.BORDER_LIGHT,
            self._on_save_preset,
            32,
            32,
        )
        self._save_btn.pack(side="right", padx=(0, 4))
        Tooltip(self._save_btn, t("tip_preset_save"))

        self._create_btn = NativeIconButton(
            bar,
            "create",
            18,
            Colors.TEXT_PRIMARY,
            Colors.BG_PANEL,
            Colors.BG_CARD,
            Colors.BORDER_LIGHT,
            self._on_create_preset,
            32,
            32,
        )
        self._create_btn.pack(side="right", padx=(0, 4))
        Tooltip(self._create_btn, t("tip_preset_create"))

    def _update_dropdown_values(self):
        """Build dropdown values list."""
        factory, user = self._preset_manager.get_all_preset_names()
        # Add lock icon to factory presets
        factory_display = [f"🔒 {name}" for name in factory]
        self._dropdown_values = factory_display + user
        # Map display names back to actual names
        self._display_to_name = {f"🔒 {name}": name for name in factory}
        self._display_to_name.update({name: name for name in user})

    def _refresh_dropdown(self):
        """Refresh dropdown with current presets."""
        self._update_dropdown_values()
        self._preset_dropdown.configure(values=self._dropdown_values)

    def get_last_output_folder(self) -> str:
        return self._preset_manager.get_last_output_folder()

    def set_last_output_folder(self, path: str):
        self._preset_manager.set_last_output_folder(path)

    def get_last_output_pattern(self) -> str:
        return self._preset_manager.get_last_output_pattern()

    def set_last_output_pattern(self, pattern: str):
        self._preset_manager.set_last_output_pattern(pattern)

    def _update_button_states(self):
        """Update button states based on current preset."""
        is_factory = self._preset_manager.is_factory_preset(self._current_preset)

        # Save button: disabled for factory presets
        if is_factory:
            self._save_btn.configure(state="disabled")
        else:
            self._save_btn.configure(state="normal")

        # Delete button: hidden for factory presets
        if is_factory:
            self._delete_btn.pack_forget()
        else:
            self._delete_btn.pack(side="right", padx=(0, 4), after=self._reset_btn)

    def _display_name(self, preset_name: str) -> str:
        if self._preset_manager.is_factory_preset(preset_name):
            return f"🔒 {preset_name}"
        return preset_name

    def _update_modified_indicator(self):
        """Update dropdown text to show modified status."""
        current_settings = self.get_settings()
        if self._saved_preset_settings:
            self._is_modified = asdict(current_settings) != asdict(self._saved_preset_settings)
        else:
            self._is_modified = False

        display_name = self._display_name(self._current_preset)
        if self._is_modified:
            display_name += " (Modified)*"
        self._preset_dropdown.set(display_name)

    def _show_toast(self, message: str, type_: str = "info"):
        """Show a toast notification."""
        toast = Toast(self.winfo_toplevel(), message, type_)
        toast.place(relx=0.5, rely=0.9, anchor="center")

    def _build_scrollable(self):
        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=Colors.BG_PANEL,
            scrollbar_button_hover_color=Colors.BORDER_LIGHT,
        )
        self._scroll.pack(fill="both", expand=True, padx=Sizing.PADDING_MEDIUM, pady=(0, Sizing.PADDING_MEDIUM))

    def _build_sections(self):
        self._sections = [
            BasicSection(
                self._scroll,
                self._widgets,
                self._mark_modified,
                self._sync_temporal_filter_limits,
            ),
            AdvancedSection(self._scroll, self._widgets, self._mark_modified),
            SecondarySection(self._scroll, self._widgets),
            ImageRestorationSection(
                self._scroll,
                self._widgets,
                self._mark_modified,
                self._show_toast,
                self._open_interactive_image_restore,
            ),
            EncodingSection(self._scroll, self._widgets, self._mark_modified),
            PostExportSection(self._scroll, self._widgets, self._mark_modified),
        ]

    def set_on_interactive_image_restore(self, callback: callable):
        self._on_interactive_image_restore = callback

    def _open_interactive_image_restore(self):
        if self._on_interactive_image_restore:
            self._on_interactive_image_restore()

    def _on_preset_changed(self, preset_display_name: str):
        # Strip modified indicator if present
        if preset_display_name.endswith(" (Modified)*"):
            return  # User re-selected current modified preset
        # Convert display name to actual name
        preset_name = self._display_to_name.get(preset_display_name, preset_display_name)
        self._apply_preset(preset_name)

    def _apply_preset(self, preset_name: str):
        self._applying_preset = True  # Prevent modification tracking
        preset_name, preset = self._preset_manager.resolve(preset_name)

        self._current_preset = preset_name
        self._saved_preset_settings = AppSettings(**asdict(preset))  # Deep copy
        self._is_modified = False

        # Save as last selected
        self._preset_manager.set_last_selected(preset_name)

        # Update UI state
        self._update_button_states()
        self._preset_dropdown.set(self._display_name(preset_name))

        for section in self._sections:
            section.apply(preset)
        self._sync_temporal_filter_limits(int(self._widgets["max_clip_size"].get()))

        self._applying_preset = False  # Re-enable modification tracking

    def _on_reset(self):
        """Reset to saved preset values."""
        self._apply_preset(self._current_preset)
        self._show_toast(t("toast_settings_reset"), "info")

    def _on_save_preset(self):
        """Save current settings to user preset."""
        if self._preset_manager.is_factory_preset(self._current_preset):
            return

        settings = self.get_settings()
        if self._preset_manager.update_preset(self._current_preset, settings):
            self._saved_preset_settings = AppSettings(**asdict(settings))
            self._is_modified = False
            self._preset_dropdown.set(self._current_preset)
            self._show_toast(t("toast_preset_saved", name=self._current_preset), "success")

    def _on_create_preset(self):
        """Open dialog to create new preset."""
        factory, user = self._preset_manager.get_all_preset_names()
        existing = factory + user

        def on_create(name: str):
            settings = self.get_settings()
            if self._preset_manager.create_preset(name, settings):
                self._refresh_dropdown()
                self._current_preset = name
                self._saved_preset_settings = AppSettings(**asdict(settings))
                self._is_modified = False
                self._preset_manager.set_last_selected(name)
                self._update_button_states()
                self._preset_dropdown.set(name)
                self._show_toast(t("toast_preset_created", name=name), "success")

        PresetDialog(self.winfo_toplevel(), on_create, existing)

    def _on_delete_preset(self):
        """Delete current user preset."""
        if self._preset_manager.is_factory_preset(self._current_preset):
            return

        def on_confirm():
            name = self._current_preset
            if self._preset_manager.delete_preset(name):
                self._refresh_dropdown()
                self._apply_preset("Default")
                self._show_toast(t("toast_preset_deleted", name=name), "success")

        ConfirmDialog(
            self.winfo_toplevel(),
            t("dialog_delete_preset"),
            t("confirm_delete", name=self._current_preset),
            on_confirm
        )

    def _mark_modified(self):
        """Mark settings as modified from preset."""
        if self._applying_preset:
            return  # Don't mark modified while applying a preset
        self._update_modified_indicator()

    def _sync_temporal_filter_limits(self, max_clip_size: int):
        limit = max(0, min(TEMPORAL_FILTER_SLIDER_MAX, int(max_clip_size) - 1))
        for key in ("max_detection_gap", "min_detection_duration"):
            slider = self._widgets[key]
            slider.configure(to=limit, number_of_steps=max(1, limit))
            value = min(int(slider.get()), limit)
            slider.set(value)
            self._widgets[f"{key}_val"].configure(text=str(value))

    def get_settings(self) -> AppSettings:
        values: dict = {}
        for section in self._sections:
            values.update(section.collect())
        return AppSettings(
            batch_size=4,  # Fixed default value
            **values,
        )

    def set_enabled(self, enabled: bool):
        """Enable or disable all settings controls."""
        state = "normal" if enabled else "disabled"

        # Preset bar buttons
        self._preset_dropdown.configure(state=state)
        self._create_btn.configure(state=state)
        self._save_btn.configure(state=state)
        self._reset_btn.configure(state=state)
        if hasattr(self, "_delete_btn"):
            self._delete_btn.configure(state=state)

        # All interactive widgets
        for key, widget in self._widgets.items():
            if key.endswith("_val"):  # Skip value labels
                continue
            try:
                widget.configure(state=state)
            except Exception:
                logger.debug("Widget %r does not support state=%s", key, state, exc_info=True)
