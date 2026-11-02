"""Settings panel - right side configuration options."""

import logging

import customtkinter as ctk
import tkinter as tk
from dataclasses import asdict

from jasna.accelerator import AcceleratorVendor
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.models import AppSettings, PresetManager
from jasna.mosaic.detection_registry import default_detection_batch_size
from jasna.gui.components import (
    AutoHidingScrollableFrame,
    ConfirmDialog,
    PresetDialog,
    Toast,
    Tooltip,
)
from jasna.gui.icons import NativeIconButton
from jasna.gui.locales import t
from jasna.gui.ltx_models import LtxModels
from jasna.gui.settings_sections.detection import TEMPORAL_FILTER_SLIDER_MAX, DetectionSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.image_restoration import ImageRestorationSection
from jasna.gui.settings_sections.post_export import PostExportSection
from jasna.gui.settings_sections.processing import ProcessingSection
from jasna.gui.settings_sections.restoration_model import RestorationModelSection
from jasna.gui.settings_sections.secondary import SecondarySection

logger = logging.getLogger(__name__)

_EDITABLE_WHILE_PROCESSING = frozenset({"post_export_action", "post_export_command"})
_VIEWS = ("simple", "advanced", "expert")
_TABS = ("model", "detection", "upscale", "output", "images")


class SettingsPanel(ctk.CTkFrame):
    """Right panel composing the settings sections; widgets live in the sections."""

    def __init__(self, master, preset_manager: PresetManager, *, ltx_models: LtxModels, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
            **kwargs
        )

        self._preset_manager = preset_manager
        self._current_preset = self._preset_manager.get_last_selected()
        self._saved_preset_settings: AppSettings | None = None  # Snapshot of preset when loaded
        self._is_modified = False
        self._applying_preset = False  # Flag to prevent modification tracking during apply
        self._widgets: dict = {}
        self._on_interactive_image_restore: callable | None = None
        self._view = self._preset_manager.get_settings_view()
        self._tab = _TABS[0]
        self._uses_standard, self._uses_ltx = True, False

        self._build_preset_bar()
        self._build_view_bar()
        self._build_scrollable()
        self._build_sections(ltx_models)
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

    def _build_view_bar(self):
        segmented_colors = dict(
            selected_color=Colors.PRIMARY,
            selected_hover_color=Colors.PRIMARY_HOVER,
            unselected_color=Colors.BG_CARD,
            unselected_hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=(0, Sizing.PADDING_SMALL))
        view_by_label = {t(f"view_{view}"): view for view in _VIEWS}
        self._view_switch = ctk.CTkSegmentedButton(
            bar,
            values=list(view_by_label),
            command=lambda label: self.set_view(view_by_label[label]),
            **segmented_colors,
        )
        self._view_switch.pack(side="left")
        self._view_switch.set(t(f"view_{self._view}"))
        view_tip = ctk.CTkLabel(
            bar, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2"
        )
        view_tip.pack(side="left", padx=6)
        Tooltip(view_tip, t("tip_settings_view"))

        tab_by_label = {t(f"tab_{tab}"): tab for tab in _TABS}
        self._tab_bar = ctk.CTkSegmentedButton(
            self,
            values=list(tab_by_label),
            command=lambda label: self.show_tab(tab_by_label[label]),
            **segmented_colors,
        )
        self._tab_bar.set(t(f"tab_{self._tab}"))

    def set_view(self, view: str) -> None:
        """Simple shows the main choices on one page; Advanced and Expert show more of them in tabs."""
        self._view = view
        self._view_switch.set(t(f"view_{view}"))
        self._preset_manager.set_settings_view(view)
        self._layout_sections()

    def show_tab(self, tab: str) -> None:
        self._tab = tab
        self._tab_bar.set(t(f"tab_{tab}"))
        self._layout_sections()

    def _layout_sections(self) -> None:
        simple = self._view == "simple"
        for section in self._sections:
            section.frame.pack_forget()
        self._tab_bar.pack_forget()
        self._scroll.pack_forget()
        if simple:
            shown = self._simple_sections
        else:
            self._tab_bar.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=(0, Sizing.PADDING_MEDIUM))
            shown = self._tab_sections[self._tab]
        self._scroll.pack(fill="both", expand=True, padx=Sizing.PADDING_MEDIUM, pady=(0, Sizing.PADDING_MEDIUM))
        visible = [
            section for section in self._sections
            if section.layout(self._view, standard=self._uses_standard, ltx=self._uses_ltx)
        ]
        for section in shown:
            if section in visible:
                section.frame.set_header_shown(simple)
                section.frame.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._scroll.scroll_to_top()

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
            display_name = t("preset_modified", name=display_name)
        self._preset_dropdown.set(display_name)

    def _show_toast(self, message: str, type_: str = "info"):
        """Show a toast notification."""
        toast = Toast(self.winfo_toplevel(), message, type_)
        toast.place(relx=0.5, rely=0.9, anchor="center")

    def _build_scrollable(self):
        self._scroll = AutoHidingScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=Colors.BORDER_LIGHT,
            scrollbar_button_hover_color=Colors.BORDER_LIGHT,
        )

    def _build_sections(self, ltx_models: LtxModels):
        self._model_section = RestorationModelSection(
            self._scroll,
            self._widgets,
            self._mark_modified,
            self._on_restoration_model_changed,
            ltx_models=ltx_models,
        )
        processing_section = ProcessingSection(
            self._scroll, self._widgets, self._mark_modified, self._sync_temporal_filter_limits
        )
        detection_section = DetectionSection(self._scroll, self._widgets, self._mark_modified)
        self._secondary_section = SecondarySection(self._scroll, self._widgets)
        image_section = ImageRestorationSection(
            self._scroll,
            self._widgets,
            self._mark_modified,
            self._show_toast,
            self._open_interactive_image_restore,
        )
        encoding_section = EncodingSection(self._scroll, self._widgets, self._mark_modified)
        post_export_section = PostExportSection(self._scroll, self._widgets, self._mark_modified)
        self._tab_sections = {
            "model": [self._model_section, processing_section],
            "detection": [detection_section],
            "upscale": [self._secondary_section],
            "output": [encoding_section, post_export_section],
            "images": [image_section],
        }
        self._sections = [section for sections in self._tab_sections.values() for section in sections]
        self._simple_sections = [self._model_section, encoding_section, post_export_section, image_section]

    def _on_restoration_model_changed(self, model: str, auto_split: bool):
        self._uses_standard, self._uses_ltx = model != "ltx" or auto_split, model == "ltx"
        self._secondary_section.set_ltx_only(not self._uses_standard)
        self._layout_sections()

    def ltx_unavailable_reason(self) -> str | None:
        """Locale key saying why the chosen LTX model cannot run now, or None when it can."""
        return self._model_section.unavailable_reason()

    def set_ltx_trial(self, trial: bool):
        self._model_section.set_trial(trial)
        self._update_modified_indicator()

    def refresh_ltx_models(self):
        self._model_section.refresh()
        self._update_modified_indicator()

    def set_gpu_support(self, *, vendor: AcceleratorVendor, blackwell: bool):
        self._model_section.set_gpu_support(vendor=vendor, blackwell=blackwell)
        self._secondary_section.set_gpu_support(vendor)
        self._update_modified_indicator()

    def set_on_interactive_image_restore(self, callback: callable):
        self._on_interactive_image_restore = callback

    def _open_interactive_image_restore(self):
        if self._on_interactive_image_restore:
            self._on_interactive_image_restore()

    def _on_preset_changed(self, preset_display_name: str):
        if preset_display_name not in self._display_to_name:
            return
        preset_name = self._display_to_name[preset_display_name]
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
        if self._saved_preset_settings.encoder_cq is None:
            # ``None`` is the portable preset sentinel; modification tracking
            # compares against the literal value currently displayed by this GPU.
            self._saved_preset_settings.encoder_cq = int(
                self._widgets["encoder_cq"].get()
            )
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
            batch_size=default_detection_batch_size("cuda:0"),
            **values,
        )

    def set_enabled(self, enabled: bool):
        """Enable or disable the settings; the queue-wide post-export action stays editable while processing."""
        state = "normal" if enabled else "disabled"

        # Preset bar buttons
        self._preset_dropdown.configure(state=state)
        self._create_btn.configure(state=state)
        self._save_btn.configure(state=state)
        self._reset_btn.configure(state=state)
        self._delete_btn.configure(state=state)

        # All interactive widgets
        for key, widget in self._widgets.items():
            if key.endswith("_val") or key in _EDITABLE_WHILE_PROCESSING or isinstance(widget, tk.Variable):
                continue
            try:
                widget.configure(state=state)
            except Exception:
                logger.debug("Widget %r does not support state=%s", key, state, exc_info=True)
        self._model_section.set_enabled(enabled)
        self._secondary_section.set_enabled(enabled)
