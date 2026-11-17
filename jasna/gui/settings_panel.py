"""Settings panel - right side configuration options."""

import customtkinter as ctk

from jasna.accelerator import AcceleratorVendor
from jasna.gui.components import AutoHidingScrollableFrame, Toast
from jasna.gui.locales import t
from jasna.gui.ltx_models import LtxModels
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.preset_bar import PresetBar
from jasna.gui.settings_sections.detection import DetectionSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.frame_upscale import FrameUpscaleSection
from jasna.gui.settings_sections.image_restoration import ImageRestorationSection
from jasna.gui.settings_sections.post_export import PostExportSection
from jasna.gui.settings_sections.processing import ProcessingSection
from jasna.gui.settings_sections.restoration_model import RestorationModelSection
from jasna.gui.settings_sections.secondary import SecondarySection
from jasna.gui.settings_sections.widgets import LEVELS, add_info_icon
from jasna.gui.theme import Colors, Fonts, Sizing

_TABS = ("model", "detection", "upscale", "output", "images")


class SettingsPanel(ctk.CTkFrame):
    """Right panel composing the preset bar and the settings sections; widgets live in the sections."""

    def __init__(self, master, preset_manager: PresetManager, *, ltx_models: LtxModels, **kwargs):
        super().__init__(master, fg_color=Colors.BG_PANEL, corner_radius=0, **kwargs)

        self._preset_manager = preset_manager
        self._on_interactive_image_restore: callable | None = None
        self._view = preset_manager.get_settings_view()
        self._tab = _TABS[0]
        self._offered_tabs = list(_TABS)
        self._uses_standard, self._uses_ltx = True, False

        self._preset_bar = PresetBar(
            self, preset_manager, get_settings=self.get_settings, apply=self._apply, show_toast=self._show_toast
        )
        self._preset_bar.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        self._build_view_bar()
        self._build_scrollable()
        self._build_sections(ltx_models)
        self._preset_bar.load(preset_manager.get_last_selected())

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
        view_by_label = {t(f"view_{view}"): view for view in LEVELS}
        self._view_switch = ctk.CTkSegmentedButton(
            bar,
            values=list(view_by_label),
            command=lambda label: self.set_view(view_by_label[label]),
            **segmented_colors,
        )
        self._view_switch.pack(side="left")
        self._view_switch.set(t(f"view_{self._view}"))
        add_info_icon(bar, t("tip_settings_view")).pack(side="left", padx=6)

        tab_by_label = {t(f"tab_{tab}"): tab for tab in _TABS}
        self._tab_bar = ctk.CTkSegmentedButton(
            self,
            values=list(tab_by_label),
            command=lambda label: self.show_tab(tab_by_label[label]),
            **segmented_colors,
        )
        self._tab_bar.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=(0, Sizing.PADDING_MEDIUM))

    def set_view(self, view: str) -> None:
        """Each level shows more settings in the same tabs; Simple leaves out the tabs it has nothing in."""
        self._view = view
        self._view_switch.set(t(f"view_{view}"))
        self._preset_manager.set_settings_view(view)
        self._layout_sections()

    def show_tab(self, tab: str) -> None:
        self._tab = tab
        self._layout_sections()

    def _offer_tabs(self, tabs: list[str]) -> None:
        if tabs == self._offered_tabs:
            return
        self._offered_tabs = tabs
        self._tab_bar.configure(values=[t(f"tab_{tab}") for tab in tabs])
        # CTkSegmentedButton keeps the weight of columns whose buttons are gone.
        for column in range(len(tabs), len(_TABS)):
            self._tab_bar.grid_columnconfigure(column, weight=0, minsize=0)

    def _layout_sections(self) -> None:
        visible = [
            section for section in self._sections
            if section.layout(self._view, standard=self._uses_standard, ltx=self._uses_ltx)
        ]
        tabs = [tab for tab in _TABS if any(section in visible for section in self._tab_sections[tab])]
        if self._tab not in tabs:
            self._tab = tabs[0]
        self._offer_tabs(tabs)
        self._tab_bar.set(t(f"tab_{self._tab}"))
        for section in self._sections:
            section.frame.pack_forget()
        for section in self._tab_sections[self._tab]:
            if section in visible:
                section.frame.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._scroll.scroll_to_top()

    def _show_toast(self, message: str, type_: str):
        Toast.show(self.winfo_toplevel(), message, type_, rely=0.9)

    def _build_scrollable(self):
        self._scroll = AutoHidingScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=Colors.BORDER_LIGHT,
            scrollbar_button_hover_color=Colors.BORDER_LIGHT,
        )
        self._scroll.pack(fill="both", expand=True, padx=Sizing.PADDING_MEDIUM, pady=(0, Sizing.PADDING_MEDIUM))

    def _build_sections(self, ltx_models: LtxModels):
        on_modified = self._preset_bar.refresh_modified
        self._model_section = RestorationModelSection(
            self._scroll, on_modified, self._on_restoration_model_changed, ltx_models=ltx_models
        )
        self._detection_section = DetectionSection(self._scroll, on_modified)
        self._processing_section = ProcessingSection(
            self._scroll, on_modified, self._detection_section.limit_to_clip_size
        )
        self._secondary_section = SecondarySection(self._scroll, on_modified)
        self._frame_upscale_section = FrameUpscaleSection(self._scroll, on_modified)
        image_section = ImageRestorationSection(
            self._scroll, on_modified, self._show_toast, self._open_interactive_image_restore
        )
        self._encoding_section = EncodingSection(self._scroll, on_modified)
        post_export_section = PostExportSection(self._scroll, on_modified)
        self._tab_sections = {
            "model": [self._model_section, self._processing_section],
            "detection": [self._detection_section],
            "upscale": [self._frame_upscale_section, self._secondary_section],
            "output": [self._encoding_section, post_export_section],
            "images": [image_section],
        }
        self._sections = [section for sections in self._tab_sections.values() for section in sections]

    def _apply(self, preset: AppSettings) -> None:
        for section in self._sections:
            section.apply(preset)
        self._detection_section.limit_to_clip_size(self._processing_section.max_clip_size())

    def _on_restoration_model_changed(self, model: str, auto_split: bool):
        self._uses_standard, self._uses_ltx = model != "ltx" or auto_split, model == "ltx"
        self._secondary_section.set_ltx_only(not self._uses_standard)
        self._layout_sections()

    def ltx_unavailable_reason(self) -> str | None:
        """Locale key saying why the chosen LTX model cannot run now, or None when it can."""
        return self._model_section.unavailable_reason()

    def ltx_runs_on_this_pc(self) -> bool:
        return self._model_section.ltx_runs_on_this_pc()

    def set_ltx_trial(self, trial: bool):
        self._model_section.set_trial(trial)
        self._preset_bar.refresh_modified()

    def refresh_ltx_models(self):
        self._model_section.refresh()
        self._preset_bar.refresh_modified()

    def set_gpu_support(self, *, vendor: AcceleratorVendor, blackwell: bool, frame_generation: bool):
        self._model_section.set_gpu_support(vendor=vendor, blackwell=blackwell)
        self._processing_section.set_gpu_support(vendor)
        self._secondary_section.set_gpu_support(vendor)
        self._frame_upscale_section.set_gpu_support(vendor)
        self._encoding_section.set_frame_generation_support(frame_generation)
        self._preset_bar.refresh_modified()

    def set_on_interactive_image_restore(self, callback: callable):
        self._on_interactive_image_restore = callback

    def _open_interactive_image_restore(self):
        if self._on_interactive_image_restore:
            self._on_interactive_image_restore()

    def get_settings(self) -> AppSettings:
        values: dict = {}
        for section in self._sections:
            values.update(section.collect())
        return AppSettings(**values)

    def set_enabled(self, enabled: bool):
        """Enable or disable the settings; the queue-wide post-export action stays editable while processing."""
        self._preset_bar.set_enabled(enabled)
        for section in self._sections:
            section.set_enabled(enabled)
