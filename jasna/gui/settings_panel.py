"""Settings panel - right side configuration options."""

import customtkinter as ctk
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from dataclasses import asdict, fields
from PIL import ImageTk

from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.components import CollapsibleSection, ConfirmDialog, PresetDialog, Toast, Tooltip
from jasna.gui.icons import create_icon, render_toggle
from jasna.gui.locales import t

# Display labels contain punctuation ("H.264 (AVC)"), so canonical values come
# from these maps, never from .lower() on the label.
CODEC_LABEL_TO_CANONICAL = {
    "HEVC (H.265)": "hevc",
    "H.264 (AVC)": "h264",
    "AV1": "av1",
}
CODEC_CANONICAL_TO_LABEL = {v: k for k, v in CODEC_LABEL_TO_CANONICAL.items()}

_AV1_CQ_OFFSET = 7
_CQ_MIN = 15
_CQ_MAX = 35


def translate_cq_for_codec(cq: int, old_codec: str, new_codec: str) -> int:
    """Keep roughly equal quality when the user visibly switches codec scales."""
    if old_codec == new_codec:
        return cq
    if old_codec == "av1":
        cq -= _AV1_CQ_OFFSET
    if new_codec == "av1":
        cq += _AV1_CQ_OFFSET
    return max(_CQ_MIN, min(_CQ_MAX, cq))


def get_tooltip(key: str) -> str:
    """Get localized tooltip for a setting key."""
    return t(f"tip_{key}")


class CompactSwitch(tk.Label):
    def __init__(self, master, command: callable, background: str):
        self._selected = False
        self._change_command = command
        self._off_image = ImageTk.PhotoImage(
            render_toggle(
                False,
                36,
                18,
                Colors.BORDER_LIGHT,
                Colors.TEXT_PRIMARY,
            ),
            master=master,
        )
        self._on_image = ImageTk.PhotoImage(
            render_toggle(
                True,
                36,
                18,
                Colors.PRIMARY,
                Colors.TEXT_PRIMARY,
            ),
            master=master,
        )
        super().__init__(
            master,
            image=self._off_image,
            width=40,
            height=24,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.bind("<Button-1>", self._toggle)

    def _toggle(self, _event=None) -> None:
        self._selected = not self._selected
        self._refresh_image()
        self._change_command()

    def _refresh_image(self) -> None:
        self.configure(image=self._on_image if self._selected else self._off_image)

    def select(self) -> None:
        self._selected = True
        self._refresh_image()

    def deselect(self) -> None:
        self._selected = False
        self._refresh_image()

    def get(self) -> int:
        return int(self._selected)


def create_compact_switch(master, command: callable, background: str) -> CompactSwitch:
    return CompactSwitch(master, command, background)


def create_slider_value_label(master, text: str, width: int) -> tk.Label:
    return tk.Label(
        master,
        text=text,
        foreground=Colors.TEXT_PRIMARY,
        background=Colors.BG_PANEL,
        font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        width=width,
        borderwidth=0,
        highlightthickness=0,
    )


class SettingsPanel(ctk.CTkFrame):
    """Right panel containing all processing settings."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
            **kwargs
        )
        
        self._preset_manager = PresetManager()
        self._settings = AppSettings()
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
        self._reset_btn = ctk.CTkButton(
            bar,
            text="",
            image=create_icon("reset", 18, Colors.TEXT_PRIMARY),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=32,
            command=self._on_reset,
        )
        self._reset_btn.pack(side="right")
        Tooltip(self._reset_btn, t("tip_preset_reset"))
        
        self._delete_btn = ctk.CTkButton(
            bar,
            text="",
            image=create_icon("delete", 18, Colors.TEXT_PRIMARY),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=32,
            command=self._on_delete_preset,
        )
        self._delete_btn.pack(side="right", padx=(0, 4))
        Tooltip(self._delete_btn, t("tip_preset_delete"))
        
        self._save_btn = ctk.CTkButton(
            bar,
            text="",
            image=create_icon("save", 18, Colors.TEXT_PRIMARY),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=32,
            command=self._on_save_preset,
        )
        self._save_btn.pack(side="right", padx=(0, 4))
        Tooltip(self._save_btn, t("tip_preset_save"))
        
        self._create_btn = ctk.CTkButton(
            bar,
            text="",
            image=create_icon("create", 18, Colors.TEXT_PRIMARY),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=32,
            command=self._on_create_preset,
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
            self._save_btn.configure(state="disabled", text_color=Colors.TEXT_PRIMARY)
        else:
            self._save_btn.configure(state="normal", text_color=Colors.TEXT_PRIMARY)
        
        # Delete button: hidden for factory presets
        if is_factory:
            self._delete_btn.pack_forget()
        else:
            self._delete_btn.pack(side="right", padx=(0, 4), after=self._reset_btn)
            
    def _update_modified_indicator(self):
        """Update dropdown text to show modified status."""
        current_settings = self.get_settings()
        if self._saved_preset_settings:
            self._is_modified = asdict(current_settings) != asdict(self._saved_preset_settings)
        else:
            self._is_modified = False
            
        # Build display name with lock icon if factory
        display_name = self._current_preset
        if self._preset_manager.is_factory_preset(self._current_preset):
            display_name = f"🔒 {display_name}"
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
        self._build_basic_section()
        self._build_advanced_section()
        self._build_secondary_section()
        self._build_image_restoration_section()
        self._build_encoding_section()
        self._build_post_export_section()
        
    def _build_basic_section(self):
        section = CollapsibleSection(self._scroll, t("section_basic"), expanded=True)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)
        
        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        # Max Clip Size slider (10-180, step 10)
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        clip_label = ctk.CTkLabel(row1, text=t("max_clip_size"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        clip_label.pack(side="left")
        clip_tooltip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        clip_tooltip.pack(side="left", padx=4)
        Tooltip(clip_tooltip, get_tooltip("max_clip_size"))
        
        self._widgets["max_clip_size_val"] = create_slider_value_label(row1, "90", 4)
        self._widgets["max_clip_size_val"].pack(side="right")
        self._widgets["max_clip_size"] = ctk.CTkSlider(
            row1, from_=10, to=180, number_of_steps=17,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=lambda v: self._on_slider_change("max_clip_size", int(v))
        )
        self._widgets["max_clip_size"].pack(side="right", padx=(0, 8))
        self._widgets["max_clip_size"].set(90)
        
        # Detection Model
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        model_label = ctk.CTkLabel(row2, text=t("detection_model"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        model_label.pack(side="left")
        model_tip = ctk.CTkLabel(row2, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        model_tip.pack(side="left", padx=4)
        Tooltip(model_tip, get_tooltip("detection_model"))
        
        from jasna.mosaic.detection_registry import DEFAULT_DETECTION_MODEL_NAME, discover_available_detection_models
        available_models = discover_available_detection_models()
        if not available_models:
            available_models = [DEFAULT_DETECTION_MODEL_NAME]
        self._widgets["detection_model"] = ctk.CTkOptionMenu(
            row2, values=available_models,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=120, command=lambda v: self._on_setting_change("detection_model", v)
        )
        self._widgets["detection_model"].pack(side="right")
        self._widgets["detection_model"].set(available_models[0])
        
        # Detection Threshold
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        thresh_label = ctk.CTkLabel(row3, text=t("detection_threshold"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        thresh_label.pack(side="left")
        thresh_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        thresh_tip.pack(side="left", padx=4)
        Tooltip(thresh_tip, get_tooltip("detection_score_threshold"))
        
        self._widgets["detection_threshold_val"] = ctk.CTkLabel(row3, text="0.25", text_color=Colors.TEXT_PRIMARY, width=40)
        self._widgets["detection_threshold_val"].pack(side="right")
        self._widgets["detection_score_threshold"] = ctk.CTkSlider(
            row3, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160, command=lambda v: self._widgets["detection_threshold_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["detection_score_threshold"].pack(side="right", padx=(0, 8))
        self._widgets["detection_score_threshold"].set(0.25)
        
        # Toggles row - FP16 Mode and Compile BasicVSR++
        row4 = ctk.CTkFrame(inner, fg_color="transparent")
        row4.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        
        fp16_frame = ctk.CTkFrame(row4, fg_color=Colors.BG_CARD, corner_radius=6)
        fp16_frame.pack(side="left", fill="x", expand=True, padx=(0, 4))
        fp16_label = ctk.CTkLabel(fp16_frame, text=t("fp16_mode"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        fp16_label.pack(side="left", padx=12, pady=8)
        fp16_tip = ctk.CTkLabel(fp16_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        fp16_tip.pack(side="left")
        Tooltip(fp16_tip, get_tooltip("fp16_mode"))
        self._widgets["fp16_mode"] = create_compact_switch(
            fp16_frame,
            lambda: self._on_toggle_change("fp16_mode"),
            Colors.BG_CARD,
        )
        self._widgets["fp16_mode"].pack(side="right", padx=12, pady=8)
        self._widgets["fp16_mode"].select()
        
        compile_frame = ctk.CTkFrame(row4, fg_color=Colors.BG_CARD, corner_radius=6)
        compile_frame.pack(side="right", fill="x", expand=True, padx=(4, 0))
        compile_label = ctk.CTkLabel(compile_frame, text=t("compile_basicvsrpp"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        compile_label.pack(side="left", padx=12, pady=8)
        compile_tip = ctk.CTkLabel(compile_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        compile_tip.pack(side="left")
        Tooltip(compile_tip, get_tooltip("compile_basicvsrpp"))
        self._widgets["compile_basicvsrpp"] = create_compact_switch(
            compile_frame,
            lambda: self._on_toggle_change("compile_basicvsrpp"),
            Colors.BG_CARD,
        )
        self._widgets["compile_basicvsrpp"].pack(side="right", padx=12, pady=8)
        self._widgets["compile_basicvsrpp"].select()
        
        # File Conflict dropdown
        row5 = ctk.CTkFrame(inner, fg_color="transparent")
        row5.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        
        conflict_label = ctk.CTkLabel(row5, text=t("file_conflict"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        conflict_label.pack(side="left")
        conflict_tip = ctk.CTkLabel(row5, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        conflict_tip.pack(side="left", padx=4)
        Tooltip(conflict_tip, get_tooltip("file_conflict"))
        
        # Warning icon for overwrite (hidden by default)
        self._widgets["conflict_warning"] = ctk.CTkLabel(
            row5, text="⚠️", text_color=Colors.STATUS_PAUSED, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL)
        )
        
        self._widgets["file_conflict"] = ctk.CTkOptionMenu(
            row5, values=[t("file_conflict_auto_rename"), t("file_conflict_overwrite"), t("file_conflict_skip")],
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=140, command=self._on_file_conflict_changed
        )
        self._widgets["file_conflict"].pack(side="right")
        self._widgets["file_conflict"].set(t("file_conflict_auto_rename"))
        
    def _on_file_conflict_changed(self, value: str):
        """Handle file conflict setting change."""
        # Show/hide warning for overwrite
        if value == t("file_conflict_overwrite"):
            self._widgets["conflict_warning"].pack(side="right", padx=(0, 8))
            Tooltip(self._widgets["conflict_warning"], t("file_conflict_overwrite_warning"))
        else:
            self._widgets["conflict_warning"].pack_forget()
        self._mark_modified()
        
    def _build_advanced_section(self):
        section = CollapsibleSection(self._scroll, t("section_advanced"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)
        
        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        # Temporal Overlap row
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        overlap_label = ctk.CTkLabel(row1, text=t("temporal_overlap"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        overlap_label.pack(side="left")
        overlap_tooltip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        overlap_tooltip.pack(side="left", padx=4)
        Tooltip(overlap_tooltip, get_tooltip("temporal_overlap"))
        
        self._widgets["temporal_overlap_val"] = ctk.CTkLabel(row1, text="8", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL), width=30)
        self._widgets["temporal_overlap_val"].pack(side="right")
        self._widgets["temporal_overlap"] = ctk.CTkSlider(
            row1, from_=0, to=30, number_of_steps=30,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=200, command=lambda v: self._on_slider_change("temporal_overlap", int(v))
        )
        self._widgets["temporal_overlap"].pack(side="right", padx=(0, 8))
        self._widgets["temporal_overlap"].set(8)
        
        # Crossfade toggle
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        crossfade_frame = ctk.CTkFrame(row2, fg_color=Colors.BG_CARD, corner_radius=6)
        crossfade_frame.pack(fill="x")
        crossfade_label = ctk.CTkLabel(crossfade_frame, text=t("enable_crossfade"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        crossfade_label.pack(side="left", padx=12, pady=8)
        crossfade_tip = ctk.CTkLabel(crossfade_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        crossfade_tip.pack(side="left")
        Tooltip(crossfade_tip, get_tooltip("enable_crossfade"))
        self._widgets["enable_crossfade"] = create_compact_switch(
            crossfade_frame,
            lambda: self._on_toggle_change("enable_crossfade"),
            Colors.BG_CARD,
        )
        self._widgets["enable_crossfade"].pack(side="right", padx=12, pady=8)
        self._widgets["enable_crossfade"].select()
        
        # Denoising Strength
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        strength_label = ctk.CTkLabel(row3, text=t("denoise_strength"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        strength_label.pack(side="left")
        strength_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        strength_tip.pack(side="left", padx=4)
        Tooltip(strength_tip, get_tooltip("denoise_strength"))
        
        # Store mapping from display to internal values
        self._denoise_strength_values = {
            t("denoise_none"): "none",
            t("denoise_low"): "low",
            t("denoise_medium"): "medium",
            t("denoise_high"): "high",
        }
        denoise_options = list(self._denoise_strength_values.keys())
        
        self._widgets["denoise_strength"] = ctk.CTkOptionMenu(
            row3, values=denoise_options,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=120, command=lambda v: self._on_setting_change("denoise_strength", self._denoise_strength_values[v])
        )
        self._widgets["denoise_strength"].pack(side="right")
        self._widgets["denoise_strength"].set(t("denoise_none"))
        
        # Denoise Step
        row4 = ctk.CTkFrame(inner, fg_color="transparent")
        row4.pack(fill="x")
        
        step_label = ctk.CTkLabel(row4, text=t("denoise_step"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        step_label.pack(side="left")
        step_tip = ctk.CTkLabel(row4, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        step_tip.pack(side="left", padx=4)
        Tooltip(step_tip, get_tooltip("denoise_step"))
        
        # Store mapping from display to internal values
        self._denoise_step_values = {
            t("after_primary"): "after_primary",
            t("after_secondary"): "after_secondary",
        }
        step_options = list(self._denoise_step_values.keys())
        
        self._widgets["denoise_step"] = ctk.CTkOptionMenu(
            row4, values=step_options,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY, text_color=Colors.TEXT_PRIMARY,
            width=140, command=lambda v: self._on_setting_change("denoise_step", self._denoise_step_values[v])
        )
        self._widgets["denoise_step"].pack(side="right")
        self._widgets["denoise_step"].set(t("after_primary"))

    def _build_secondary_section(self):
        section = CollapsibleSection(self._scroll, t("section_secondary"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)
        
        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        # Engine selection (radio-like)
        self._widgets["secondary_var"] = ctk.StringVar(value="none")
        
        engines_frame = ctk.CTkFrame(inner, fg_color="transparent")
        engines_frame.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        secondary_tip = ctk.CTkLabel(engines_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        secondary_tip.pack(side="right")
        Tooltip(secondary_tip, get_tooltip("secondary_restoration"))
        
        none_rb = ctk.CTkRadioButton(
            engines_frame, text=t("secondary_none"), variable=self._widgets["secondary_var"], value="none",
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._on_secondary_changed
        )
        none_rb.pack(side="left", padx=(0, 16))

        from jasna.engine_paths import UNET4X_ONNX_ENC_PATH, UNET4X_ONNX_PATH
        unet4x_available = UNET4X_ONNX_PATH.exists() or UNET4X_ONNX_ENC_PATH.exists()
        unet4x_rb = ctk.CTkRadioButton(
            engines_frame, text=f"{t('secondary_unet_4x')} ({t('secondary_unet_4x_hint')})",
            variable=self._widgets["secondary_var"], value="unet-4x",
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._on_secondary_changed,
            state="normal" if unet4x_available else "disabled",
        )
        unet4x_rb.pack(side="left", padx=(0, 16))
        Tooltip(unet4x_rb, get_tooltip("secondary_unet_4x"))
        self._unet4x_rb = unet4x_rb

        tvai_rb = ctk.CTkRadioButton(
            engines_frame, text=f"{t('secondary_tvai')} ({t('secondary_tvai_hint')})", variable=self._widgets["secondary_var"], value="tvai",
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._on_secondary_changed
        )
        tvai_rb.pack(side="left", padx=(0, 16))
        Tooltip(tvai_rb, get_tooltip("secondary_tvai"))
        
        rtx_rb = ctk.CTkRadioButton(
            engines_frame, text=f"{t('secondary_rtx_super_res')} ({t('secondary_rtx_hint')})", variable=self._widgets["secondary_var"], value="rtx-super-res",
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._on_secondary_changed
        )
        rtx_rb.pack(side="left")
        Tooltip(rtx_rb, get_tooltip("secondary_rtx"))

        # TVAI options (hidden by default)
        self._tvai_frame = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)
        
        tvai_inner = ctk.CTkFrame(self._tvai_frame, fg_color="transparent")
        tvai_inner.pack(fill="x", padx=12, pady=12)
        
        # TVAI ffmpeg path
        tvai_path_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_path_row.pack(fill="x", pady=(0, 8))
        tvai_path_label = ctk.CTkLabel(tvai_path_row, text=t("ffmpeg_path"), text_color=Colors.TEXT_PRIMARY)
        tvai_path_label.pack(side="left")
        tvai_path_tip = ctk.CTkLabel(tvai_path_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        tvai_path_tip.pack(side="left", padx=4)
        Tooltip(tvai_path_tip, get_tooltip("tvai_ffmpeg_path"))
        
        tvai_path_input_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_path_input_row.pack(fill="x", pady=(0, 8))
        self._widgets["tvai_ffmpeg_path"] = ctk.CTkEntry(
            tvai_path_input_row, fg_color=Colors.BG_PANEL, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY
        )
        self._widgets["tvai_ffmpeg_path"].pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._widgets["tvai_ffmpeg_path"].insert(0, r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe")
        
        tvai_browse_btn = ctk.CTkButton(
            tvai_path_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_PANEL, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_tvai_ffmpeg
        )
        tvai_browse_btn.pack(side="right")
        
        # TVAI model
        tvai_model_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_model_row.pack(fill="x", pady=(0, 8))
        tvai_model_label = ctk.CTkLabel(tvai_model_row, text=t("model"), text_color=Colors.TEXT_PRIMARY)
        tvai_model_label.pack(side="left")
        tvai_model_tip = ctk.CTkLabel(tvai_model_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        tvai_model_tip.pack(side="left", padx=4)
        Tooltip(tvai_model_tip, get_tooltip("tvai_model"))
        self._widgets["tvai_model"] = ctk.CTkOptionMenu(
            tvai_model_row, values=["iris-2", "iris-3", "prob-4", "nyx-1"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=100
        )
        self._widgets["tvai_model"].pack(side="right")
        self._widgets["tvai_model"].set("iris-2")
        
        # TVAI scale
        tvai_scale_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_scale_row.pack(fill="x", pady=(0, 8))
        tvai_scale_label = ctk.CTkLabel(tvai_scale_row, text=t("scale"), text_color=Colors.TEXT_PRIMARY)
        tvai_scale_label.pack(side="left")
        tvai_scale_tip = ctk.CTkLabel(tvai_scale_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        tvai_scale_tip.pack(side="left", padx=4)
        Tooltip(tvai_scale_tip, get_tooltip("tvai_scale"))
        self._widgets["tvai_scale"] = ctk.CTkOptionMenu(
            tvai_scale_row, values=["1x", "2x", "4x"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=80
        )
        self._widgets["tvai_scale"].pack(side="right")
        self._widgets["tvai_scale"].set("4x")
        
        # TVAI workers
        tvai_workers_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_workers_row.pack(fill="x")
        tvai_workers_label = ctk.CTkLabel(tvai_workers_row, text=t("workers"), text_color=Colors.TEXT_PRIMARY)
        tvai_workers_label.pack(side="left")
        tvai_workers_tip = ctk.CTkLabel(tvai_workers_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        tvai_workers_tip.pack(side="left", padx=4)
        Tooltip(tvai_workers_tip, get_tooltip("tvai_workers"))
        self._widgets["tvai_workers_val"] = ctk.CTkLabel(tvai_workers_row, text="2", text_color=Colors.TEXT_PRIMARY, width=20)
        self._widgets["tvai_workers_val"].pack(side="right")
        self._widgets["tvai_workers"] = ctk.CTkSlider(
            tvai_workers_row, from_=1, to=8, number_of_steps=7,
            fg_color=Colors.BG_PANEL, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=120, command=lambda v: self._widgets["tvai_workers_val"].configure(text=str(int(v)))
        )
        self._widgets["tvai_workers"].pack(side="right", padx=(0, 8))
        self._widgets["tvai_workers"].set(2)
        
        # RTX Super Res options (hidden by default)
        self._rtx_frame = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)

        rtx_inner = ctk.CTkFrame(self._rtx_frame, fg_color="transparent")
        rtx_inner.pack(fill="x", padx=12, pady=12)

        # RTX scale
        rtx_scale_row = ctk.CTkFrame(rtx_inner, fg_color="transparent")
        rtx_scale_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(rtx_scale_row, text=t("rtx_scale"), text_color=Colors.TEXT_PRIMARY).pack(side="left")
        rtx_scale_tip = ctk.CTkLabel(rtx_scale_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        rtx_scale_tip.pack(side="left", padx=4)
        Tooltip(rtx_scale_tip, get_tooltip("rtx_scale"))
        self._widgets["rtx_scale"] = ctk.CTkOptionMenu(
            rtx_scale_row, values=["2x", "4x"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=80
        )
        self._widgets["rtx_scale"].pack(side="right")
        self._widgets["rtx_scale"].set("4x")

        # RTX quality
        rtx_quality_row = ctk.CTkFrame(rtx_inner, fg_color="transparent")
        rtx_quality_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(rtx_quality_row, text=t("rtx_quality"), text_color=Colors.TEXT_PRIMARY).pack(side="left")
        rtx_quality_tip = ctk.CTkLabel(rtx_quality_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        rtx_quality_tip.pack(side="left", padx=4)
        Tooltip(rtx_quality_tip, get_tooltip("rtx_quality"))
        self._widgets["rtx_quality"] = ctk.CTkOptionMenu(
            rtx_quality_row, values=["Low", "Medium", "High", "Ultra"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=100
        )
        self._widgets["rtx_quality"].pack(side="right")
        self._widgets["rtx_quality"].set("High")

        # RTX denoise
        rtx_denoise_row = ctk.CTkFrame(rtx_inner, fg_color="transparent")
        rtx_denoise_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(rtx_denoise_row, text=t("rtx_denoise"), text_color=Colors.TEXT_PRIMARY).pack(side="left")
        rtx_denoise_tip = ctk.CTkLabel(rtx_denoise_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        rtx_denoise_tip.pack(side="left", padx=4)
        Tooltip(rtx_denoise_tip, get_tooltip("rtx_denoise"))
        self._widgets["rtx_denoise"] = ctk.CTkOptionMenu(
            rtx_denoise_row, values=["None", "Low", "Medium", "High", "Ultra"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=100
        )
        self._widgets["rtx_denoise"].pack(side="right")
        self._widgets["rtx_denoise"].set("Medium")

        # RTX deblur
        rtx_deblur_row = ctk.CTkFrame(rtx_inner, fg_color="transparent")
        rtx_deblur_row.pack(fill="x")
        ctk.CTkLabel(rtx_deblur_row, text=t("rtx_deblur"), text_color=Colors.TEXT_PRIMARY).pack(side="left")
        rtx_deblur_tip = ctk.CTkLabel(rtx_deblur_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        rtx_deblur_tip.pack(side="left", padx=4)
        Tooltip(rtx_deblur_tip, get_tooltip("rtx_deblur"))
        self._widgets["rtx_deblur"] = ctk.CTkOptionMenu(
            rtx_deblur_row, values=["None", "Low", "Medium", "High", "Ultra"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=100
        )
        self._widgets["rtx_deblur"].pack(side="right")
        self._widgets["rtx_deblur"].set("None")

    def _browse_tvai_ffmpeg(self):
        filepath = filedialog.askopenfilename(
            title=t("dialog_select_tvai_ffmpeg"),
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            initialdir=r"C:\Program Files\Topaz Labs LLC\Topaz Video"
        )
        if filepath:
            self._widgets["tvai_ffmpeg_path"].delete(0, "end")
            self._widgets["tvai_ffmpeg_path"].insert(0, filepath)
        
    def _build_image_restoration_section(self):
        section = CollapsibleSection(self._scroll, t("section_image_restoration"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        info = ctk.CTkLabel(
            inner, text=t("image_restore_hint"), text_color=Colors.STATUS_PENDING,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY), wraplength=320, justify="left",
        )
        info.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        # Model download (fixed location: model_weights/sd-15-jav)
        row_dl = ctk.CTkFrame(inner, fg_color="transparent")
        row_dl.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        dl_label = ctk.CTkLabel(row_dl, text=t("image_restore_model"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        dl_label.pack(side="left")
        self._widgets["image_restore_download_btn"] = ctk.CTkButton(
            row_dl, text=t("image_restore_download"), width=160,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._on_download_sd15,
        )
        self._widgets["image_restore_download_btn"].pack(side="right")

        row_interactive = ctk.CTkFrame(inner, fg_color="transparent")
        row_interactive.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._widgets["image_restore_interactive_btn"] = ctk.CTkButton(
            row_interactive,
            text=t("image_restore_interactive"),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
            command=self._open_interactive_image_restore,
        )
        self._widgets["image_restore_interactive_btn"].pack(fill="x")

        # Steps slider (5-60, step 5)
        row_steps = ctk.CTkFrame(inner, fg_color="transparent")
        row_steps.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        steps_label = ctk.CTkLabel(row_steps, text=t("image_restore_steps"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        steps_label.pack(side="left")
        steps_tip = ctk.CTkLabel(row_steps, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        steps_tip.pack(side="left", padx=4)
        Tooltip(steps_tip, get_tooltip("image_restore_steps"))
        self._widgets["image_restore_steps_val"] = ctk.CTkLabel(row_steps, text="25", text_color=Colors.TEXT_PRIMARY, width=40)
        self._widgets["image_restore_steps_val"].pack(side="right")
        self._widgets["image_restore_steps"] = ctk.CTkSlider(
            row_steps, from_=5, to=60, number_of_steps=11,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=180, command=lambda v: self._on_slider_change("image_restore_steps", int(v)),
        )
        self._widgets["image_restore_steps"].pack(side="right", padx=(0, 8))
        self._widgets["image_restore_steps"].set(25)

        # Strength slider (0.1-0.7, step 0.05)
        row_str = ctk.CTkFrame(inner, fg_color="transparent")
        row_str.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        str_label = ctk.CTkLabel(row_str, text=t("image_restore_strength"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        str_label.pack(side="left")
        str_tip = ctk.CTkLabel(row_str, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        str_tip.pack(side="left", padx=4)
        Tooltip(str_tip, get_tooltip("image_restore_strength"))
        self._widgets["image_restore_strength_val"] = ctk.CTkLabel(row_str, text="0.60", text_color=Colors.TEXT_PRIMARY, width=40)
        self._widgets["image_restore_strength_val"].pack(side="right")
        self._widgets["image_restore_strength"] = ctk.CTkSlider(
            row_str, from_=0.1, to=0.7, number_of_steps=12,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=180, command=lambda v: self._widgets["image_restore_strength_val"].configure(text=f"{v:.2f}"),
        )
        self._widgets["image_restore_strength"].pack(side="right", padx=(0, 8))
        self._widgets["image_restore_strength"].set(0.6)

        # Variants slider (1-8, step 1)
        row_var = ctk.CTkFrame(inner, fg_color="transparent")
        row_var.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        var_label = ctk.CTkLabel(row_var, text=t("image_restore_variants"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        var_label.pack(side="left")
        var_tip = ctk.CTkLabel(row_var, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        var_tip.pack(side="left", padx=4)
        Tooltip(var_tip, get_tooltip("image_restore_variants"))
        self._widgets["image_restore_variants_val"] = ctk.CTkLabel(row_var, text="1", text_color=Colors.TEXT_PRIMARY, width=40)
        self._widgets["image_restore_variants_val"].pack(side="right")
        self._widgets["image_restore_variants"] = ctk.CTkSlider(
            row_var, from_=1, to=8, number_of_steps=7,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=180, command=lambda v: self._on_slider_change("image_restore_variants", int(v)),
        )
        self._widgets["image_restore_variants"].pack(side="right", padx=(0, 8))
        self._widgets["image_restore_variants"].set(1)

        # Seed entry + FreeU toggle
        row_last = ctk.CTkFrame(inner, fg_color="transparent")
        row_last.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

        seed_frame = ctk.CTkFrame(row_last, fg_color=Colors.BG_CARD, corner_radius=6)
        seed_frame.pack(side="left", fill="x", expand=True, padx=(0, 4))
        seed_label = ctk.CTkLabel(seed_frame, text=t("image_restore_seed"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        seed_label.pack(side="left", padx=(12, 4), pady=8)
        seed_tip = ctk.CTkLabel(seed_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        seed_tip.pack(side="left")
        Tooltip(seed_tip, get_tooltip("image_restore_seed"))
        self._widgets["image_restore_seed"] = ctk.CTkEntry(
            seed_frame, width=70, fg_color=Colors.BG_PANEL, text_color=Colors.TEXT_PRIMARY, border_color=Colors.BORDER_LIGHT,
        )
        self._widgets["image_restore_seed"].pack(side="right", padx=12, pady=8)
        self._widgets["image_restore_seed"].insert(0, "0")

        freeu_frame = ctk.CTkFrame(row_last, fg_color=Colors.BG_CARD, corner_radius=6)
        freeu_frame.pack(side="right", fill="x", expand=True, padx=(4, 0))
        freeu_label = ctk.CTkLabel(freeu_frame, text=t("image_restore_freeu"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        freeu_label.pack(side="left", padx=12, pady=8)
        freeu_tip = ctk.CTkLabel(freeu_frame, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        freeu_tip.pack(side="left")
        Tooltip(freeu_tip, get_tooltip("image_restore_freeu"))
        self._widgets["image_restore_freeu"] = create_compact_switch(
            freeu_frame,
            lambda: self._on_toggle_change("image_restore_freeu"),
            Colors.BG_CARD,
        )
        self._widgets["image_restore_freeu"].pack(side="right", padx=12, pady=8)
        self._widgets["image_restore_freeu"].select()

        self._refresh_sd15_download_state()

    def _refresh_sd15_download_state(self):
        """Grey out the download button when the model is already installed."""
        from jasna.engine_paths import SD15_DIR
        from jasna.restorer.sd15_download import bundle_present

        btn = self._widgets.get("image_restore_download_btn")
        if btn is None:
            return
        if bundle_present(SD15_DIR):
            btn.configure(state="disabled", text=t("image_restore_installed"))
        else:
            btn.configure(state="normal", text=t("image_restore_download"))

    def _on_download_sd15(self):
        import threading
        from tkinter import messagebox

        from jasna.engine_paths import SD15_DIR, SD15_HF_REPO
        from jasna.restorer.sd15_download import bundle_present, download_sd15_bundle

        if bundle_present(SD15_DIR):
            self._refresh_sd15_download_state()
            return
        if not messagebox.askyesno(
            t("section_image_restoration"),
            t("image_restore_download_confirm", repo=SD15_HF_REPO),
        ):
            return

        btn = self._widgets["image_restore_download_btn"]
        btn.configure(state="disabled", text=t("image_restore_downloading"))
        progress_lock = threading.Lock()
        last_percent = -1

        def progress(downloaded: int, total: int | None):
            nonlocal last_percent
            if not total:
                return
            percent = max(0, min(100, int(downloaded * 100 / total)))
            with progress_lock:
                if percent == last_percent:
                    return
                last_percent = percent

            def update_button():
                btn.configure(text=f"{t('image_restore_downloading')} {percent}%")

            btn.after(0, update_button)

        def worker():
            error = None
            try:
                download_sd15_bundle(SD15_DIR, SD15_HF_REPO, progress_callback=progress)
            except Exception as exc:  # surface failure to the user
                error = str(exc)

            def done():
                if error:
                    btn.configure(state="normal", text=t("image_restore_download"))
                    self._show_toast(error, "error")
                else:
                    self._show_toast(t("image_restore_download_done"), "success")
                    self._refresh_sd15_download_state()

            btn.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def set_on_interactive_image_restore(self, callback: callable):
        self._on_interactive_image_restore = callback

    def _open_interactive_image_restore(self):
        if self._on_interactive_image_restore:
            self._on_interactive_image_restore()

    def _build_encoding_section(self):
        section = CollapsibleSection(self._scroll, t("section_encoding"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)
        
        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        # Codec
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        codec_label = ctk.CTkLabel(row1, text=t("codec"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        codec_label.pack(side="left")
        codec_tip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        codec_tip.pack(side="left", padx=4)
        Tooltip(codec_tip, get_tooltip("codec"))
        self._widgets["codec"] = ctk.CTkOptionMenu(
            row1, values=list(CODEC_LABEL_TO_CANONICAL),
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY, width=120,
            command=self._on_codec_changed,
        )
        self._widgets["codec"].pack(side="right")
        self._widgets["codec"].set(CODEC_CANONICAL_TO_LABEL["hevc"])
        self._active_codec = "hevc"
        
        # Quality/CQ
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        cq_label = ctk.CTkLabel(row2, text=t("quality_cq"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        cq_label.pack(side="left")
        cq_tip = ctk.CTkLabel(row2, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        cq_tip.pack(side="left", padx=4)
        Tooltip(cq_tip, get_tooltip("encoder_cq"))
        
        self._widgets["encoder_cq_val"] = ctk.CTkLabel(row2, text="22", text_color=Colors.TEXT_PRIMARY, width=30)
        self._widgets["encoder_cq_val"].pack(side="right")
        self._widgets["encoder_cq"] = ctk.CTkSlider(
            row2, from_=15, to=35, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160, command=lambda v: self._widgets["encoder_cq_val"].configure(text=str(int(v)))
        )
        self._widgets["encoder_cq"].pack(side="right", padx=(0, 8))
        self._widgets["encoder_cq"].set(22)

        # Optional exact 60/59.94 -> 30/29.97 frame-rate retargeting.
        retarget_row = ctk.CTkFrame(inner, fg_color="transparent")
        retarget_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        retarget_label = ctk.CTkLabel(
            retarget_row,
            text=t("retarget_high_fps"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        retarget_label.pack(side="left")
        retarget_tip = ctk.CTkLabel(
            retarget_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        retarget_tip.pack(side="left", padx=4)
        Tooltip(retarget_tip, get_tooltip("retarget_high_fps"))
        self._widgets["retarget_high_fps"] = create_compact_switch(
            retarget_row,
            lambda: self._on_toggle_change("retarget_high_fps"),
            Colors.BG_PANEL,
        )
        self._widgets["retarget_high_fps"].pack(side="right")
        
        # Custom args
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x")
        
        args_label = ctk.CTkLabel(row3, text=t("custom_args"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        args_label.pack(side="left", anchor="w")
        args_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        args_tip.pack(side="left", padx=4)
        Tooltip(args_tip, get_tooltip("encoder_custom_args"))
        
        args_row = ctk.CTkFrame(inner, fg_color="transparent")
        args_row.pack(fill="x", pady=(4, 0))
        self._widgets["encoder_custom_args"] = ctk.CTkEntry(
            args_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("placeholder_encoder_args")
        )
        self._widgets["encoder_custom_args"].pack(fill="x")

        # LUT (color correction)
        lut_row = ctk.CTkFrame(inner, fg_color="transparent")
        lut_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        lut_label = ctk.CTkLabel(lut_row, text=t("lut_path"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        lut_label.pack(side="left")
        lut_tip = ctk.CTkLabel(lut_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        lut_tip.pack(side="left", padx=4)
        Tooltip(lut_tip, get_tooltip("lut_path"))

        lut_input_row = ctk.CTkFrame(inner, fg_color="transparent")
        lut_input_row.pack(fill="x", pady=(4, 0))
        self._widgets["lut_path"] = ctk.CTkEntry(
            lut_input_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("lut_path_placeholder"),
        )
        self._widgets["lut_path"].pack(side="left", fill="x", expand=True, padx=(0, 4))

        lut_browse_btn = ctk.CTkButton(
            lut_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_lut_path,
        )
        lut_browse_btn.pack(side="right")

    def _on_codec_changed(self, label: str):
        new_codec = CODEC_LABEL_TO_CANONICAL[label]
        old_codec = getattr(self, "_active_codec", "hevc")
        cq = translate_cq_for_codec(
            int(self._widgets["encoder_cq"].get()), old_codec, new_codec
        )
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._active_codec = new_codec
        self._mark_modified()

    def _browse_lut_path(self):
        filepath = filedialog.askopenfilename(
            title=t("dialog_select_lut"),
            filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")],
        )
        if filepath:
            self._widgets["lut_path"].delete(0, "end")
            self._widgets["lut_path"].insert(0, filepath)

    def _build_post_export_section(self):
        section = CollapsibleSection(self._scroll, t("section_post_export_action"), expanded=True)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x")

        action_label = ctk.CTkLabel(row1, text=t("post_export_action"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        action_label.pack(side="left")
        action_tip = ctk.CTkLabel(row1, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        action_tip.pack(side="left", padx=4)
        Tooltip(action_tip, get_tooltip("post_export_action"))

        self._widgets["post_export_action"] = ctk.CTkOptionMenu(
            row1,
            values=[t("post_export_none"), t("post_export_shutdown"), t("post_export_command")],
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=160,
            command=self._on_post_export_action_changed,
        )
        self._widgets["post_export_action"].pack(side="right")
        self._widgets["post_export_action"].set(t("post_export_none"))

        self._post_export_command_frame = ctk.CTkFrame(inner, fg_color="transparent")
        command_label = ctk.CTkLabel(
            self._post_export_command_frame,
            text=t("post_export_command"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        command_label.pack(anchor="w", pady=(Sizing.PADDING_SMALL, 4))
        self._widgets["post_export_command"] = ctk.CTkEntry(
            self._post_export_command_frame,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text=t("post_export_command_placeholder"),
        )
        self._widgets["post_export_command"].pack(fill="x")
        self._widgets["post_export_command"].bind("<KeyRelease>", lambda _event: self._mark_modified())

    def _on_post_export_action_changed(self, _value: str):
        if self._widgets["post_export_action"].get() == t("post_export_command"):
            self._post_export_command_frame.pack(fill="x")
        else:
            self._post_export_command_frame.pack_forget()
        self._mark_modified()

    def _on_preset_changed(self, preset_display_name: str):
        # Strip modified indicator if present
        if preset_display_name.endswith(" (Modified)*"):
            return  # User re-selected current modified preset
        # Convert display name to actual name
        preset_name = self._display_to_name.get(preset_display_name, preset_display_name)
        self._apply_preset(preset_name)
        
    def _apply_preset(self, preset_name: str):
        self._applying_preset = True  # Prevent modification tracking
        preset = self._preset_manager.get_preset(preset_name)
        if preset is None:
            preset_name = "Default"
            preset = self._preset_manager.get_preset(preset_name)
            
        self._current_preset = preset_name
        self._saved_preset_settings = AppSettings(**asdict(preset))  # Deep copy
        self._is_modified = False
        
        # Save as last selected
        self._preset_manager.set_last_selected(preset_name)
        
        # Update UI state
        self._update_button_states()
        # Convert actual name to display name
        display_name = preset_name
        if self._preset_manager.is_factory_preset(preset_name):
            display_name = f"🔒 {preset_name}"
        self._preset_dropdown.set(display_name)
        
        # Apply settings to widgets
        self._widgets["max_clip_size"].set(preset.max_clip_size)
        self._widgets["max_clip_size_val"].configure(text=str(preset.max_clip_size))
        self._widgets["temporal_overlap"].set(preset.temporal_overlap)
        self._widgets["temporal_overlap_val"].configure(text=str(preset.temporal_overlap))
        
        if preset.enable_crossfade:
            self._widgets["enable_crossfade"].select()
        else:
            self._widgets["enable_crossfade"].deselect()
            
        if preset.fp16_mode:
            self._widgets["fp16_mode"].select()
        else:
            self._widgets["fp16_mode"].deselect()
            
        if preset.compile_basicvsrpp:
            self._widgets["compile_basicvsrpp"].select()
        else:
            self._widgets["compile_basicvsrpp"].deselect()
            
        det_model = preset.detection_model
        det_threshold = preset.detection_score_threshold
        if det_model not in self._widgets["detection_model"].cget("values"):
            from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
            det_model = "rfdetr-v5"
            det_threshold = max(det_threshold, RfDetrMosaicDetectionModel.DEFAULT_SCORE_THRESHOLD)
        self._widgets["detection_model"].set(det_model)
        self._widgets["detection_score_threshold"].set(det_threshold)
        self._widgets["detection_threshold_val"].configure(text=f"{det_threshold:.2f}")
        
        # Map internal values to translated display values
        denoise_strength_display = {
            "none": t("denoise_none"),
            "low": t("denoise_low"),
            "medium": t("denoise_medium"),
            "high": t("denoise_high"),
        }
        denoise_step_display = {
            "after_primary": t("after_primary"),
            "after_secondary": t("after_secondary"),
        }
        
        self._widgets["denoise_strength"].set(denoise_strength_display.get(preset.denoise_strength, t("denoise_none")))
        self._widgets["denoise_step"].set(denoise_step_display.get(preset.denoise_step, t("after_primary")))
        
        self._widgets["secondary_var"].set(preset.secondary_restoration)
            
        self._widgets["tvai_ffmpeg_path"].delete(0, "end")
        self._widgets["tvai_ffmpeg_path"].insert(0, preset.tvai_ffmpeg_path)
        self._widgets["tvai_model"].set(preset.tvai_model)
        self._widgets["tvai_scale"].set(f"{preset.tvai_scale}x")
        self._widgets["tvai_workers"].set(preset.tvai_workers)
        self._widgets["tvai_workers_val"].configure(text=str(preset.tvai_workers))

        self._widgets["rtx_scale"].set(f"{getattr(preset, 'rtx_scale', 4)}x")
        self._widgets["rtx_quality"].set(getattr(preset, "rtx_quality", "high").capitalize())
        self._widgets["rtx_denoise"].set(getattr(preset, "rtx_denoise", "medium").capitalize())
        self._widgets["rtx_deblur"].set(getattr(preset, "rtx_deblur", "none").capitalize())
        
        self._widgets["image_restore_steps"].set(preset.image_restore_steps)
        self._widgets["image_restore_steps_val"].configure(text=str(preset.image_restore_steps))
        self._widgets["image_restore_strength"].set(preset.image_restore_strength)
        self._widgets["image_restore_strength_val"].configure(text=f"{preset.image_restore_strength:.2f}")
        self._widgets["image_restore_variants"].set(preset.image_restore_variants)
        self._widgets["image_restore_variants_val"].configure(text=str(preset.image_restore_variants))
        self._widgets["image_restore_seed"].delete(0, "end")
        self._widgets["image_restore_seed"].insert(0, str(preset.image_restore_seed))
        if preset.image_restore_freeu:
            self._widgets["image_restore_freeu"].select()
        else:
            self._widgets["image_restore_freeu"].deselect()

        self._widgets["codec"].set(
            CODEC_CANONICAL_TO_LABEL.get(preset.codec, CODEC_CANONICAL_TO_LABEL["hevc"])
        )
        self._active_codec = preset.codec if preset.codec in CODEC_CANONICAL_TO_LABEL else "hevc"
        self._widgets["encoder_cq"].set(preset.encoder_cq)
        self._widgets["encoder_cq_val"].configure(text=str(preset.encoder_cq))
        self._widgets["encoder_custom_args"].delete(0, "end")
        self._widgets["encoder_custom_args"].insert(0, preset.encoder_custom_args)
        if preset.retarget_high_fps:
            self._widgets["retarget_high_fps"].select()
        else:
            self._widgets["retarget_high_fps"].deselect()

        self._widgets["lut_path"].delete(0, "end")
        self._widgets["lut_path"].insert(0, getattr(preset, "lut_path", "") or "")

        post_export_display = {
            "none": t("post_export_none"),
            "shutdown": t("post_export_shutdown"),
            "command": t("post_export_command"),
        }
        self._widgets["post_export_action"].set(post_export_display.get(getattr(preset, "post_export_action", "none"), t("post_export_none")))
        self._widgets["post_export_command"].delete(0, "end")
        self._widgets["post_export_command"].insert(0, getattr(preset, "post_export_command", "") or "")
        self._on_post_export_action_changed(self._widgets["post_export_action"].get())

        # File conflict setting
        file_conflict_display = {
            "auto_rename": t("file_conflict_auto_rename"),
            "overwrite": t("file_conflict_overwrite"),
            "skip": t("file_conflict_skip"),
        }
        self._widgets["file_conflict"].set(file_conflict_display.get(preset.file_conflict, t("file_conflict_auto_rename")))
        self._on_file_conflict_changed(self._widgets["file_conflict"].get())
        
        self._on_secondary_changed()
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
        
    def _on_slider_change(self, key: str, value: int):
        if f"{key}_val" in self._widgets:
            self._widgets[f"{key}_val"].configure(text=str(value))
        self._mark_modified()
        
    def _on_setting_change(self, key: str, value):
        self._mark_modified()
        
    def _on_toggle_change(self, key: str):
        self._mark_modified()
        
    def _on_secondary_changed(self):
        secondary = self._widgets["secondary_var"].get()
        self._tvai_frame.pack_forget()
        self._rtx_frame.pack_forget()
        
        if secondary == "tvai":
            self._tvai_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        elif secondary == "rtx-super-res":
            self._rtx_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
            
        self._settings.secondary_restoration = secondary
        
    def get_settings(self) -> AppSettings:
        # Map translated file conflict value back to internal value
        file_conflict_map = {
            t("file_conflict_auto_rename"): "auto_rename",
            t("file_conflict_overwrite"): "overwrite",
            t("file_conflict_skip"): "skip",
        }
        file_conflict = file_conflict_map.get(self._widgets["file_conflict"].get(), "auto_rename")
        
        # Map translated denoise_step value back to internal value
        denoise_step_map = {
            t("after_primary"): "after_primary",
            t("after_secondary"): "after_secondary",
        }
        denoise_step = denoise_step_map.get(self._widgets["denoise_step"].get(), "after_primary")
        
        # Map translated denoise_strength value back to internal value
        denoise_strength_map = {
            t("denoise_none"): "none",
            t("denoise_low"): "low",
            t("denoise_medium"): "medium",
            t("denoise_high"): "high",
        }
        denoise_strength = denoise_strength_map.get(self._widgets["denoise_strength"].get(), "none")

        post_export_action_map = {
            t("post_export_none"): "none",
            t("post_export_shutdown"): "shutdown",
            t("post_export_command"): "command",
        }
        post_export_action = post_export_action_map.get(self._widgets["post_export_action"].get(), "none")
        
        try:
            image_restore_seed = int(self._widgets["image_restore_seed"].get().strip() or "0")
        except ValueError:
            image_restore_seed = 0

        return AppSettings(
            batch_size=4,  # Fixed default value
            image_restore_steps=int(self._widgets["image_restore_steps"].get()),
            image_restore_strength=round(float(self._widgets["image_restore_strength"].get()), 2),
            image_restore_freeu=self._widgets["image_restore_freeu"].get() == 1,
            image_restore_seed=image_restore_seed,
            image_restore_variants=int(self._widgets["image_restore_variants"].get()),
            max_clip_size=int(self._widgets["max_clip_size"].get()),
            temporal_overlap=int(self._widgets["temporal_overlap"].get()),
            enable_crossfade=self._widgets["enable_crossfade"].get() == 1,
            fp16_mode=self._widgets["fp16_mode"].get() == 1,
            denoise_strength=denoise_strength,
            denoise_step=denoise_step,
            secondary_restoration=self._widgets["secondary_var"].get(),
            tvai_ffmpeg_path=self._widgets["tvai_ffmpeg_path"].get(),
            tvai_model=self._widgets["tvai_model"].get(),
            tvai_scale=int(self._widgets["tvai_scale"].get().replace("x", "")),
            tvai_workers=int(self._widgets["tvai_workers"].get()),
            rtx_scale=int(self._widgets["rtx_scale"].get().replace("x", "")),
            rtx_quality=self._widgets["rtx_quality"].get().lower(),
            rtx_denoise=self._widgets["rtx_denoise"].get().lower(),
            rtx_deblur=self._widgets["rtx_deblur"].get().lower(),
            detection_model=self._widgets["detection_model"].get(),
            detection_score_threshold=float(self._widgets["detection_score_threshold"].get()),
            compile_basicvsrpp=self._widgets["compile_basicvsrpp"].get() == 1,
            codec=CODEC_LABEL_TO_CANONICAL[self._widgets["codec"].get()],
            encoder_cq=int(self._widgets["encoder_cq"].get()),
            encoder_custom_args=self._widgets["encoder_custom_args"].get(),
            retarget_high_fps=self._widgets["retarget_high_fps"].get() == 1,
            file_conflict=file_conflict,
            lut_path=self._widgets["lut_path"].get().strip(),
            post_export_action=post_export_action,
            post_export_command=self._widgets["post_export_command"].get().strip(),
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
                pass  # Some widgets don't support state
