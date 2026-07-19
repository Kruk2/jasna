"""Image restoration (SD 1.5) settings section."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import create_compact_switch
from jasna.gui.locales import t
from jasna.gui.sd15_download_worker import start_sd15_download
from jasna.gui.settings_sections.widgets import create_slider_value_label, get_tooltip
from jasna.gui.theme import Colors, Fonts, Sizing


class ImageRestorationSection:
    def __init__(self, parent, widgets: dict, on_modified, show_toast, on_interactive):
        self._widgets = widgets
        self._on_modified = on_modified
        self._show_toast = show_toast
        self._on_interactive = on_interactive

        section = CollapsibleSection(parent, t("section_image_restoration"), expanded=False)
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
            command=self._on_interactive,
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
        self._widgets["image_restore_steps_val"] = create_slider_value_label(
            row_steps, "25", 4, Colors.BG_PANEL
        )
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
        self._widgets["image_restore_strength_val"] = create_slider_value_label(
            row_str, "0.60", 4, Colors.BG_PANEL
        )
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
        self._widgets["image_restore_variants_val"] = create_slider_value_label(
            row_var, "1", 4, Colors.BG_PANEL
        )
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
            self._on_modified,
            Colors.BG_CARD,
        )
        self._widgets["image_restore_freeu"].pack(side="right", padx=12, pady=8)
        self._widgets["image_restore_freeu"].select()

        self._refresh_sd15_download_state()

    def _on_slider_change(self, key: str, value: int):
        self._widgets[f"{key}_val"].configure(text=str(value))
        self._on_modified()

    def _refresh_sd15_download_state(self):
        """Grey out the download button when the model is already installed."""
        from jasna.engine_paths import SD15_DIR
        from jasna.restorer.sd15_download import bundle_present

        btn = self._widgets["image_restore_download_btn"]
        if bundle_present(SD15_DIR):
            btn.configure(state="disabled", text=t("image_restore_installed"))
        else:
            btn.configure(state="normal", text=t("image_restore_download"))

    def _on_download_sd15(self):
        from tkinter import messagebox

        from jasna.engine_paths import SD15_DIR, SD15_HF_REPO
        from jasna.restorer.sd15_download import bundle_present

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

        def on_percent(percent: int):
            btn.after(
                0,
                lambda: btn.configure(text=f"{t('image_restore_downloading')} {percent}%"),
            )

        def on_done(error: str | None):
            btn.after(0, lambda: self._finish_download(error))

        start_sd15_download(SD15_DIR, SD15_HF_REPO, on_percent, on_done)

    def _finish_download(self, error: str | None):
        btn = self._widgets["image_restore_download_btn"]
        if error:
            btn.configure(state="normal", text=t("image_restore_download"))
            self._show_toast(error, "error")
        else:
            self._show_toast(t("image_restore_download_done"), "success")
            self._refresh_sd15_download_state()

    def apply(self, preset):
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

    def collect(self) -> dict:
        try:
            image_restore_seed = int(self._widgets["image_restore_seed"].get().strip() or "0")
        except ValueError:
            image_restore_seed = 0
        return {
            "image_restore_steps": int(self._widgets["image_restore_steps"].get()),
            "image_restore_strength": round(float(self._widgets["image_restore_strength"].get()), 2),
            "image_restore_freeu": self._widgets["image_restore_freeu"].get() == 1,
            "image_restore_seed": image_restore_seed,
            "image_restore_variants": int(self._widgets["image_restore_variants"].get()),
        }
