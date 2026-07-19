"""Secondary restoration settings section."""

import customtkinter as ctk
from tkinter import filedialog

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import create_icon
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import create_slider_value_label, get_tooltip
from jasna.gui.theme import Colors, Fonts, Sizing


class SecondarySection:
    def __init__(self, parent, widgets: dict):
        self._widgets = widgets

        section = CollapsibleSection(parent, t("section_secondary"), expanded=False)
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
        self._widgets["tvai_workers_val"] = create_slider_value_label(
            tvai_workers_row, "2", 2, Colors.BG_CARD
        )
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

    def _on_secondary_changed(self):
        secondary = self._widgets["secondary_var"].get()
        self._tvai_frame.pack_forget()
        self._rtx_frame.pack_forget()

        if secondary == "tvai":
            self._tvai_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        elif secondary == "rtx-super-res":
            self._rtx_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

    def apply(self, preset):
        self._widgets["secondary_var"].set(preset.secondary_restoration)

        self._widgets["tvai_ffmpeg_path"].delete(0, "end")
        self._widgets["tvai_ffmpeg_path"].insert(0, preset.tvai_ffmpeg_path)
        self._widgets["tvai_model"].set(preset.tvai_model)
        self._widgets["tvai_scale"].set(f"{preset.tvai_scale}x")
        self._widgets["tvai_workers"].set(preset.tvai_workers)
        self._widgets["tvai_workers_val"].configure(text=str(preset.tvai_workers))

        self._widgets["rtx_scale"].set(f"{preset.rtx_scale}x")
        self._widgets["rtx_quality"].set(preset.rtx_quality.capitalize())
        self._widgets["rtx_denoise"].set(preset.rtx_denoise.capitalize())
        self._widgets["rtx_deblur"].set(preset.rtx_deblur.capitalize())

        self._on_secondary_changed()

    def collect(self) -> dict:
        return {
            "secondary_restoration": self._widgets["secondary_var"].get(),
            "tvai_ffmpeg_path": self._widgets["tvai_ffmpeg_path"].get(),
            "tvai_model": self._widgets["tvai_model"].get(),
            "tvai_scale": int(self._widgets["tvai_scale"].get().replace("x", "")),
            "tvai_workers": int(self._widgets["tvai_workers"].get()),
            "rtx_scale": int(self._widgets["rtx_scale"].get().replace("x", "")),
            "rtx_quality": self._widgets["rtx_quality"].get().lower(),
            "rtx_denoise": self._widgets["rtx_denoise"].get().lower(),
            "rtx_deblur": self._widgets["rtx_deblur"].get().lower(),
        }
