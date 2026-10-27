"""Secondary restoration settings section."""

import customtkinter as ctk
from tkinter import filedialog

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import create_icon
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    ValueOptionMenu,
    add_setting_label,
    create_slider_value_label,
    get_tooltip,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.session_config import AMD_UPSCALE_ENGINE_MODELS, AMD_UPSCALE_MODEL_DEFAULT

# Labels are the real checkpoint file names, so the picker reads as a weight list.
_AMD_MODEL_LABELS = {
    "x4v3": "realesr-general-x4v3",
    "lsdir-c3": "4xLSDIRCompactC3",
    "parimg-2x": "2xParimgCompact (2x)",
}


def _amd_model_options() -> dict[str, str]:
    """Value->label map for the AMD model row."""
    models = AMD_UPSCALE_ENGINE_MODELS["real-esr"]
    return {model: _AMD_MODEL_LABELS.get(model, model) for model in models}


class SecondarySection:
    def __init__(self, parent, widgets: dict):
        self._widgets = widgets

        self._previous_section = parent.pack_slaves()[-1]
        section = CollapsibleSection(parent, t("section_secondary"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        self._section = section
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
        self._unet4x_available = UNET4X_ONNX_PATH.exists() or UNET4X_ONNX_ENC_PATH.exists()
        unet4x_rb = ctk.CTkRadioButton(
            engines_frame, text=f"{t('secondary_unet_4x')} ({t('secondary_unet_4x_hint')})",
            variable=self._widgets["secondary_var"], value="unet-4x",
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._on_secondary_changed,
            state="normal" if self._unet4x_available else "disabled",
        )
        unet4x_rb.pack(side="left", padx=(0, 16))
        Tooltip(unet4x_rb, get_tooltip("secondary_unet_4x"))

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
        rtx_rb.pack(side="left", padx=(0, 16))
        Tooltip(rtx_rb, get_tooltip("secondary_rtx"))

        amd_rb = ctk.CTkRadioButton(
            engines_frame, text=f"{t('secondary_amd_upscale')} ({t('secondary_amd_hint')})",
            variable=self._widgets["secondary_var"], value="amd-upscale",
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._on_secondary_changed
        )
        amd_rb.pack(side="left")
        Tooltip(amd_rb, get_tooltip("secondary_amd"))
        self._engine_radios = {
            "none": none_rb,
            "unet-4x": unet4x_rb,
            "tvai": tvai_rb,
            "rtx-super-res": rtx_rb,
            "amd-upscale": amd_rb,
        }

        # TVAI options (hidden by default)
        self._tvai_frame = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)

        tvai_inner = ctk.CTkFrame(self._tvai_frame, fg_color="transparent")
        tvai_inner.pack(fill="x", padx=12, pady=12)

        # TVAI ffmpeg path
        tvai_path_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_path_row.pack(fill="x", pady=(0, 8))
        add_setting_label(tvai_path_row, "ffmpeg_path", "tvai_ffmpeg_path")

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
        add_setting_label(tvai_model_row, "model", "tvai_model")
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
        add_setting_label(tvai_scale_row, "scale", "tvai_scale")
        self._widgets["tvai_scale"] = ctk.CTkOptionMenu(
            tvai_scale_row, values=["1x", "2x", "4x"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=80
        )
        self._widgets["tvai_scale"].pack(side="right")
        self._widgets["tvai_scale"].set("4x")

        # TVAI Denoise Operation
        tvai_denoise_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_denoise_row.pack(fill="x", pady=(0, 8))
        tvai_denoise_label = ctk.CTkLabel(
            tvai_denoise_row,
            text=t("tvai_denoise"),
            text_color=Colors.TEXT_PRIMARY,
        )
        tvai_denoise_label.pack(side="left")
        tvai_denoise_tip = ctk.CTkLabel(
            tvai_denoise_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        tvai_denoise_tip.pack(side="left", padx=4)
        Tooltip(tvai_denoise_tip, get_tooltip("tvai_denoise"))
        self._widgets["tvai_denoise"] = ctk.BooleanVar(value=False)
        self._tvai_denoise_switch = ctk.CTkSwitch(
            tvai_denoise_row,
            text="",
            variable=self._widgets["tvai_denoise"],
            fg_color=Colors.BG_PANEL,
            progress_color=Colors.PRIMARY,
            button_color=Colors.TEXT_PRIMARY,
            width=42,
        )
        self._tvai_denoise_switch.pack(side="right")

        # TVAI workers
        tvai_workers_row = ctk.CTkFrame(tvai_inner, fg_color="transparent")
        tvai_workers_row.pack(fill="x")
        add_setting_label(tvai_workers_row, "workers", "tvai_workers")
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

        # AMD upscale options (hidden by default)
        self._amd_frame = ctk.CTkFrame(inner, fg_color=Colors.BG_CARD, corner_radius=6)

        amd_inner = ctk.CTkFrame(self._amd_frame, fg_color="transparent")
        amd_inner.pack(fill="x", padx=12, pady=12)

        # AMD model: the dropdown shows the actual weight file names.
        amd_model_row = ctk.CTkFrame(amd_inner, fg_color="transparent")
        amd_model_row.pack(fill="x", pady=(0, 8))
        add_setting_label(amd_model_row, "model", "amd_model")
        self._widgets["amd_upscale_model"] = ValueOptionMenu(
            amd_model_row,
            options=_amd_model_options(),
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=200
        )
        self._widgets["amd_upscale_model"].pack(side="right")
        self._widgets["amd_upscale_model"].set_value(AMD_UPSCALE_MODEL_DEFAULT["real-esr"])

        # AMD scale
        amd_scale_row = ctk.CTkFrame(amd_inner, fg_color="transparent")
        amd_scale_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(amd_scale_row, text=t("scale"), text_color=Colors.TEXT_PRIMARY).pack(side="left")
        self._widgets["amd_upscale_scale"] = ctk.CTkOptionMenu(
            amd_scale_row, values=["2x", "4x"],
            fg_color=Colors.BG_PANEL, button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY, width=80
        )
        self._widgets["amd_upscale_scale"].pack(side="right")
        self._widgets["amd_upscale_scale"].set("2x")

        # AMD sharpness: CAS on the restored patch only, at its final resolution.
        amd_sharpness_row = ctk.CTkFrame(amd_inner, fg_color="transparent")
        amd_sharpness_row.pack(fill="x")
        ctk.CTkLabel(amd_sharpness_row, text=t("amd_sharpness"), text_color=Colors.TEXT_PRIMARY).pack(side="left")
        amd_sharpness_tip = ctk.CTkLabel(amd_sharpness_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        amd_sharpness_tip.pack(side="left", padx=4)
        Tooltip(amd_sharpness_tip, get_tooltip("amd_sharpness_patch"))
        self._widgets["amd_upscale_sharpness"] = ctk.CTkEntry(
            amd_sharpness_row, fg_color=Colors.BG_PANEL, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, width=80
        )
        self._widgets["amd_upscale_sharpness"].pack(side="right")
        self._widgets["amd_upscale_sharpness"].insert(0, "0")

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
        self._amd_frame.pack_forget()

        if secondary == "tvai":
            self._tvai_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        elif secondary == "rtx-super-res":
            self._rtx_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        elif secondary == "amd-upscale":
            self._amd_frame.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

    def set_model(self, model: str) -> None:
        if model == "ltx":
            self._section.pack_forget()
        else:
            self._section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL), after=self._previous_section)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for value, radio in self._engine_radios.items():
            unavailable = value == "unet-4x" and not self._unet4x_available
            radio.configure(state="disabled" if unavailable else state)
        self._tvai_denoise_switch.configure(state=state)

    def apply(self, preset):
        self._widgets["secondary_var"].set(preset.secondary_restoration)

        self._widgets["tvai_ffmpeg_path"].delete(0, "end")
        self._widgets["tvai_ffmpeg_path"].insert(0, preset.tvai_ffmpeg_path)
        self._widgets["tvai_model"].set(preset.tvai_model)
        self._widgets["tvai_scale"].set(f"{preset.tvai_scale}x")
        self._widgets["tvai_workers"].set(preset.tvai_workers)
        self._widgets["tvai_workers_val"].configure(text=str(preset.tvai_workers))
        self._widgets["tvai_denoise"].set(preset.tvai_denoise)

        self._widgets["rtx_scale"].set(f"{preset.rtx_scale}x")
        self._widgets["rtx_quality"].set(preset.rtx_quality.capitalize())
        self._widgets["rtx_denoise"].set(preset.rtx_denoise.capitalize())
        self._widgets["rtx_deblur"].set(preset.rtx_deblur.capitalize())

        self._widgets["amd_upscale_model"].set_value(preset.amd_upscale_model)
        # A preset written by a build that also offered amf-sr can carry 6x/8x or a
        # -1 sharpness sentinel; neither exists for the in-process networks, so snap
        # both into the range this panel actually offers.
        amd_scale = int(preset.amd_upscale_scale)
        self._widgets["amd_upscale_scale"].set(f"{amd_scale}x" if amd_scale in (2, 4) else "2x")
        amd_sharpness = max(0.0, float(preset.amd_upscale_sharpness))
        self._widgets["amd_upscale_sharpness"].delete(0, "end")
        self._widgets["amd_upscale_sharpness"].insert(0, str(amd_sharpness))

        self._on_secondary_changed()

    def collect(self) -> dict:
        return {
            "secondary_restoration": self._widgets["secondary_var"].get(),
            "tvai_ffmpeg_path": self._widgets["tvai_ffmpeg_path"].get(),
            "tvai_model": self._widgets["tvai_model"].get(),
            "tvai_scale": int(self._widgets["tvai_scale"].get().replace("x", "")),
            "tvai_workers": int(self._widgets["tvai_workers"].get()),
            "tvai_denoise": bool(self._widgets["tvai_denoise"].get()),
            "rtx_scale": int(self._widgets["rtx_scale"].get().replace("x", "")),
            "rtx_quality": self._widgets["rtx_quality"].get().lower(),
            "rtx_denoise": self._widgets["rtx_denoise"].get().lower(),
            "rtx_deblur": self._widgets["rtx_deblur"].get().lower(),
            "amd_upscale_model": self._widgets["amd_upscale_model"].get_value(),
            "amd_upscale_scale": int(self._widgets["amd_upscale_scale"].get().replace("x", "")),
            "amd_upscale_sharpness": float(self._widgets["amd_upscale_sharpness"].get() or 0),
        }
