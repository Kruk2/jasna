from __future__ import annotations

from dataclasses import replace

import customtkinter as ctk

from jasna.gui.locales import t
from jasna.gui.models import AppSettings
from jasna.gui.settings_sections.widgets import option_menu, set_widgets_enabled
from jasna.gui.theme import Colors, Sizing
from jasna.mosaic.detection_registry import (
    detection_model_choices,
    recommended_score_threshold,
)
from jasna.session_factory import secondary_restorations_for


def available_player_secondary_restorations() -> tuple[str, ...]:
    from jasna.accelerator import vendor_for_device
    from jasna.engine_paths import UNET4X_ONNX_ENC_PATH, UNET4X_ONNX_PATH

    supported = secondary_restorations_for(vendor_for_device())
    unet4x_available = UNET4X_ONNX_PATH.exists() or UNET4X_ONNX_ENC_PATH.exists()
    return tuple(
        value
        for value in ("none", "rtx-super-res", "unet-4x")
        if value in supported and (value != "unet-4x" or unet4x_available)
    )


class PlayerSettingsCard(ctk.CTkFrame):
    """The player's detection model, secondary restoration and detection threshold;
    ``on_changed()`` runs when a choice needs the pipeline reloaded."""

    def __init__(self, master, settings: AppSettings, on_changed) -> None:
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=Sizing.BORDER_RADIUS)
        self._on_changed = on_changed
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(
            self,
            text=t("player_detection_model"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=(12, 6), pady=10, sticky="w")
        self._model = option_menu(
            self,
            {model: model for model in detection_model_choices()},
            self._model_changed,
            width=160,
            background=Colors.BG_PANEL,
        )
        self._model.set_value(settings.detection_model)
        self._model.grid(row=0, column=1, padx=(0, 14), pady=10, sticky="w")

        ctk.CTkLabel(
            self,
            text=t("player_secondary"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=0, column=2, padx=(0, 6), pady=10, sticky="w")
        self._secondary = option_menu(
            self,
            {
                value: {
                    "none": t("secondary_none"),
                    "rtx-super-res": t("secondary_rtx_super_res"),
                    "unet-4x": t("secondary_unet_4x"),
                }[value]
                for value in available_player_secondary_restorations()
            },
            lambda _value: on_changed(),
            width=180,
            background=Colors.BG_PANEL,
        )
        self._secondary.set_value(settings.secondary_restoration)
        self._secondary.grid(row=0, column=3, padx=(0, 14), pady=10, sticky="w")

        ctk.CTkLabel(
            self,
            text=t("player_detection_threshold"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=1, column=0, padx=(12, 6), pady=(0, 10), sticky="w")
        threshold_row = ctk.CTkFrame(self, fg_color="transparent")
        threshold_row.grid(
            row=1,
            column=1,
            columnspan=3,
            padx=(0, 14),
            pady=(0, 10),
            sticky="ew",
        )
        self._threshold = ctk.CTkSlider(
            threshold_row,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self._threshold_changed,
        )
        self._threshold.pack(side="left", fill="x", expand=True)
        self._threshold.set(settings.detection_score_threshold)
        self._threshold.bind("<ButtonRelease-1>", lambda _event: on_changed())
        self._threshold_value = ctk.CTkLabel(
            threshold_row,
            text=f"{settings.detection_score_threshold:.2f}",
            width=42,
            text_color=Colors.TEXT_PRIMARY,
        )
        self._threshold_value.pack(side="left", padx=(8, 0))

    def _model_changed(self, model: str) -> None:
        threshold = recommended_score_threshold(model)
        self._threshold.set(threshold)
        self._threshold_value.configure(text=f"{threshold:.2f}")
        self._on_changed()

    def _threshold_changed(self, value: float) -> None:
        self._threshold_value.configure(text=f"{float(value):.2f}")

    def apply_to(self, settings: AppSettings) -> AppSettings:
        return replace(
            settings,
            detection_model=self._model.get_value(),
            detection_score_threshold=float(self._threshold.get()),
            secondary_restoration=self._secondary.get_value(),
        )

    def set_enabled(self, enabled: bool) -> None:
        set_widgets_enabled((self._model, self._threshold, self._secondary), enabled)
