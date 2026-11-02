"""Post-export action settings section."""

import customtkinter as ctk

from jasna.gui.components import Tooltip
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import Row, ValueOptionMenu, add_setting_label, get_tooltip, layout_rows
from jasna.gui.theme import Colors, Fonts, Sizing


class PostExportSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified
        self._level = "expert"

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        row1 = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(row1, "post_export_action")

        self._widgets["post_export_action"] = ValueOptionMenu(
            row1,
            options={
                "none": t("post_export_none"),
                "shutdown": t("post_export_shutdown"),
                "command": t("post_export_command"),
            },
            command=self._on_action_changed,
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=160,
        )
        self._widgets["post_export_action"].pack(side="right")
        self._widgets["post_export_action"].set_value("none")

        self._command_frame = ctk.CTkFrame(inner, fg_color="transparent")
        command_label = ctk.CTkLabel(
            self._command_frame,
            text=t("post_export_command"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        command_label.pack(anchor="w", pady=(Sizing.PADDING_SMALL, 4))
        self._widgets["post_export_command"] = ctk.CTkEntry(
            self._command_frame,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text=t("post_export_command_placeholder"),
        )
        self._widgets["post_export_command"].pack(fill="x")
        self._widgets["post_export_command"].bind("<KeyRelease>", lambda _event: self._on_modified())

        video_command_label_row = ctk.CTkFrame(inner, fg_color="transparent")
        video_command_label = ctk.CTkLabel(
            video_command_label_row,
            text=t("post_export_video_command"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        video_command_label.pack(side="left")
        video_command_tip = ctk.CTkLabel(
            video_command_label_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        video_command_tip.pack(side="left", padx=4)
        Tooltip(video_command_tip, get_tooltip("post_export_video_command"))
        self._widgets["post_export_video_command"] = ctk.CTkEntry(
            inner,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text=t("post_export_video_command_placeholder"),
        )
        self._widgets["post_export_video_command"].bind(
            "<KeyRelease>", lambda _event: self._on_modified()
        )

        self._rows = [
            Row(row1, dict(fill="x"), "simple"),
            Row(self._command_frame, dict(fill="x"), "simple"),
            Row(video_command_label_row, dict(fill="x", pady=(Sizing.PADDING_MEDIUM, 4)), "expert"),
            Row(self._widgets["post_export_video_command"], dict(fill="x"), "expert"),
        ]
        self._refresh()

    def _refresh(self) -> None:
        action = self._widgets["post_export_action"].get_value()
        self._widgets["post_export_action"].offer(
            [value for value in ("none", "shutdown", "command") if value != "command" or self._level == "expert" or action == "command"]
        )
        hidden = () if action == "command" else (self._command_frame,)
        layout_rows(self._rows, self._level, standard=True, ltx=True, hidden=hidden)

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        """Running a command is for experts; a preset that already runs one keeps showing it."""
        self._level = level
        self._refresh()
        return True

    def _on_action_changed(self, _value: str):
        self._refresh()
        self._on_modified()

    def apply(self, preset):
        self._widgets["post_export_action"].set_value(preset.post_export_action)
        self._widgets["post_export_command"].delete(0, "end")
        self._widgets["post_export_command"].insert(0, preset.post_export_command or "")
        self._widgets["post_export_video_command"].delete(0, "end")
        self._widgets["post_export_video_command"].insert(
            0, preset.post_export_video_command or ""
        )
        self._on_action_changed(self._widgets["post_export_action"].get_value())

    def collect(self) -> dict:
        return {
            "post_export_action": self._widgets["post_export_action"].get_value(),
            "post_export_command": self._widgets["post_export_command"].get().strip(),
            "post_export_video_command": self._widgets[
                "post_export_video_command"
            ].get().strip(),
        }
