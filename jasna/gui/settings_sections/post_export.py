"""Post-export action settings section."""

import customtkinter as ctk

from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import (
    Row,
    add_setting_label,
    bind_edits,
    layout_rows,
    option_menu,
    set_entry_text,
)
from jasna.gui.theme import Colors, Fonts, Sizing

_ACTIONS = ("none", "shutdown", "command")


class PostExportSection:
    def __init__(self, parent, on_modified):
        self._widgets = {}
        self._on_modified = on_modified
        self._level = "expert"

        self.frame = inner = ctk.CTkFrame(parent, fg_color="transparent")

        row1 = ctk.CTkFrame(inner, fg_color="transparent")

        add_setting_label(row1, "post_export_action")

        self._widgets["post_export_action"] = option_menu(
            row1, {action: t(f"post_export_{action}") for action in _ACTIONS}, self._on_action_changed, width=160
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
        bind_edits(self._widgets["post_export_command"], self._on_modified)

        video_command_label_row = ctk.CTkFrame(inner, fg_color="transparent")
        add_setting_label(video_command_label_row, "post_export_video_command")
        self._widgets["post_export_video_command"] = ctk.CTkEntry(
            inner,
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text=t("post_export_video_command_placeholder"),
        )
        bind_edits(self._widgets["post_export_video_command"], self._on_modified)

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
            [value for value in _ACTIONS if value != "command" or self._level == "expert" or action == "command"]
        )
        hidden = () if action == "command" else (self._command_frame,)
        layout_rows(self._rows, self._level, standard=True, ltx=True, hidden=hidden)

    def layout(self, level: str, *, standard: bool, ltx: bool) -> bool:
        """Running a command is for experts; a preset that already runs one keeps showing it."""
        self._level = level
        self._refresh()
        return True

    def set_enabled(self, enabled: bool) -> None:
        """The queue-wide action and its command stay editable while the queue runs: they are read
        when it ends."""
        self._widgets["post_export_video_command"].configure(state="normal" if enabled else "disabled")

    def _on_action_changed(self, _value: str):
        self._refresh()
        self._on_modified()

    def apply(self, preset):
        self._widgets["post_export_action"].set_value(preset.post_export_action)
        set_entry_text(self._widgets["post_export_command"], preset.post_export_command or "")
        set_entry_text(self._widgets["post_export_video_command"], preset.post_export_video_command or "")
        self._on_action_changed(self._widgets["post_export_action"].get_value())

    def collect(self) -> dict:
        return {
            "post_export_action": self._widgets["post_export_action"].get_value(),
            "post_export_command": self._widgets["post_export_command"].get().strip(),
            "post_export_video_command": self._widgets[
                "post_export_video_command"
            ].get().strip(),
        }
