"""Post-export action settings section."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import add_setting_label, ValueOptionMenu, get_tooltip, pack_rows
from jasna.gui.theme import Colors, Fonts, Sizing


class PostExportSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified
        self._simple = False

        self.frame = CollapsibleSection(parent, t("section_post_export_action"), expanded=True)
        content = self.frame.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

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
            (row1, dict(fill="x")),
            (self._command_frame, dict(fill="x")),
            (video_command_label_row, dict(fill="x", pady=(Sizing.PADDING_MEDIUM, 4))),
            (self._widgets["post_export_video_command"], dict(fill="x")),
        ]
        self._video_command_rows = (video_command_label_row, self._widgets["post_export_video_command"])
        self._layout()

    def _layout(self) -> None:
        hidden = self._video_command_rows if self._simple else ()
        if self._widgets["post_export_action"].get_value() != "command":
            hidden = (*hidden, self._command_frame)
        pack_rows(self._rows, hidden)

    def set_simple(self, simple: bool) -> None:
        """Simple view keeps the queue-wide action and leaves out the per-video command."""
        self._simple = simple
        self._layout()

    def _on_action_changed(self, _value: str):
        self._layout()
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
