"""Post-export action settings section."""

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import ValueOptionMenu, get_tooltip
from jasna.gui.theme import Colors, Fonts, Sizing


class PostExportSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified

        section = CollapsibleSection(parent, t("section_post_export_action"), expanded=True)
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

    def _on_action_changed(self, value: str):
        if value == "command":
            self._command_frame.pack(fill="x")
        else:
            self._command_frame.pack_forget()
        self._on_modified()

    def apply(self, preset):
        self._widgets["post_export_action"].set_value(preset.post_export_action)
        self._widgets["post_export_command"].delete(0, "end")
        self._widgets["post_export_command"].insert(0, preset.post_export_command or "")
        self._on_action_changed(self._widgets["post_export_action"].get_value())

    def collect(self) -> dict:
        return {
            "post_export_action": self._widgets["post_export_action"].get_value(),
            "post_export_command": self._widgets["post_export_command"].get().strip(),
        }
