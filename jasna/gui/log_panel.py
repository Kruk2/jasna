"""Log panel - collapsible bottom section for processing logs."""

import customtkinter as ctk
from datetime import datetime
from pathlib import Path
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t
from jasna.gui.log_export import export_log_entries_txt
from jasna.gui.log_filter import should_include_log_entry


class LogPanel(ctk.CTkFrame):
    """Collapsible log panel at the bottom of the window."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
            height=Sizing.LOG_PANEL_HEIGHT,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._entries: list[tuple[str, str, str]] = []  # (timestamp, level, message)
        self._filter_level = "all"
        
        self._build_toolbar()
        self._build_log_area()
        
    def _build_toolbar(self):
        toolbar = ctk.CTkFrame(self, fg_color=Colors.BG_CARD, height=32)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        
        label = ctk.CTkLabel(
            toolbar,
            text=t("system_output"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY, "bold"),
            text_color=Colors.TEXT_MUTED,
        )
        label.pack(side="left", padx=Sizing.PADDING_MEDIUM)
        
        # Filter dropdown
        filter_icon = ctk.CTkLabel(
            toolbar,
            text="🔽",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_MUTED,
        )
        filter_icon.pack(side="left", padx=(8, 2))
        
        self._filter_dropdown = ctk.CTkOptionMenu(
            toolbar,
            values=[t("filter_debug"), t("filter_info"), t("filter_warn"), t("filter_error")],
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color=Colors.BG_PANEL,
            button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_PANEL,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_SECONDARY,
            width=100,
            height=24,
            command=self._on_filter_changed,
        )
        self._filter_dropdown.pack(side="left")
        self._filter_dropdown.set(t("filter_info"))
        self._filter_level = "info"  # Default to Info
        
        # Right side buttons
        self._export_btn = ctk.CTkButton(
            toolbar,
            text=t("btn_export"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
            width=80,
            height=24,
            command=self._on_export,
        )
        self._export_btn.pack(side="right", padx=Sizing.PADDING_SMALL)
        
    def _build_log_area(self):
        self._log_text = ctk.CTkTextbox(
            self,
            fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
            wrap="none",
            state="disabled",
        )
        self._log_text.pack(fill="both", expand=True, padx=Sizing.PADDING_SMALL, pady=(0, Sizing.PADDING_SMALL))
        
        # Configure tags for colored log levels
        self._log_text._textbox.tag_config("INFO", foreground=Colors.LOG_INFO)
        self._log_text._textbox.tag_config("WARNING", foreground=Colors.LOG_WARNING)
        self._log_text._textbox.tag_config("ERROR", foreground=Colors.LOG_ERROR)
        self._log_text._textbox.tag_config("DEBUG", foreground=Colors.LOG_DEBUG)
        self._log_text._textbox.tag_config("timestamp", foreground=Colors.TEXT_MUTED)
        self._log_text._textbox.tag_config("message", foreground=Colors.TEXT_PRIMARY)
        
    def _on_filter_changed(self, value: str):
        # Map translated values to internal filter levels
        filter_map = {
            t("filter_debug"): "debug",
            t("filter_error"): "error",
            t("filter_warn"): "warning",
            t("filter_info"): "info",
        }
        self._filter_level = filter_map.get(value, "debug")
        self._refresh_display()
        
    def _refresh_display(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")

        for timestamp, level, message in self._entries:
            if should_include_log_entry(level=level, filter_level=self._filter_level):
                self._insert_entry(timestamp, level, message)
                
        self._log_text.configure(state="disabled")
        self._log_text.see("end")
        
    def _insert_entry(self, timestamp: str, level: str, message: str):
        self._log_text._textbox.insert("end", timestamp + " ", "timestamp")
        self._log_text._textbox.insert("end", level.ljust(8), level)
        self._log_text._textbox.insert("end", message + "\n", "message")
        
    def _on_export(self):
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not filepath:
            return

        export_log_entries_txt(Path(filepath), self._entries)
        
    def add_log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        level = level.upper()
        self._entries.append((timestamp, level, message))

        if should_include_log_entry(level=level, filter_level=self._filter_level):
            self._log_text.configure(state="normal")
            self._insert_entry(timestamp, level, message)
            self._log_text.configure(state="disabled")
            self._log_text.see("end")
            
    def info(self, message: str):
        self.add_log("INFO", message)
        
    def warning(self, message: str):
        self.add_log("WARNING", message)
        
    def error(self, message: str):
        self.add_log("ERROR", message)
        
    def debug(self, message: str):
        self.add_log("DEBUG", message)
