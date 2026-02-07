"""Log panel - collapsible bottom section for processing logs."""

import customtkinter as ctk
from datetime import datetime
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t


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
            values=[t("filter_all_levels"), t("filter_errors_only"), t("filter_warnings_plus"), t("filter_info_plus")],
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
        self._filter_dropdown.set(t("filter_info_plus"))
        self._filter_level = "info"  # Default to Info+
        
        # Right side buttons
        self._save_btn = ctk.CTkButton(
            toolbar,
            text="💾",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color="transparent",
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_MUTED,
            width=24,
            height=24,
            command=self._on_save,
        )
        self._save_btn.pack(side="right", padx=2)
        
        self._copy_btn = ctk.CTkButton(
            toolbar,
            text="📋",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color="transparent",
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_MUTED,
            width=24,
            height=24,
            command=self._on_copy,
        )
        self._copy_btn.pack(side="right", padx=2)
        
        self._clear_btn = ctk.CTkButton(
            toolbar,
            text="🗑",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color="transparent",
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_MUTED,
            width=24,
            height=24,
            command=self._on_clear,
        )
        self._clear_btn.pack(side="right", padx=2)
        
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
            t("filter_all_levels"): "all",
            t("filter_errors_only"): "error",
            t("filter_warnings_plus"): "warning",
            t("filter_info_plus"): "info",
        }
        self._filter_level = filter_map.get(value, "all")
        self._refresh_display()
        
    def _refresh_display(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        
        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        filter_priority = {"all": -1, "debug": 0, "info": 1, "warning": 2, "error": 3}
        min_priority = filter_priority.get(self._filter_level, -1)
        
        for timestamp, level, message in self._entries:
            if level_priority.get(level, 0) >= min_priority:
                self._insert_entry(timestamp, level, message)
                
        self._log_text.configure(state="disabled")
        self._log_text.see("end")
        
    def _insert_entry(self, timestamp: str, level: str, message: str):
        self._log_text._textbox.insert("end", timestamp + " ", "timestamp")
        self._log_text._textbox.insert("end", level.ljust(8), level)
        self._log_text._textbox.insert("end", message + "\n", "message")
        
    def _on_save(self):
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
        )
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                for timestamp, level, message in self._entries:
                    f.write(f"{timestamp} {level.ljust(8)} {message}\n")
                    
    def _on_copy(self):
        text = ""
        for timestamp, level, message in self._entries:
            text += f"{timestamp} {level.ljust(8)} {message}\n"
        self.clipboard_clear()
        self.clipboard_append(text)
        
    def _on_clear(self):
        self._entries.clear()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        
    def add_log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        level = level.upper()
        self._entries.append((timestamp, level, message))
        
        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        filter_priority = {"all": -1, "debug": 0, "info": 1, "warning": 2, "error": 3}
        min_priority = filter_priority.get(self._filter_level, -1)
        
        if level_priority.get(level, 0) >= min_priority:
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
