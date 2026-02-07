"""Reusable UI components for Jasna GUI."""

import customtkinter as ctk
import webbrowser
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t


# Buy Me a Coffee URL
BMC_URL = "https://buymeacoffee.com/Kruk2"


class BuyMeCoffeeButton(ctk.CTkButton):
    """Buy Me a Coffee button with brand styling and hover animation."""
    
    def __init__(self, master, compact: bool = False, **kwargs):
        # Keep the rocket emoji present; translate the support text
        if compact:
            text = "☕ 🚀"
            width = 40
        else:
            label = t("bmc_support")
            text = f"☕ {label} 🚀"
            width = 100
        
        super().__init__(
            master,
            text=text,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            fg_color=Colors.BMC_YELLOW,
            hover_color=Colors.BMC_YELLOW,  # Keep same, we scale on hover
            text_color=Colors.BMC_TEXT,
            corner_radius=6,
            height=28,
            width=width,
            command=self._open_bmc,
            **kwargs
        )
        
        self._original_width = width
        self._original_height = 28
        
        # Bind hover events for scale effect
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
    def _on_enter(self, event=None):
        # Scale up 1.05x on hover
        new_width = int(self._original_width * 1.05)
        new_height = int(self._original_height * 1.05)
        self.configure(width=new_width, height=new_height)
        
    def _on_leave(self, event=None):
        # Restore original size
        self.configure(width=self._original_width, height=self._original_height)
        
    def _open_bmc(self):
        webbrowser.open(BMC_URL)


class StatusPill(ctk.CTkFrame):
    """Status indicator pill shown in header."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=16,
            height=28,
            **kwargs
        )
        self.grid_propagate(False)
        
        self._indicator = ctk.CTkLabel(
            self,
            text="",
            width=8,
            height=8,
            fg_color=Colors.TEXT_SECONDARY,
            corner_radius=4,
        )
        self._indicator.grid(row=0, column=0, padx=(12, 6), pady=8)
        
        self._label = ctk.CTkLabel(
            self,
            text="IDLE",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._label.grid(row=0, column=1, padx=(0, 12), pady=4)
        
    def set_status(self, status: str, color: str):
        self._label.configure(text=status.upper())
        self._indicator.configure(fg_color=color)


class CollapsibleSection(ctk.CTkFrame):
    """Accordion-style collapsible section for settings."""
    
    def __init__(self, master, title: str, expanded: bool = True, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._expanded = expanded
        
        self._header = ctk.CTkFrame(
            self,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
            height=40,
        )
        self._header.pack(fill="x")
        self._header.pack_propagate(False)
        
        self._arrow = ctk.CTkLabel(
            self._header,
            text="▼" if expanded else "▶",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_SECONDARY,
            width=20,
        )
        self._arrow.pack(side="left", padx=(12, 4), pady=8)
        
        self._title_label = ctk.CTkLabel(
            self._header,
            text=title.upper(),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._title_label.pack(side="left", fill="x", expand=True, pady=8)
        
        self._content = ctk.CTkFrame(
            self,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
        )
        if expanded:
            self._content.pack(fill="x", pady=(2, 0))
        
        self._header.bind("<Button-1>", self._toggle)
        self._arrow.bind("<Button-1>", self._toggle)
        self._title_label.bind("<Button-1>", self._toggle)
        
    def _toggle(self, event=None):
        self._expanded = not self._expanded
        self._arrow.configure(text="▼" if self._expanded else "▶")
        if self._expanded:
            self._content.pack(fill="x", pady=(2, 0))
        else:
            self._content.pack_forget()
            
    @property
    def content(self) -> ctk.CTkFrame:
        return self._content


class SettingRow(ctk.CTkFrame):
    """A row containing a label and a control widget."""
    
    def __init__(self, master, label: str, tooltip: str = "", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self._label = ctk.CTkLabel(
            self,
            text=label,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_SECONDARY,
            anchor="w",
        )
        self._label.pack(side="left", padx=(0, 8))
        
        if tooltip:
            self._tooltip_icon = ctk.CTkLabel(
                self,
                text="ⓘ",
                font=(Fonts.FAMILY, Fonts.SIZE_TINY),
                text_color=Colors.TEXT_MUTED,
                cursor="hand2",
            )
            self._tooltip_icon.pack(side="left")
            
        self._control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._control_frame.pack(side="right")
        
    @property  
    def control_frame(self) -> ctk.CTkFrame:
        return self._control_frame


class JobListItem(ctk.CTkFrame):
    """Individual job item in the queue list."""
    
    def __init__(
        self,
        master,
        filename: str,
        duration: str,
        status: str,
        on_remove: callable = None,
        on_drag_start: callable = None,
        on_drag_move: callable = None,
        on_drag_end: callable = None,
        **kwargs
    ):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
            height=64,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._on_remove = on_remove
        # Drag callbacks (set by QueuePanel)
        self._on_drag_start = on_drag_start
        self._on_drag_move = on_drag_move
        self._on_drag_end = on_drag_end
        self._progress_visible = False
        self._conflict_visible = False
        
        # Main content container
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=8, pady=6)
        
        # Top row: handle + filename + duration
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x")
        
        # Conflict indicator (amber dot)
        self._conflict_dot = ctk.CTkLabel(
            top_row,
            text="●",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.STATUS_CONFLICT,
            width=16,
        )
        
        # Drag handle
        self._handle = ctk.CTkLabel(
            top_row,
            text="⋮⋮",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_MUTED,
            width=20,
            cursor="hand2",
        )
        self._handle.pack(side="left")
        # Bind drag events to handle
        self._handle.bind("<ButtonPress-1>", self._internal_drag_start)
        self._handle.bind("<B1-Motion>", self._internal_drag_move)
        self._handle.bind("<ButtonRelease-1>", self._internal_drag_end)
        
        # Info area (filename + duration inline)
        self._info = ctk.CTkFrame(top_row, fg_color="transparent")
        self._info.pack(side="left", fill="x", expand=True, padx=4)
        
        self._filename = ctk.CTkLabel(
            self._info,
            text=filename,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._filename.pack(side="left")
        
        self._duration = ctk.CTkLabel(
            self._info,
            text=f"  •  {duration}" if duration else "",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_MUTED,
            anchor="w",
        )
        self._duration.pack(side="left")
        
        # Bottom row: status
        bottom_row = ctk.CTkFrame(content, fg_color="transparent")
        bottom_row.pack(fill="x", pady=(4, 0))
        
        # Status area
        self._status_frame = ctk.CTkFrame(bottom_row, fg_color="transparent")
        self._status_frame.pack(side="left")
        
        self._status_icon = ctk.CTkLabel(
            self._status_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            width=16,
        )
        
        self._status_label = ctk.CTkLabel(
            self._status_frame,
            text=status,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_SECONDARY,
        )
        self._status_label.pack(side="left")
        
        # FPS / ETA small labels on the right of bottom row
        self._stats_frame = ctk.CTkFrame(bottom_row, fg_color="transparent")
        self._stats_frame.pack(side="right")

        self._fps_label = ctk.CTkLabel(
            self._stats_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_MUTED,
        )
        self._fps_label.pack(side="left", padx=(0, 8))

        self._eta_label = ctk.CTkLabel(
            self._stats_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_MUTED,
        )
        self._eta_label.pack(side="left")
        
        # Progress bar (hidden by default)
        self._progress = ctk.CTkProgressBar(
            self,
            height=3,
            fg_color=Colors.BG_PANEL,
            progress_color=Colors.PRIMARY,
        )
        
        # Store references for hover binding
        self._top_row = top_row
        self._bottom_row = bottom_row
        
        self._remove_btn = ctk.CTkButton(
            self,
            text="✕",
            width=24,
            height=24,
            fg_color="transparent",
            hover_color=Colors.STATUS_ERROR,
            text_color=Colors.TEXT_MUTED,
            command=self._handle_remove,
        )
        # By default items are removable; can be toggled when queue is running
        self._removable = True
        
        # Show remove on hover
        # Show remove on hover. Bind both Enter and Leave on children to avoid
        # flicker when moving between the parent and its child widgets.
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        # Debounce hide to avoid flicker
        self._hide_after_id = None

        child_widgets = [
            self._handle, self._info, self._filename, self._duration,
            self._status_frame, self._status_label, self._top_row, self._bottom_row,
        ]
        for child in child_widgets:
            try:
                child.bind("<Enter>", self._on_enter)
                child.bind("<Leave>", self._on_leave)
            except Exception:
                pass
        # Ensure remove button keeps the enter binding so moving onto it still shows
        self._remove_btn.bind("<Enter>", self._on_enter)
        self._remove_btn.bind("<Leave>", self._on_leave)
        
    def _on_enter(self, event=None):
        # Only show remove button if the pointer is actually inside this widget
        if not getattr(self, "_removable", True):
            return
        # Cancel any pending hide
        try:
            if getattr(self, "_hide_after_id", None):
                self.after_cancel(self._hide_after_id)
                self._hide_after_id = None
        except Exception:
            pass
        try:
            x, y = self.winfo_pointerxy()
            widget_x = self.winfo_rootx()
            widget_y = self.winfo_rooty()
            widget_w = self.winfo_width()
            widget_h = self.winfo_height()
            if widget_x <= x <= widget_x + widget_w and widget_y <= y <= widget_y + widget_h:
                self._remove_btn.place(relx=1.0, rely=0, anchor="ne", x=-4, y=4)
        except Exception:
            # Fallback: show button if any error querying pointer
            self._remove_btn.place(relx=1.0, rely=0, anchor="ne", x=-4, y=4)
        
    def _on_leave(self, event=None):
        # Debounced hide: schedule hide shortly to prevent flicker when moving
        # between child widgets
        try:
            if self._hide_after_id:
                self.after_cancel(self._hide_after_id)
        except Exception:
            pass

        def _hide_if_outside():
            try:
                x, y = self.winfo_pointerxy()
                widget_x = self.winfo_rootx()
                widget_y = self.winfo_rooty()
                widget_w = self.winfo_width()
                widget_h = self.winfo_height()
                if not (widget_x <= x <= widget_x + widget_w and widget_y <= y <= widget_y + widget_h):
                    self._remove_btn.place_forget()
            except Exception:
                try:
                    self._remove_btn.place_forget()
                except Exception:
                    pass

        # Schedule a short delayed check
        try:
            self._hide_after_id = self.after(80, _hide_if_outside)
        except Exception:
            _hide_if_outside()
        
    def _handle_remove(self):
        if self._on_remove:
            self._on_remove()

    # Internal drag event proxies to allow QueuePanel to handle reordering
    def _internal_drag_start(self, event):
        if callable(self._on_drag_start):
            self._on_drag_start(self, event)

    def _internal_drag_move(self, event):
        if callable(self._on_drag_move):
            self._on_drag_move(self, event)

    def _internal_drag_end(self, event):
        if callable(self._on_drag_end):
            self._on_drag_end(self, event)
            
    def set_status(self, status: str, icon: str = "", color: str = Colors.TEXT_SECONDARY):
        self._status_label.configure(text=status, text_color=color)
        if icon:
            self._status_icon.configure(text=icon, text_color=color)
            self._status_icon.pack(side="left", padx=(0, 4), before=self._status_label)
        else:
            self._status_icon.pack_forget()

    def set_removable(self, removable: bool):
        """Enable or disable the remove action for this item.

        When not removable the remove button is hidden and user cannot remove
        the item (used to protect the currently processing job).
        """
        self._removable = bool(removable)
        if not self._removable:
            try:
                self._remove_btn.place_forget()
            except Exception:
                pass
            
    def set_progress(self, value: float):
        if not self._progress_visible:
            self._progress.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            self._progress_visible = True
        # value expected in range 0.0-1.0
        try:
            self._progress.set(value)
        except Exception:
            # if value is percent (0-100), normalize
            try:
                self._progress.set(float(value) / 100.0)
            except Exception:
                pass

    def set_fps_eta(self, fps: float = 0.0, eta_seconds: float = 0.0):
        """Update small FPS and ETA labels shown on the tile."""
        if fps and fps > 0:
            self._fps_label.configure(text=f"{fps:.1f}fps")
        else:
            self._fps_label.configure(text="")

        if eta_seconds and eta_seconds > 0:
            mins, secs = divmod(int(eta_seconds), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                eta_str = f"{hours}h {mins}m"
            elif mins:
                eta_str = f"{mins}m {secs}s"
            else:
                eta_str = f"{secs}s"
            self._eta_label.configure(text=f"ETA: {eta_str}")
        else:
            self._eta_label.configure(text="")
        
    def hide_progress(self):
        self._progress.place_forget()
        self._progress_visible = False

    def set_conflict(self, has_conflict: bool, tooltip: str = ""):
        """Show or hide the conflict indicator (amber dot)."""
        if has_conflict and not self._conflict_visible:
            self._conflict_dot.pack(side="left", before=self._handle)
            self._conflict_visible = True
            if tooltip:
                # Create tooltip on hover
                self._conflict_dot.bind("<Enter>", lambda e: self._show_tooltip(tooltip))
                self._conflict_dot.bind("<Leave>", lambda e: self._hide_tooltip())
        elif not has_conflict and self._conflict_visible:
            self._conflict_dot.pack_forget()
            self._conflict_visible = False
            
    def _show_tooltip(self, text: str):
        """Show tooltip near the conflict indicator."""
        if hasattr(self, '_tooltip'):
            self._tooltip.destroy()
        self._tooltip = ctk.CTkLabel(
            self.winfo_toplevel(),
            text=text,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            corner_radius=4,
            text_color=Colors.STATUS_CONFLICT,
            padx=8,
            pady=4,
        )
        x = self._conflict_dot.winfo_rootx() + 20
        y = self._conflict_dot.winfo_rooty() - 10
        self._tooltip.place(x=x - self.winfo_toplevel().winfo_rootx(), 
                           y=y - self.winfo_toplevel().winfo_rooty())
        
    def _hide_tooltip(self):
        """Hide the tooltip."""
        if hasattr(self, '_tooltip'):
            self._tooltip.destroy()
            del self._tooltip


class LogEntry(ctk.CTkFrame):
    """Single log entry with timestamp and colored level."""
    
    def __init__(self, master, timestamp: str, level: str, message: str, **kwargs):
        super().__init__(master, fg_color="transparent", height=20, **kwargs)
        self.pack_propagate(False)
        
        level_colors = {
            "INFO": Colors.LOG_INFO,
            "WARNING": Colors.LOG_WARNING,
            "ERROR": Colors.LOG_ERROR,
            "DEBUG": Colors.LOG_DEBUG,
        }
        
        self._time = ctk.CTkLabel(
            self,
            text=timestamp,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_MUTED,
            width=80,
            anchor="w",
        )
        self._time.pack(side="left")
        
        self._level = ctk.CTkLabel(
            self,
            text=level,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY, "bold"),
            text_color=level_colors.get(level, Colors.TEXT_SECONDARY),
            width=60,
            anchor="w",
        )
        self._level.pack(side="left")
        
        self._message = ctk.CTkLabel(
            self,
            text=message,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._message.pack(side="left", fill="x", expand=True)


class Toast(ctk.CTkFrame):
    """Toast notification that auto-dismisses."""
    
    def __init__(self, master, message: str, type_: str = "info", duration_ms: int = 3000, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=8,
            border_width=1,
            border_color=Colors.BORDER,
            height=44,
            width=640,
            **kwargs
        )
        self.pack_propagate(False)
        
        colors = {
            "success": Colors.STATUS_COMPLETED,
            "error": Colors.STATUS_ERROR,
            "warning": Colors.STATUS_PAUSED,
            "info": Colors.PRIMARY,
        }
        accent = colors.get(type_, Colors.PRIMARY)
        
        indicator = ctk.CTkFrame(self, fg_color=accent, width=4, height=28, corner_radius=2)
        indicator.pack(side="left", padx=(8, 0))
        
        label = ctk.CTkLabel(
            self,
            text=message,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=560,
        )
        label.pack(side="left", fill="x", expand=True, padx=12)
        
        self.after(duration_ms, self._dismiss)
        
    def _dismiss(self):
        self.destroy()


class PresetDialog(ctk.CTkToplevel):
    """Modal dialog for creating a new preset."""
    
    def __init__(self, master, on_create: callable, existing_names: list[str], **kwargs):
        super().__init__(master, **kwargs)
        
        self.title("Create Preset")
        self.configure(fg_color=Colors.BG_MAIN)
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        
        self._on_create = on_create
        self._existing_names = [n.lower() for n in existing_names]
        self._result = None
        
        # Center on parent
        self.geometry("320x180")
        self.update_idletasks()
        parent_x = master.winfo_rootx()
        parent_y = master.winfo_rooty()
        parent_w = master.winfo_width()
        parent_h = master.winfo_height()
        x = parent_x + (parent_w - 320) // 2
        y = parent_y + (parent_h - 180) // 2
        self.geometry(f"+{x}+{y}")
        
        # Content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            content,
            text="Preset Name",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_SECONDARY,
        ).pack(anchor="w")
        
        self._entry = ctk.CTkEntry(
            content,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text="My Custom Preset",
        )
        self._entry.pack(fill="x", pady=(8, 0))
        self._entry.bind("<Return>", lambda e: self._on_ok())
        
        self._error_label = ctk.CTkLabel(
            content,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_ERROR,
        )
        self._error_label.pack(anchor="w", pady=(4, 0))
        
        # Buttons
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(20, 0))
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_SECONDARY,
            width=90,
            height=36,
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        
        ctk.CTkButton(
            btn_frame,
            text="Create",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
            width=90,
            height=36,
            command=self._on_ok,
        ).pack(side="right")
        
        self._entry.focus_set()
        
    def _on_ok(self):
        name = self._entry.get().strip()
        if not name:
            self._error_label.configure(text="Name cannot be empty")
            return
        if name.lower() in self._existing_names:
            self._error_label.configure(text="Name already exists")
            return
        
        self._on_create(name)
        self.destroy()


class ConfirmDialog(ctk.CTkToplevel):
    """Confirmation dialog."""
    
    def __init__(self, master, title: str, message: str, on_confirm: callable, **kwargs):
        super().__init__(master, **kwargs)
        
        self.title(title)
        self.configure(fg_color=Colors.BG_MAIN)
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        
        self._on_confirm = on_confirm
        
        self.geometry("320x140")
        self.update_idletasks()
        parent_x = master.winfo_rootx()
        parent_y = master.winfo_rooty()
        parent_w = master.winfo_width()
        parent_h = master.winfo_height()
        x = parent_x + (parent_w - 320) // 2
        y = parent_y + (parent_h - 140) // 2
        self.geometry(f"+{x}+{y}")
        
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            content,
            text=message,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=280,
        ).pack(pady=(0, 16))
        
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x")
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_SECONDARY,
            width=80,
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        
        ctk.CTkButton(
            btn_frame,
            text="Delete",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.STATUS_ERROR,
            hover_color="#dc2626",
            text_color=Colors.TEXT_PRIMARY,
            width=80,
            command=self._do_confirm,
        ).pack(side="right")
        
    def _do_confirm(self):
        self._on_confirm()
        self.destroy()
