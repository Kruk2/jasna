"""Reusable UI components for Jasna GUI."""

import logging
import tkinter
import customtkinter as ctk
import webbrowser
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t

logger = logging.getLogger(__name__)


# Support page URLs — two ways to back the project
BMC_URL = "https://buymeacoffee.com/Kruk2"
UNIFANS_URL = "https://app.unifans.io/c/kruk2"


class Tooltip:
    """Simple tooltip implementation for CustomTkinter widgets."""

    _SHOW_DELAY_MS = 150

    def __init__(self, widget, text: str):
        self._widget = widget
        self._text = text
        self._tooltip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self.hide)

    def set_text(self, text: str):
        self._text = text

    def _schedule_show(self, event=None):
        self._cancel_schedule()
        self._after_id = self._widget.after(self._SHOW_DELAY_MS, self._show)

    def _cancel_schedule(self):
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._tooltip_window:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 5

        self._tooltip_window = tw = ctk.CTkToplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(fg_color=Colors.BG_CARD)
        tw.wm_attributes("-topmost", True)

        label = ctk.CTkLabel(
            tw,
            text=self._text,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
            fg_color=Colors.BG_CARD,
            corner_radius=4,
            wraplength=300,
            justify="left",
        )
        label.pack(padx=8, pady=6)
        tw.bind("<Leave>", self.hide)

    def hide(self, event=None):
        self._cancel_schedule()
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None


class _SupportButton(ctk.CTkButton):
    """Brand-styled button that opens a support page and scales 1.05x on hover."""

    def __init__(self, master, text: str, url: str, fg_color: str, text_color: str, width: int, **kwargs):
        super().__init__(
            master,
            text=text,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            fg_color=fg_color,
            hover_color=fg_color,  # Keep same, we scale on hover
            text_color=text_color,
            corner_radius=6,
            height=28,
            width=width,
            command=lambda: webbrowser.open(url),
            **kwargs
        )

        self._original_width = width
        self._original_height = 28

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, event=None):
        self.configure(
            width=int(self._original_width * 1.05),
            height=int(self._original_height * 1.05),
        )

    def _on_leave(self, event=None):
        self.configure(width=self._original_width, height=self._original_height)


def attach_entry_context_menu(entry: ctk.CTkEntry) -> tkinter.Menu:
    """Right-click menu with clipboard actions, for users who don't know Ctrl+V."""
    widget = entry._entry
    menu = tkinter.Menu(
        widget, tearoff=0,
        background=Colors.BG_PANEL, foreground=Colors.TEXT_PRIMARY,
        activebackground=Colors.PRIMARY, activeforeground=Colors.TEXT_PRIMARY,
    )
    menu.add_command(label=t("ctx_cut"), command=lambda: widget.event_generate("<<Cut>>"))
    menu.add_command(label=t("ctx_copy"), command=lambda: widget.event_generate("<<Copy>>"))
    menu.add_command(label=t("ctx_paste"), command=lambda: widget.event_generate("<<Paste>>"))
    menu.add_separator()
    menu.add_command(label=t("ctx_select_all"), command=lambda: widget.select_range(0, "end"))

    def _popup(event):
        widget.focus_set()
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    widget.bind("<Button-3>", _popup)
    return menu


class BuyMeCoffeeButton(_SupportButton):
    """Buy Me a Coffee button with brand styling and hover animation."""

    def __init__(self, master, compact: bool = False, **kwargs):
        if compact:
            text, width = "BMC", 50
        else:
            text, width = t("bmc_support"), 100
        super().__init__(
            master, text=text, url=BMC_URL,
            fg_color=Colors.BMC_YELLOW, text_color=Colors.BMC_TEXT, width=width, **kwargs,
        )


class UnifansButton(_SupportButton):
    """Unifans button — the other way to support the project, shown next to Buy Me a Coffee."""

    def __init__(self, master, compact: bool = False, **kwargs):
        if compact:
            text, width = "Unifans", 70
        else:
            text, width = t("unifans_support"), 110
        super().__init__(
            master, text=text, url=UNIFANS_URL,
            fg_color=Colors.UNIFANS_PURPLE, text_color=Colors.UNIFANS_TEXT, width=width, **kwargs,
        )


class LicenseDialog(ctk.CTkToplevel):
    """Modal popup to enter the supporter email + license key. Persisted by
    license_store (in the user config dir); on success calls on_activated so the
    header chip can refresh."""

    def __init__(self, master, on_activated):
        super().__init__(master)
        self._on_activated = on_activated

        self.title(t("supporter_title"))
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master)
        self.wait_visibility()  # X11: window must be viewable before grab_set, else TclError
        self.grab_set()
        self.lift()
        self.focus_force()

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=24, pady=24)

        ctk.CTkLabel(
            outer, text=t("supporter_title"),
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"), text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w", pady=(0, 6))
        ctk.CTkLabel(
            outer, text=t("supporter_blurb"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=340, justify="left",
        ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(
            outer, text=t("supporter_perks"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=340, justify="left",
        ).pack(anchor="w", pady=(0, 10))
        ctk.CTkLabel(
            outer, text=t("license_crypto_info"), text_color=Colors.STATUS_PENDING,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=340, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self._email = ctk.CTkEntry(
            outer, width=340, fg_color=Colors.BG_PANEL, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("license_email_placeholder"),
        )
        self._email.pack(fill="x", pady=(0, 6))
        attach_entry_context_menu(self._email)
        self._key = ctk.CTkEntry(
            outer, width=340, fg_color=Colors.BG_PANEL, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("license_key_placeholder"),
        )
        self._key.pack(fill="x", pady=(0, 10))
        attach_entry_context_menu(self._key)

        action = ctk.CTkFrame(outer, fg_color="transparent")
        action.pack(fill="x")
        ctk.CTkButton(
            action, text=t("license_activate"), width=110,
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER, text_color=Colors.TEXT_PRIMARY,
            command=self._activate,
        ).pack(side="left")
        self._status = ctk.CTkLabel(action, text="", text_color=Colors.TEXT_PRIMARY)
        self._status.pack(side="left", padx=10)

        from jasna.protection import license_store
        stored = license_store.load_license()
        if stored:
            self._email.insert(0, stored[0])
            self._key.insert(0, stored[1])
            if license_store.is_licensed():
                self._status.configure(text=t("license_active"), text_color=Colors.STATUS_COMPLETED)

        self.update_idletasks()
        w = max(388, self.winfo_reqwidth())
        h = self.winfo_reqheight()
        x = master.winfo_x() + (master.winfo_width() - w) // 2
        y = master.winfo_y() + (master.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _activate(self):
        from jasna.protection import ProtectionError, license_store
        email = self._email.get().strip()
        key = self._key.get().strip()
        try:
            license_store.set_license(email, key)
        except ProtectionError as exc:
            self._status.configure(text=str(exc), text_color=Colors.STATUS_ERROR)
            return
        self._status.configure(text=t("license_active"), text_color=Colors.STATUS_COMPLETED)
        self._on_activated()


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
            fg_color=Colors.STATUS_PENDING,
            corner_radius=4,
        )
        self._indicator.grid(row=0, column=0, padx=(12, 6), pady=8)
        
        self._label = ctk.CTkLabel(
            self,
            text=t("status_idle"),
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
            text_color=Colors.TEXT_PRIMARY,
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
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._label.pack(side="left", padx=(0, 8))
        
        if tooltip:
            self._tooltip_icon = ctk.CTkLabel(
                self,
                text="ⓘ",
                font=(Fonts.FAMILY, Fonts.SIZE_TINY),
                text_color=Colors.TEXT_PRIMARY,
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
        on_edit_segments: callable = None,
        **kwargs
    ):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
            height=72,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._on_remove = on_remove
        self._on_edit_segments = on_edit_segments
        # Drag callbacks (set by QueuePanel)
        self._on_drag_start = on_drag_start
        self._on_drag_move = on_drag_move
        self._on_drag_end = on_drag_end
        self._progress_visible = False
        self._conflict_visible = False
        self._segments_editable = True
        self._segment_tooltips: list[Tooltip] = []
        
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
            text_color=Colors.TEXT_PRIMARY,
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
            text_color=Colors.TEXT_PRIMARY,
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
            text_color=Colors.TEXT_PRIMARY,
        )
        self._status_label.pack(side="left")

        self._segment_summary = ctk.CTkLabel(
            bottom_row,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            cursor="hand2" if self._on_edit_segments else "arrow",
        )
        self._segment_summary.pack(side="left", padx=(8, 0))
        if self._on_edit_segments:
            self._segment_summary.bind("<Button-1>", lambda _event: self._handle_edit_segments())
        
        # FPS / ETA small labels on the right of bottom row
        self._stats_frame = ctk.CTkFrame(bottom_row, fg_color="transparent")
        self._stats_frame.pack(side="right")

        self._segments_btn = ctk.CTkButton(
            bottom_row,
            text="✂",
            width=30,
            height=22,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            cursor="hand2",
            command=self._handle_edit_segments,
        )
        if self._on_edit_segments:
            self._segments_btn.pack(side="right", padx=(0, 6))
            self._segment_tooltips = [
                Tooltip(self._segments_btn, t("segments_edit_tooltip")),
                Tooltip(self._segment_summary, t("segments_edit_tooltip")),
            ]

        self._fps_label = ctk.CTkLabel(
            self._stats_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._fps_label.pack(side="left", padx=(0, 8))

        self._eta_label = ctk.CTkLabel(
            self._stats_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
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
            text_color=Colors.TEXT_PRIMARY,
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
            self._segment_summary, self._segments_btn,
        ]
        for child in child_widgets:
            try:
                child.bind("<Enter>", self._on_enter)
                child.bind("<Leave>", self._on_leave)
            except Exception:
                logger.debug("Failed to bind hover events on child widget", exc_info=True)
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
            logger.debug("Failed to cancel pending hide on enter", exc_info=True)
        try:
            x, y = self.winfo_pointerxy()
            widget_x = self.winfo_rootx()
            widget_y = self.winfo_rooty()
            widget_w = self.winfo_width()
            widget_h = self.winfo_height()
            if widget_x <= x <= widget_x + widget_w and widget_y <= y <= widget_y + widget_h:
                self._remove_btn.place(relx=1.0, rely=0, anchor="ne", x=-4, y=4)
        except Exception:
            logger.debug("Pointer query failed on enter; showing remove button", exc_info=True)
            self._remove_btn.place(relx=1.0, rely=0, anchor="ne", x=-4, y=4)
        
    def _on_leave(self, event=None):
        # Debounced hide: schedule hide shortly to prevent flicker when moving
        # between child widgets
        try:
            if self._hide_after_id:
                self.after_cancel(self._hide_after_id)
        except Exception:
            logger.debug("Failed to cancel pending hide on leave", exc_info=True)

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
                logger.debug("Pointer query failed in delayed hide; forcing hide", exc_info=True)
                try:
                    self._remove_btn.place_forget()
                except Exception:
                    logger.debug("place_forget failed in delayed hide", exc_info=True)

        # Schedule a short delayed check
        try:
            self._hide_after_id = self.after(80, _hide_if_outside)
        except Exception:
            logger.debug("Could not schedule delayed hide; hiding now", exc_info=True)
            _hide_if_outside()
        
    def _handle_remove(self):
        if self._on_remove:
            self._on_remove()

    def _handle_edit_segments(self):
        if self._on_edit_segments and self._segments_editable:
            for tooltip in self._segment_tooltips:
                tooltip.hide()
            self._on_edit_segments()

    def set_segment_summary(self, text: str, *, selected: bool = False) -> None:
        self._segment_summary.configure(
            text=text,
            text_color=Colors.PRIMARY if selected else Colors.STATUS_PENDING,
        )

    def set_segments_editable(self, editable: bool) -> None:
        if self._on_edit_segments:
            self._segments_editable = bool(editable)
            if editable:
                self._segments_btn.configure(state="normal")
                if not self._segments_btn.winfo_manager():
                    self._segments_btn.pack(side="right", padx=(0, 6))
            else:
                for tooltip in self._segment_tooltips:
                    tooltip.hide()
                self._segments_btn.pack_forget()
            self._segment_summary.configure(cursor="hand2" if editable else "arrow")

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
            
    def set_status(self, status: str, icon: str = "", color: str = Colors.STATUS_PENDING):
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
                logger.debug("place_forget failed in set_removable", exc_info=True)

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
                logger.debug("Failed to set progress value %r", value, exc_info=True)

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

    def set_completed(self, elapsed_seconds: float):
        mins, secs = divmod(int(elapsed_seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            duration_str = f"{hours}h {mins}m"
        elif mins:
            duration_str = f"{mins}m {secs}s"
        else:
            duration_str = f"{secs}s"
        self._status_label.configure(text=f"{t('completed_in')} {duration_str}")
        self._fps_label.configure(text="")
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
            text_color=Colors.TEXT_PRIMARY,
            width=80,
            anchor="w",
        )
        self._time.pack(side="left")
        
        self._level = ctk.CTkLabel(
            self,
            text=level,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY, "bold"),
            text_color=level_colors.get(level, Colors.TEXT_PRIMARY),
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
        
        self.title(t("dialog_create_preset"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.resizable(False, False)
        self.transient(master)
        self.wait_visibility()  # X11: window must be viewable before grab_set, else TclError
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
            text=t("preset_name"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w")
        
        self._entry = ctk.CTkEntry(
            content,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            placeholder_text=t("preset_placeholder"),
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
            text=t("btn_cancel"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=90,
            height=36,
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        
        ctk.CTkButton(
            btn_frame,
            text=t("btn_create_preset"),
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
            self._error_label.configure(text=t("error_name_empty"))
            return
        if name.lower() in self._existing_names:
            self._error_label.configure(text=t("error_name_exists"))
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
        self.wait_visibility()  # X11: window must be viewable before grab_set, else TclError
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
            text=t("btn_cancel"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=80,
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        
        ctk.CTkButton(
            btn_frame,
            text=t("btn_delete_confirm"),
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
