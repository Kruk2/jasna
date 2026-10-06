"""Reusable UI components for Jasna GUI."""

import tkinter
import customtkinter as ctk
import webbrowser
from jasna.gui import scaling
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t


# Support page URLs — two ways to back the project
BMC_URL = "https://buymeacoffee.com/Kruk2"
UNIFANS_URL = "https://app.unifans.io/c/kruk2"


_MODAL_BINDTAG = "JasnaModal"


def grab_modal(dialog) -> None:
    """Make a toplevel modal and focused; X11 refuses a grab until the window is viewable.

    A modal hidden by Win+D would otherwise keep its grab invisibly, so it reappears on focus.
    """
    if not dialog.winfo_viewable():
        dialog.wait_visibility()
    dialog.grab_set()
    dialog.lift()
    dialog.focus_force()
    bindtags = dialog.bindtags()
    if _MODAL_BINDTAG not in bindtags:
        dialog.bindtags((_MODAL_BINDTAG, *bindtags))
    if not dialog.bind_class(_MODAL_BINDTAG, "<FocusIn>"):
        dialog.bind_class(_MODAL_BINDTAG, "<FocusIn>", _restore_hidden_modal)


def _restore_hidden_modal(event) -> None:
    if not event.widget.winfo_ismapped():
        event.widget.deiconify()


def format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


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

    def __init__(self, master, text: str, url: str, fg_color: str, text_color: str, width: int, height: int):
        super().__init__(
            master,
            text=text,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            fg_color=fg_color,
            hover_color=fg_color,
            text_color=text_color,
            corner_radius=6,
            height=height,
            width=width,
            command=lambda: webbrowser.open(url),
        )

        self._original_width = width
        self._original_height = height

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
    def __init__(self, master, width: int, height: int):
        super().__init__(
            master, text=t("bmc_support"), url=BMC_URL,
            fg_color=Colors.BMC_YELLOW, text_color=Colors.BMC_TEXT, width=width, height=height,
        )


class UnifansButton(_SupportButton):
    def __init__(self, master, width: int, height: int):
        super().__init__(
            master, text=t("unifans_support"), url=UNIFANS_URL,
            fg_color=Colors.UNIFANS_PURPLE, text_color=Colors.UNIFANS_TEXT, width=width, height=height,
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
        grab_modal(self)

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
        ).pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            outer, text=t("license_official_sellers"), text_color=Colors.STATUS_PENDING,
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

        from jasna.license_api import license_store
        stored = license_store.load_license()
        if stored:
            self._email.insert(0, stored[0])
            self._key.insert(0, stored[1])
            if license_store.is_licensed():
                self._status.configure(text=t("license_active"), text_color=Colors.STATUS_COMPLETED)

        self.update_idletasks()
        minimum_width, _ = scaling.to_physical(self, 388, 0)
        scaling.place_centered_on_parent(
            self,
            master,
            max(minimum_width, self.winfo_reqwidth()),
            self.winfo_reqheight(),
        )

    def _activate(self):
        from jasna.license_api import (
            ForgedLicenseError, LicenseError, MalformedLicenseError,
            ProtectionError, RetiredLicenseError, license_store,
        )
        email = self._email.get().strip()
        key = self._key.get().strip()
        try:
            license_store.set_license(email, key)
        except ForgedLicenseError:
            self._show_error(t("license_forged"))
            return
        except RetiredLicenseError:
            self._show_error(t("license_retired"))
            return
        except MalformedLicenseError:
            self._show_error(t("license_malformed"))
            return
        except LicenseError:
            self._show_error(t("license_invalid"))
            return
        except ProtectionError as exc:
            self._show_error(str(exc))
            return
        self._status.configure(text=t("license_active"), text_color=Colors.STATUS_COMPLETED)
        self._on_activated()

    def _show_error(self, text: str) -> None:
        self._status.configure(text=text, text_color=Colors.STATUS_ERROR, wraplength=220, justify="left")


class StatusPill(ctk.CTkFrame):
    """Status indicator pill shown in header."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=16,
            width=1,
            height=28,
            **kwargs
        )
        self.grid_propagate(True)
        self._status_key = "idle"
        
        self._indicator = ctk.CTkLabel(
            self,
            text="",
            width=8,
            height=8,
            fg_color=Colors.STATUS_PENDING,
            corner_radius=4,
        )
        self._indicator.grid(row=0, column=0, padx=(10, 5), pady=8)
        
        self._label = ctk.CTkLabel(
            self,
            text=t(f"status_{self._status_key}"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            height=20,
        )
        self._label.grid(row=0, column=1, padx=(0, 10), pady=4)
        
    def set_status(self, status: str, color: str):
        self._status_key = status.lower()
        self.refresh_text()
        self._indicator.configure(fg_color=color)

    def refresh_text(self) -> None:
        self._label.configure(text=t(f"status_{self._status_key}").upper())


class AutoHidingScrollableFrame(ctk.CTkScrollableFrame):
    """Scrollable frame whose scrollbar is gridded only while the content overflows."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._scrollbar_visible = True
        # The scrollbar's default 200px height request joins the panel's size
        # negotiation, so under height shortage gridding it in/out re-splits the
        # space and flips the overflow state back — an endless <Configure> storm
        # that froze the GUI at 200% display scaling (#253). A token request
        # keeps show/hide geometry-neutral vertically; sticky="ns" sizes it.
        self._scrollbar.configure(height=8)
        self._parent_canvas.configure(yscrollcommand=self._update_scrollbar)
        self.after_idle(lambda: self._update_scrollbar(*self._parent_canvas.yview()))

    def _update_scrollbar(self, first: str | float, last: str | float) -> None:
        self._scrollbar.set(first, last)
        should_show = float(first) > 0.0 or float(last) < 1.0
        if should_show == self._scrollbar_visible:
            return
        if should_show:
            self._scrollbar.grid()
        else:
            self._scrollbar.grid_remove()
        self._scrollbar_visible = should_show


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
            "warning": Colors.STATUS_WARNING,
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
        grab_modal(self)
        
        self._on_create = on_create
        self._existing_names = [n.lower() for n in existing_names]
        self._result = None
        
        self.update_idletasks()
        scaling.place_centered_on_parent(self, master, *scaling.to_physical(self, 320, 180))
        
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
        grab_modal(self)
        
        self._on_confirm = on_confirm
        
        self.update_idletasks()
        scaling.place_centered_on_parent(self, master, *scaling.to_physical(self, 320, 140))
        
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


class ShutdownCountdownDialog(ctk.CTkToplevel):
    """Counts down before shutting the PC down; Cancel or closing the window aborts it."""

    def __init__(self, master, seconds: int, on_expired, on_cancelled):
        super().__init__(master)
        self._remaining = seconds
        self._on_expired = on_expired
        self._on_cancelled = on_cancelled

        self.title(t("shutdown_countdown_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        self._message = ctk.CTkLabel(
            content,
            text=t("shutdown_countdown_message", seconds=seconds),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            wraplength=280,
        )
        self._message.pack(pady=(0, 16))
        self.cancel_button = ctk.CTkButton(
            content,
            text=t("btn_cancel"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=self.cancel,
        )
        self.cancel_button.pack()

        self.update_idletasks()
        minimum_width, _ = scaling.to_physical(self, 320, 0)
        scaling.place_centered_on_parent(
            self, master, max(minimum_width, self.winfo_reqwidth()), self.winfo_reqheight()
        )
        grab_modal(self)
        self._tick_id = self.after(1000, self._tick)

    def _tick(self):
        self._remaining -= 1
        if self._remaining > 0:
            self._message.configure(text=t("shutdown_countdown_message", seconds=self._remaining))
            self._tick_id = self.after(1000, self._tick)
            return
        self.destroy()
        self._on_expired()

    def cancel(self):
        self.after_cancel(self._tick_id)
        self.destroy()
        self._on_cancelled()
