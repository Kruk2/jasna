"""The main window's header: title, status, language, player, support and license chips, update chip."""

import logging
import webbrowser
from collections.abc import Callable

import customtkinter as ctk

from jasna import __version__, license_status, update_check
from jasna.gui import activation_worker
from jasna.gui.branding import HEADER_LOGO_SIZE, create_header_logo
from jasna.gui.components import (
    KofiButton,
    StatusPill,
    Tooltip,
    UnifansButton,
    XianyuButton,
    XianyuQrDialog,
    shows_xianyu_seller,
)
from jasna.gui.icons import create_icon
from jasna.gui.license_dialog import LicenseDialog
from jasna.gui.locales import LANGUAGE_NAMES, get_locale, t
from jasna.gui.queues import MainThreadCalls
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui.theme import Colors, Fonts, Sizing

logger = logging.getLogger(__name__)


class HeaderBar(ctk.CTkFrame):
    def __init__(
        self,
        master,
        main_thread: MainThreadCalls,
        *,
        on_options: Callable[[], None],
        on_open_player: Callable[[], None],
        on_system_check: Callable[[], None],
    ):
        super().__init__(master, fg_color=Colors.BG_PANEL, height=Sizing.HEADER_HEIGHT, corner_radius=0)
        self.pack_propagate(False)
        self._main_thread = main_thread

        # Left: Logo and title
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.pack(side="left", padx=Sizing.PADDING_MEDIUM)

        self._header_logo = create_header_logo()
        logo = ctk.CTkLabel(
            left,
            text="",
            image=self._header_logo,
            fg_color="transparent",
            width=HEADER_LOGO_SIZE[0],
            height=HEADER_LOGO_SIZE[1],
        )
        logo.pack(side="left")

        title = ctk.CTkLabel(
            left,
            text=t("app_title"),
            font=(Fonts.FAMILY, Fonts.SIZE_TITLE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        title.pack(side="left", padx=(8, 4))

        version = ctk.CTkLabel(
            left,
            text=f"v{__version__}",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        version.pack(side="left", pady=(4, 0))

        # Status pill, left-aligned next to the version (the right side holds the
        # license/support chips, which used to overlap a centered pill).
        self._status_pill = StatusPill(left)
        self._status_pill.pack(side="left", padx=(12, 0))

        # Right: Language, Help and About
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="right", padx=Sizing.PADDING_MEDIUM)

        self._options_icon = create_icon("settings", 20, Colors.TEXT_PRIMARY)
        self._options_btn = ctk.CTkButton(
            right,
            text="",
            image=self._options_icon,
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            width=32,
            height=32,
            command=on_options,
        )
        self._options_btn.pack(side="left", padx=(0, 8))
        Tooltip(self._options_btn, t("dialog_options_title"))

        locale = get_locale()
        self._lang_dropdown = ValueOptionMenu(
            right,
            options={code: LANGUAGE_NAMES[code] for code in locale.available_languages},
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=100,
            height=28,
            command=self._on_language_changed,
        )
        self._lang_dropdown.pack(side="left", padx=(0, 12))
        self._lang_dropdown.set_value(locale.current_language)

        self._video_player_icon = create_icon("play", 16, Colors.PLAYER_TEXT)
        self._video_player_btn = ctk.CTkButton(
            right,
            text=t("btn_video_player"),
            image=self._video_player_icon,
            compound="left",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            fg_color=Colors.PLAYER,
            hover_color=Colors.PLAYER_HOVER,
            border_color=Colors.PLAYER_BORDER,
            border_width=1,
            border_spacing=6,
            text_color=Colors.PLAYER_TEXT,
            text_color_disabled=Colors.STATUS_PENDING,
            corner_radius=8,
            width=0,
            height=34,
            command=on_open_player,
        )
        self._video_player_btn.pack(side="left", padx=(0, 12))

        # Support buttons — back the project on Ko-fi, Unifans or (Chinese UI) Xianyu
        KofiButton(right, width=100, height=28).pack(side="left", padx=(0, 8))
        show_xianyu = shows_xianyu_seller(get_locale().current_language)
        UnifansButton(right, width=110, height=28).pack(side="left", padx=(0, 8 if show_xianyu else 12))
        if show_xianyu:
            XianyuButton(right, width=110, height=28).pack(side="left", padx=(0, 12))

        self._license_chip = ctk.CTkButton(
            right, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent", hover_color=Colors.BG_CARD,
            text=t("license_chip_inactive"), text_color=Colors.STATUS_WARNING,
            width=0, command=self._open_license_dialog,
        )
        self._license_chip.pack(side="left", padx=(0, 12))
        self.refresh_license()
        if show_xianyu:
            activation_worker.start_unofficial_notice_check(
                lambda due, error: self._main_thread.post(lambda: self._unofficial_notice_ready(due, error))
            )

        self._system_check_btn = ctk.CTkButton(
            right,
            text=t("btn_system_check"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=0,
            command=on_system_check,
        )
        self._system_check_btn.pack(side="left", padx=(0, 4))

        self._help_btn = ctk.CTkButton(
            right,
            text=t("btn_help"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=0,
            command=self._show_help,
        )
        self._help_btn.pack(side="left")

        self._update_icon = create_icon("download", 14, Colors.STATUS_COMPLETED)
        self._update_chip = ctk.CTkButton(
            right,
            image=self._update_icon,
            compound="left",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.STATUS_COMPLETED,
            border_width=1,
            border_color=Colors.STATUS_COMPLETED,
            corner_radius=14,
            width=0,
            height=26,
        )
        Tooltip(self._update_chip, t("tip_update_available"))

    def set_status(self, status: str, color: str) -> None:
        self._status_pill.set_status(status, color)

    def set_player_enabled(self, enabled: bool) -> None:
        self._video_player_btn.configure(
            state="normal" if enabled else "disabled",
            fg_color=Colors.PLAYER if enabled else Colors.BG_CARD,
            border_color=Colors.PLAYER_BORDER if enabled else Colors.BORDER_LIGHT,
        )

    def show_update(self, version: str | None) -> None:
        """Show the chip for a newer ``version``; None hides it."""
        if version is None:
            self._update_chip.pack_forget()
            return
        self._update_chip.configure(
            text=t("update_available", version=version),
            command=lambda: webbrowser.open(update_check.release_page_url(version)),
        )
        self._update_chip.pack(side="left", padx=(8, 0), after=self._help_btn)

    def refresh_license(self) -> None:
        activation_worker.start_license_info(
            lambda info, error: self._main_thread.post(lambda: self._license_chip_ready(info, error))
        )

    def _license_chip_ready(self, info, error):
        if error is not None:
            logger.warning("Could not read activation status: %s", error)
        licensed = error is None and info[0]
        self._license_chip.configure(
            text=t(license_status.license_chip_key(info[1])) if licensed else t("license_chip_inactive"),
            text_color=Colors.STATUS_COMPLETED if licensed else Colors.STATUS_WARNING,
        )

    def _unofficial_notice_ready(self, due, error):
        if error is not None:
            logger.warning("Could not check the activation note: %s", error)
        elif due:
            XianyuQrDialog(self.winfo_toplevel(), unofficial_notice=True)

    def _open_license_dialog(self):
        LicenseDialog(self.winfo_toplevel(), on_activated=self.refresh_license)

    def _on_language_changed(self, code: str):
        locale = get_locale()
        if code == locale.current_language:
            return
        locale.set_language(code)
        self._refresh_ui_text()
        from tkinter import messagebox
        messagebox.showinfo(t("dialog_language_changed"), t("dialog_language_restart"))

    def _refresh_ui_text(self):
        self._help_btn.configure(text=t("btn_help"))
        self._video_player_btn.configure(text=t("btn_video_player"))
        self._status_pill.refresh_text()

    def _show_help(self):
        webbrowser.open("https://github.com/Kruk2/jasna")
