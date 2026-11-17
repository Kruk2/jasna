"""The license window: activate with the License code and show this PC's activation."""

import logging
import sys
import customtkinter as ctk
from jasna.gui import scaling
from jasna.gui import activation_worker
from jasna.gui.components import XianyuButton, _Dialog, attach_entry_context_menu, grab_modal, shows_xianyu_seller
from jasna.gui.queues import MainThreadCalls
from jasna.gui.theme import Colors, Fonts
from jasna.gui.locales import get_locale, t
from jasna.license_api import protection_available
from jasna import license_status

logger = logging.getLogger(__name__)


class LicenseDialog(_Dialog):
    """Activate with the License code."""

    def __init__(self, master, on_activated):
        super().__init__(master, t("supporter_title"))
        self._closed = False
        self._on_activated = on_activated
        self._main_thread = MainThreadCalls(self, 50)
        grab_modal(self)

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=24, pady=24)

        ctk.CTkLabel(
            outer, text=t("supporter_title"),
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"), text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w", pady=(0, 6))
        ctk.CTkLabel(
            outer, text=t("supporter_blurb"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=460, justify="left",
        ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(
            outer, text=t("supporter_perks"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=460, justify="left",
        ).pack(anchor="w", pady=(0, 10))
        ctk.CTkLabel(
            outer, text=t("license_official_sellers"), text_color=Colors.STATUS_PENDING,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=460, justify="left",
        ).pack(anchor="w", pady=(0, 10))
        if shows_xianyu_seller(get_locale().current_language):
            XianyuButton(outer, width=150, height=32).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(
            outer, text=t("license_code"), text_color=Colors.TEXT_PRIMARY,
        ).pack(anchor="w")
        self._code = ctk.CTkEntry(
            outer, width=460, fg_color=Colors.BG_PANEL, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("license_code"),
        )
        self._code.pack(fill="x", pady=(0, 10))
        attach_entry_context_menu(self._code)
        self._try_offline = ctk.BooleanVar(value=True)
        self._mode_buttons = [
            ctk.CTkRadioButton(outer, text=t(key), variable=self._try_offline, value=value, text_color=Colors.TEXT_PRIMARY)
            for key, value in (("license_mode_offline", True), ("license_mode_online", False))
        ]
        for button in self._mode_buttons:
            button.pack(anchor="w", pady=(0, 6))
        self._activate_button = ctk.CTkButton(
            outer, text=t("license_activate"),
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER,
            command=self._activate, state="disabled",
        )
        self._activate_button.pack(anchor="w")
        ctk.CTkLabel(
            outer, text=t("license_lease_hint"), text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=460, justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self._status = ctk.CTkLabel(
            outer, text="", text_color=Colors.TEXT_PRIMARY,
            wraplength=460, justify="left",
        )
        self._status.pack(anchor="w", pady=(6, 0))
        self._devices = ctk.CTkLabel(
            outer, text="", text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL), wraplength=460, justify="left",
        )
        if protection_available():
            activation_worker.start_license_info(
                lambda info, error: self._post_result(self._loaded, info, error)
            )
        else:
            self._set_busy(True, lock_inputs=True)
            self._show_error(t("license_source_unavailable"))

        self._resize()

    def _set_busy(self, busy, *, lock_inputs=False):
        self._activate_button.configure(state="disabled" if busy else "normal")
        for widget in (self._code, *self._mode_buttons):
            widget.configure(state="disabled" if busy and lock_inputs else "normal")

    def _resize(self):
        self.update_idletasks()
        minimum_width, _ = scaling.to_physical(self, 508, 0)
        scaling.place_centered_on_parent(
            self,
            self.master,
            max(minimum_width, self.winfo_reqwidth()),
            self.winfo_reqheight(),
        )

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        self._main_thread.close()
        super().destroy()

    def _post_result(self, callback, *args):
        if self._closed:
            logger.debug("Ignoring worker result for closed activation dialog")
            return
        self._main_thread.post(lambda: callback(*args))

    def _loaded(self, info, error):
        self._set_busy(False)
        if error is not None:
            self._show_error(str(error))
            return
        licensed, mode = info
        if licensed:
            self._status.configure(text=t(license_status.license_status_key(mode)), text_color=Colors.STATUS_COMPLETED)
            self._devices.configure(text=t("license_devices_checking"))
            self._devices.pack(anchor="w", pady=(6, 0))
            activation_worker.start_device_count(lambda count, error: self._post_result(self._devices_ready, count, error))
        self._resize()

    def _devices_ready(self, count, error):
        if error is not None:
            self._devices.configure(text=t("license_devices_unavailable"))
        else:
            self._devices.configure(text=license_status.device_use_text(count))
        self._resize()

    def _activate(self):
        self._set_busy(True, lock_inputs=True)
        windows_offline = sys.platform == "win32" and self._try_offline.get()
        self._status.configure(text=t("license_offline_permission" if windows_offline else "license_connecting"),
                               text_color=Colors.TEXT_PRIMARY)
        activation_worker.start_lease_activation(
            self._code.get().strip(),
            self._try_offline.get(),
            lambda result, error: self._post_result(self._lease_activation_done, result, error),
        )

    def _lease_activation_done(self, result, error):
        from jasna.protection.activation import ActivationServiceError
        from jasna.protection.lease import LeaseError, activated_message

        self._set_busy(False)
        if isinstance(error, ActivationServiceError):
            code = "license_lease_network_error" if error.code in ("network_error", "invalid_response") else f"license_service_{error.code}"
            self._show_error(t(code))
            return
        if isinstance(error, LeaseError):
            self._show_error(t(f"license_problem_{error.code}"))
            return
        if error is not None:
            self._show_error(str(error))
            return
        self._code.delete(0, "end")
        self._devices.pack_forget()
        self._status.configure(text=activated_message(result.online_reason), text_color=Colors.STATUS_COMPLETED)
        self._resize()
        self._on_activated()

    def _show_error(self, text: str) -> None:
        self._status.configure(text=text, text_color=Colors.STATUS_ERROR)
        self._resize()
