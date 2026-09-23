"""First-run wizard for dependency checking."""

import logging
import os
import threading
import time
import webbrowser

import customtkinter as ctk

from jasna.gui import scaling
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t
from jasna.gui.components import BuyMeCoffeeButton, UnifansButton, grab_modal
from jasna.gui.system_checks import WARNING_ONLY_CHECKS, evaluate_check_results, run_system_checks

logger = logging.getLogger(__name__)
_WINDOW_WIDTH = 820
_CHECKS_TIMEOUT_SECONDS = 30.0

_HELP_URLS = {
    "sysmem": "https://docs.cognex.com/deep-learning_420/web/EN/deep-learning/Content/Topics/optimization/gpu-disable-shared.htm?TocPath=Optimization%20Guidelines%7CNVIDIA%C2%AE%20GPU%20Guidelines%7C_____6",
}


class FirstRunWizard(ctk.CTkToplevel):
    """Modal wizard shown on first run to check dependencies."""
    
    def __init__(self, master, on_complete: callable = None, **kwargs):
        super().__init__(master, **kwargs)
        
        self._on_complete = on_complete
        # Pessimistic until the checks actually finish: a crash mid-run must not leave
        # the wizard claiming everything passed.
        self._checks_passed = False
        self._has_required_failure = True
        self._check_results = {}
        
        self.title(t("wizard_window_title"))
        self.resizable(True, False)
        self.configure(fg_color=Colors.BG_MAIN)

        self.transient(master)
        grab_modal(self)
        # The wizard is modal, so closing it must always be possible - otherwise a check
        # that never finishes leaves the whole app unusable.
        self.protocol("WM_DELETE_WINDOW", self._on_exit)
        self.bind("<Escape>", lambda _event: self._on_exit())
        self.lift()
        self.focus_force()

        # Build UI immediately with loading state
        self._build_ui_loading()

        # Let geometry settle, then size to content and center on parent
        self.update_idletasks()
        minimum_width, _ = scaling.to_physical(self, _WINDOW_WIDTH, 0)
        scaling.place_centered_on_parent(
            self,
            master,
            max(minimum_width, self.winfo_reqwidth()),
            self.winfo_reqheight(),
        )

        self.after(50, self._start_checks_in_background)
        
    def _build_ui_loading(self):
        """Build UI with loading/checking state shown immediately."""
        # Header
        self._header = ctk.CTkFrame(self, fg_color="transparent")
        self._header.pack(fill="x", padx=40, pady=(40, 20))
        
        title = ctk.CTkLabel(
            self._header,
            text=t("wizard_title"),
            font=(Fonts.FAMILY, 24, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        title.pack()
        
        self._subtitle = ctk.CTkLabel(
            self._header,
            text=t("wizard_subtitle"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._subtitle.pack(pady=(8, 0))

        # Footer before the checks frame: the packer starves its last slaves when the
        # window is shorter than the requested layout, and the only way out of this modal
        # dialog must never be the casualty.
        self._build_footer_loading()

        self._checks_frame = ctk.CTkFrame(
            self,
            fg_color=Colors.BG_PANEL,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._checks_frame.pack(fill="both", expand=True, padx=40, pady=20)

        self._check_labels = {}
        checks = [
            ("ascii_path", t("wizard_check_ascii_path")),
            ("ffprobe", t("wizard_check_ffprobe")),
            ("gpu", t("wizard_check_gpu")),
            ("cuda", t("wizard_check_cuda")),
            ("driver", t("wizard_check_driver")),
        ]
        if os.name == "nt":
            checks.append(("sysmem", t("wizard_check_sysmem")))
        
        for key, label in checks:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            
            status_label = ctk.CTkLabel(
                row,
                text="○",
                font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
                text_color=Colors.TEXT_PRIMARY,
                width=24,
            )
            status_label.pack(side="left")
            
            name_label = ctk.CTkLabel(
                row,
                text=label,
                font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
                text_color=Colors.TEXT_PRIMARY,
            )
            name_label.pack(side="left", padx=(8, 0))
            
            info_label = ctk.CTkLabel(
                row,
                text=t("wizard_checking"),
                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                text_color=Colors.TEXT_PRIMARY,
                justify="right",
                anchor="e",
            )
            info_label.pack(side="right", fill="x", expand=True)

            help_label = None
            if key in _HELP_URLS:
                help_label = ctk.CTkLabel(
                    row,
                    text=t(f"wizard_{key}_how_to_fix"),
                    font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "underline"),
                    text_color=Colors.PRIMARY,
                    cursor="hand2",
                )
                url = _HELP_URLS[key]
                help_label.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

            self._check_labels[key] = (status_label, info_label, help_label)

    def _build_footer_loading(self):
        """Footer with the disabled continue button shown while the checks run."""
        self._footer = ctk.CTkFrame(self, fg_color="transparent")
        self._footer.pack(fill="x", side="bottom", padx=40, pady=(20, 40))
        
        # Button container for centering both buttons
        btn_container = ctk.CTkFrame(self._footer, fg_color="transparent")
        btn_container.pack()
        
        self._continue_btn = ctk.CTkButton(
            btn_container,
            text=t("btn_get_started"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            height=48,
            width=200,
            command=self._on_continue,
            state="disabled",
        )
        self._continue_btn.pack(side="left", padx=(0, 12))
        
        # Support the project — Buy Me a Coffee or Unifans
        self._bmc_btn = BuyMeCoffeeButton(btn_container, width=140, height=48)
        self._bmc_btn.pack(side="left")

        self._unifans_btn = UnifansButton(btn_container, width=150, height=48)
        self._unifans_btn.pack(side="left", padx=(12, 0))
        
    def _start_checks_in_background(self) -> None:
        self._checks_deadline = time.monotonic() + _CHECKS_TIMEOUT_SECONDS
        self._checks_thread = threading.Thread(target=run_system_checks, args=(self._check_results,), daemon=True)
        self._checks_thread.start()
        self.after(50, self._poll_checks_thread)

    def _poll_checks_thread(self) -> None:
        if self._checks_thread.is_alive():
            if time.monotonic() < self._checks_deadline:
                self.after(100, self._poll_checks_thread)
                return
            # A wedged check (driver query, cold torch import) must not leave the modal
            # wizard stuck on "checking" forever. The thread is a daemon, so abandoning
            # it here cannot block process exit; the unfinished rows read as failures.
            logger.warning(
                "System check did not finish within %.0fs; showing partial results",
                _CHECKS_TIMEOUT_SECONDS,
            )
        self._apply_check_results_to_ui()

    def _apply_check_results_to_ui(self) -> None:
        if not self.winfo_exists():
            return

        self._checks_passed, self._has_required_failure = evaluate_check_results(
            self._check_results, self._check_labels.keys()
        )

        if self._checks_passed:
            subtitle_text = t("wizard_all_passed")
            subtitle_color = Colors.STATUS_COMPLETED
        elif self._has_required_failure:
            subtitle_text = t("wizard_required_failed")
            subtitle_color = Colors.STATUS_ERROR
        else:
            subtitle_text = t("wizard_warnings_only")
            subtitle_color = Colors.STATUS_WARNING
        self._subtitle.configure(text=subtitle_text, text_color=subtitle_color)

        for key, (status_label, info_label, help_label) in self._check_labels.items():
            passed, info = self._check_results.get(key, (False, t("wizard_not_checked")))
            if passed:
                icon, color = "✓", Colors.STATUS_COMPLETED
            elif key in WARNING_ONLY_CHECKS:
                icon, color = "⚠", Colors.STATUS_WARNING
            else:
                icon, color = "✕", Colors.STATUS_ERROR
            status_label.configure(text=icon, text_color=color)
            info_label.configure(text=info)
            if help_label is not None:
                if passed:
                    help_label.pack_forget()
                else:
                    help_label.pack(side="right", padx=(0, 8))

        if self._has_required_failure:
            self._continue_btn.configure(text=t("btn_exit"), state="normal", command=self._on_exit)
        else:
            self._continue_btn.configure(text=t("btn_get_started"), state="normal")
        
    def _on_exit(self):
        self.grab_release()
        self.destroy()
        if self._on_complete:
            self._on_complete(False, False)

    def _on_continue(self):
        self.grab_release()
        self.destroy()
        if self._on_complete:
            self._on_complete(True, self._checks_passed)
