"""First-run wizard for dependency checking."""

import customtkinter as ctk
import os
import subprocess
import threading

from jasna import os_utils
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t
from jasna.gui.components import BuyMeCoffeeButton


_WINDOW_WIDTH = 820


class FirstRunWizard(ctk.CTkToplevel):
    """Modal wizard shown on first run to check dependencies."""
    
    def __init__(self, master, on_complete: callable = None, **kwargs):
        super().__init__(master, **kwargs)
        
        self._on_complete = on_complete
        self._checks_passed = True
        self._check_results = {}
        
        self.title(t("wizard_window_title"))
        self.resizable(True, False)
        self.configure(fg_color=Colors.BG_MAIN)

        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.lift()
        self.focus_force()
        
        # Build UI immediately with loading state
        self._build_ui_loading()

        # Let geometry settle, then size to content and center on parent
        self.update_idletasks()
        w = max(_WINDOW_WIDTH, self.winfo_reqwidth())
        h = self.winfo_reqheight()
        x = master.winfo_x() + (master.winfo_width() - w) // 2
        y = master.winfo_y() + (master.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        
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
            text_color=Colors.TEXT_SECONDARY,
        )
        self._subtitle.pack(pady=(8, 0))
        
        self._checks_frame = ctk.CTkFrame(
            self,
            fg_color=Colors.BG_PANEL,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._checks_frame.pack(fill="both", expand=True, padx=40, pady=20)
        
        self._check_labels = {}
        checks = [
            ("ffmpeg", t("wizard_check_ffmpeg")),
            ("ffprobe", t("wizard_check_ffprobe")),
            ("mkvmerge", t("wizard_check_mkvmerge")),
            ("gpu", t("wizard_check_gpu")),
            ("cuda", t("wizard_check_cuda")),
        ]
        if os.name == "nt":
            checks.append(("hags", t("wizard_check_hags")))
        
        for key, label in checks:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            
            status_label = ctk.CTkLabel(
                row,
                text="○",
                font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
                text_color=Colors.TEXT_MUTED,
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
                text_color=Colors.TEXT_MUTED,
                justify="right",
                anchor="e",
            )
            info_label.pack(side="right", fill="x", expand=True)
            
            self._check_labels[key] = (status_label, info_label)
            
        # Footer with disabled button during checking
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
        
        # Buy Me a Coffee button - secondary action
        self._bmc_btn = BuyMeCoffeeButton(btn_container, compact=False)
        self._bmc_btn.configure(height=48, width=140)
        self._bmc_btn._original_height = 48
        self._bmc_btn._original_width = 140
        self._bmc_btn.pack(side="left")
        
    def _start_checks_in_background(self) -> None:
        self._checks_thread = threading.Thread(target=self._run_checks_blocking, daemon=True)
        self._checks_thread.start()
        self.after(50, self._poll_checks_thread)

    def _poll_checks_thread(self) -> None:
        if not getattr(self, "_checks_thread", None):
            return
        if self._checks_thread.is_alive():
            self.after(100, self._poll_checks_thread)
            return
        self._apply_check_results_to_ui()

    def _apply_check_results_to_ui(self) -> None:
        if not self.winfo_exists():
            return

        subtitle_text = t("wizard_all_passed") if self._checks_passed else t("wizard_some_failed")
        subtitle_color = Colors.STATUS_COMPLETED if self._checks_passed else Colors.STATUS_PAUSED
        self._subtitle.configure(text=subtitle_text, text_color=subtitle_color)

        for key, (status_label, info_label) in self._check_labels.items():
            passed, info = self._check_results.get(key, (False, t("wizard_not_checked")))
            status_label.configure(
                text="✓" if passed else "✕",
                text_color=Colors.STATUS_COMPLETED if passed else Colors.STATUS_ERROR,
            )
            info_label.configure(text=info)

        btn_text = t("btn_get_started") if self._checks_passed else t("btn_continue_anyway")
        self._continue_btn.configure(text=btn_text, state="normal")
        
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=40, pady=(40, 20))
        
        title = ctk.CTkLabel(
            header,
            text=t("wizard_title"),
            font=(Fonts.FAMILY, 24, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        title.pack()
        
        subtitle_text = t("wizard_all_passed") if self._checks_passed else t("wizard_some_failed")
        subtitle_color = Colors.STATUS_COMPLETED if self._checks_passed else Colors.STATUS_PAUSED
        
        self._subtitle = ctk.CTkLabel(
            header,
            text=subtitle_text,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=subtitle_color,
        )
        self._subtitle.pack(pady=(8, 0))
        
        # Checks area - show results
        self._checks_frame = ctk.CTkFrame(self, fg_color=Colors.BG_PANEL, corner_radius=Sizing.BORDER_RADIUS)
        self._checks_frame.pack(fill="both", expand=True, padx=40, pady=20)
        
        checks = [
            ("ffmpeg", t("wizard_check_ffmpeg")),
            ("ffprobe", t("wizard_check_ffprobe")),
            ("mkvmerge", t("wizard_check_mkvmerge")),
            ("gpu", t("wizard_check_gpu")),
            ("cuda", t("wizard_check_cuda")),
        ]
        if os.name == "nt":
            checks.append(("hags", t("wizard_check_hags")))

        for key, label in checks:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            
            passed, info = self._check_results.get(key, (False, t("wizard_not_checked")))
            
            status_label = ctk.CTkLabel(
                row,
                text="✓" if passed else "✕",
                font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
                text_color=Colors.STATUS_COMPLETED if passed else Colors.STATUS_ERROR,
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
                text=info,
                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                text_color=Colors.TEXT_MUTED,
                justify="right",
                anchor="e",
            )
            info_label.pack(side="right", fill="x", expand=True)
            
        # Footer with OK button (enabled since checks are done)
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=40, pady=(20, 40))
        
        btn_text = t("btn_get_started") if self._checks_passed else t("btn_continue_anyway")
        
        self._continue_btn = ctk.CTkButton(
            footer,
            text=btn_text,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            height=48,
            width=200,
            command=self._on_continue,
        )
        self._continue_btn.pack()
        
    def _run_checks_blocking(self):
        """Run all dependency checks (blocking)."""
        self._check_results["ffmpeg"] = self._check_executable("ffmpeg")
        self._check_results["ffprobe"] = self._check_executable("ffprobe")
        self._check_results["mkvmerge"] = self._check_executable("mkvmerge")
        self._check_results["gpu"] = self._check_gpu()
        self._check_results["cuda"] = self._check_cuda()
        if os.name == "nt":
            self._check_results["hags"] = os_utils.check_windows_hardware_accelerated_gpu_scheduling()
        
        # Determine overall pass
        self._checks_passed = all(passed for passed, _ in self._check_results.values())
        
    def _check_executable(self, name: str) -> tuple[bool, str]:
        path = os_utils.find_executable(name)
        if not path:
            return False, t("wizard_not_found")
        if name in {"ffmpeg", "ffprobe"}:
            completed = subprocess.run(
                [path, "-version"],
                capture_output=True,
                text=True,
                check=False,
                startupinfo=os_utils.get_subprocess_startup_info(),
            )
            if completed.returncode != 0:
                return False, t("wizard_not_callable", path=path)
            try:
                major = os_utils._parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
            except ValueError:
                return False, t("wizard_found_no_major", path=path)
            if major != 8:
                return False, t("wizard_found_bad_major", path=path, major=major)
            return True, t("wizard_found_major", path=path, major=major)

        if name == "mkvmerge":
            completed = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                check=False,
                startupinfo=os_utils.get_subprocess_startup_info(),
            )
            if completed.returncode != 0:
                return False, t("wizard_not_callable", path=path)
            first = ((completed.stdout or "") + (completed.stderr or "")).splitlines()
            ver = first[0].strip() if first else "OK"
            return True, t("wizard_found_version", path=path, version=ver)

        return True, t("wizard_found", path=path)
        
    def _check_gpu(self) -> tuple[bool, str]:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return True, gpu_name
            return False, t("wizard_no_cuda")
        except Exception as e:
            return False, str(e)
            
    def _check_cuda(self) -> tuple[bool, str]:
        try:
            import torch
            if torch.cuda.is_available():
                version = torch.version.cuda
                return True, t("wizard_cuda_version", version=version)
            return False, t("wizard_not_available")
        except Exception as e:
            return False, str(e)
            
    def _on_continue(self):
        self.grab_release()
        self.destroy()
        if self._on_complete:
            self._on_complete(self._checks_passed)
