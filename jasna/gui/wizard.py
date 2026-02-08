"""First-run wizard for dependency checking."""

import customtkinter as ctk
import os
import subprocess

from jasna import os_utils
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t
from jasna.gui.components import BuyMeCoffeeButton


_WINDOW_WIDTH = 820
_WINDOW_HEIGHT = 520
_INFO_COLUMN_WIDTH = 520


class FirstRunWizard(ctk.CTkToplevel):
    """Modal wizard shown on first run to check dependencies."""
    
    def __init__(self, master, on_complete: callable = None, **kwargs):
        super().__init__(master, **kwargs)
        
        self._on_complete = on_complete
        self._checks_passed = True
        self._check_results = {}
        
        self.title("Jasna - System Check")
        self.geometry(f"{_WINDOW_WIDTH}x{_WINDOW_HEIGHT}")
        self.resizable(True, False)
        self.configure(fg_color=Colors.BG_MAIN)

        # Avoid using overrideredirect() which can cause grab/focus issues
        # on some window managers when the user clicks away. Instead keep
        # the normal title bar but disable the close action so the X button
        # is inert (prevents accidental close).
        # Make modal
        self.transient(master)
        self.grab_set()
        # Disable the close window action (WM_DELETE_WINDOW) so user can't
        # dismiss the wizard via the X button; they must click the provided button.
        try:
            self.protocol("WM_DELETE_WINDOW", lambda: None)
        except Exception:
            pass
        # Ensure dialog is on top and has focus
        try:
            self.lift()
            self.focus_force()
        except Exception:
            pass
        
        # Center on parent
        self.update_idletasks()
        x = master.winfo_x() + (master.winfo_width() - _WINDOW_WIDTH) // 2
        y = master.winfo_y() + (master.winfo_height() - _WINDOW_HEIGHT) // 2
        self.geometry(f"+{x}+{y}")
        
        # Build UI immediately with loading state
        self._build_ui_loading()
        
        # Run checks AFTER UI is visible (defer to next event loop)
        self.after(50, self._run_checks_and_update)
        
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
            text=t("wizard_checking") if hasattr(t, '__call__') and t("wizard_checking") != "wizard_checking" else "Checking system requirements...",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_SECONDARY,
        )
        self._subtitle.pack(pady=(8, 0))
        
        # Checks area - show loading state
        self._checks_frame = ctk.CTkFrame(self, fg_color=Colors.BG_PANEL, corner_radius=Sizing.BORDER_RADIUS)
        self._checks_frame.pack(fill="both", expand=True, padx=40, pady=20)
        
        self._check_labels = {}
        checks = [
            ("ffmpeg", "FFmpeg"),
            ("ffprobe", "FFprobe"),
            ("mkvmerge", "MKVmerge"),
            ("gpu", "NVIDIA GPU"),
            ("cuda", "CUDA Runtime"),
        ]
        if os.name == "nt":
            checks.append(("hags", "HW GPU Scheduling"))
        
        for key, label in checks:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            
            status_label = ctk.CTkLabel(
                row,
                text="○",  # Loading indicator
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
                text="Checking...",
                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                text_color=Colors.TEXT_MUTED,
                width=_INFO_COLUMN_WIDTH,
                wraplength=_INFO_COLUMN_WIDTH,
                justify="right",
                anchor="e",
            )
            info_label.pack(side="right")
            
            self._check_labels[key] = (status_label, info_label)
            
        # Footer with disabled button during checking
        self._footer = ctk.CTkFrame(self, fg_color="transparent")
        self._footer.pack(fill="x", padx=40, pady=(20, 40))
        
        # Button container for centering both buttons
        btn_container = ctk.CTkFrame(self._footer, fg_color="transparent")
        btn_container.pack()
        
        self._continue_btn = ctk.CTkButton(
            btn_container,
            text=t("btn_get_started") if hasattr(t, '__call__') else "Get Started",
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
        
    def _run_checks_and_update(self):
        """Run checks and update UI with results."""
        self._run_checks_blocking()
        
        # Update subtitle
        subtitle_text = t("wizard_all_passed") if self._checks_passed else t("wizard_some_failed")
        subtitle_color = Colors.STATUS_COMPLETED if self._checks_passed else Colors.STATUS_PAUSED
        self._subtitle.configure(text=subtitle_text, text_color=subtitle_color)
        
        # Update each check result
        for key, (status_label, info_label) in self._check_labels.items():
            passed, info = self._check_results.get(key, (False, "Not checked"))
            status_label.configure(
                text="✓" if passed else "✕",
                text_color=Colors.STATUS_COMPLETED if passed else Colors.STATUS_ERROR,
            )
            info_label.configure(text=info)
            
        # Enable continue button
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
            ("ffmpeg", "FFmpeg"),
            ("ffprobe", "FFprobe"),
            ("mkvmerge", "MKVmerge"),
            ("gpu", "NVIDIA GPU"),
            ("cuda", "CUDA Runtime"),
        ]
        if os.name == "nt":
            checks.append(("hags", "HW GPU Scheduling"))
        
        for key, label in checks:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            
            passed, info = self._check_results.get(key, (False, "Not checked"))
            
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
                width=_INFO_COLUMN_WIDTH,
                wraplength=_INFO_COLUMN_WIDTH,
                justify="right",
                anchor="e",
            )
            info_label.pack(side="right")
            
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
            return False, "Not found"
        env = os_utils.get_subprocess_env_for_executable(path)

        if name in {"ffmpeg", "ffprobe"}:
            completed = subprocess.run(
                [path, "-version"],
                capture_output=True,
                text=True,
                check=False,
                startupinfo=os_utils.get_subprocess_startup_info(),
                env=env,
            )
            if completed.returncode != 0:
                return False, f"Not callable: {path}"
            try:
                major = os_utils._parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
            except ValueError:
                return False, f"Found: {path} (could not detect major version)"
            if major != 8:
                return False, f"Found: {path} (major={major}, expected=8)"
            return True, f"Found: {path} (major={major})"

        if name == "mkvmerge":
            completed = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                check=False,
                startupinfo=os_utils.get_subprocess_startup_info(),
                env=env,
            )
            if completed.returncode != 0:
                return False, f"Not callable: {path}"
            first = ((completed.stdout or "") + (completed.stderr or "")).splitlines()
            ver = first[0].strip() if first else "OK"
            return True, f"Found: {path} ({ver})"

        return True, f"Found: {path}"
        
    def _check_gpu(self) -> tuple[bool, str]:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return True, gpu_name
            return False, "No CUDA device"
        except Exception as e:
            return False, str(e)
            
    def _check_cuda(self) -> tuple[bool, str]:
        try:
            import torch
            if torch.cuda.is_available():
                version = torch.version.cuda
                return True, f"CUDA {version}"
            return False, "Not available"
        except Exception as e:
            return False, str(e)
            
    def _on_continue(self):
        self.grab_release()
        self.destroy()
        if self._on_complete:
            self._on_complete(self._checks_passed)
