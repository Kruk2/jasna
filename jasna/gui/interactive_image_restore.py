from __future__ import annotations

import logging
import queue
import threading
from contextlib import nullcontext
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from jasna.gui.locales import t
from jasna.gui.models import AppSettings
from jasna.gui.theme import Colors, Fonts, Sizing

logger = logging.getLogger(__name__)


def interactive_output_path(input_path: Path, output_folder: str, output_pattern: str) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_folder) if output_folder else input_path.parent
    pattern = output_pattern or "{original}_restored.mp4"
    output_name = pattern.replace("{original}", input_path.stem)
    candidate = (output_dir / output_name).with_suffix(input_path.suffix)
    return _unique_path(candidate)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for counter in range(1, 10000):
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique filename after 9999 attempts: {path}")


def _dialog_geometry_for_image(image_size: tuple[int, int], screen_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    screen_w, screen_h = screen_size
    max_window_w = max(640, screen_w - 80)
    max_window_h = max(520, screen_h - 80)
    horizontal_chrome = 36
    vertical_chrome = 170
    max_preview_w = max(360, max_window_w - horizontal_chrome)
    max_preview_h = max(260, max_window_h - vertical_chrome)

    scale = min(max_preview_w / image_w, max_preview_h / image_h, 1.0)
    preview_w = max(360, int(image_w * scale))
    preview_h = max(260, int(image_h * scale))
    window_w = min(max_window_w, preview_w + horizontal_chrome)
    window_h = min(max_window_h, preview_h + vertical_chrome)
    return window_w, window_h, preview_w, preview_h


class InteractiveImageRestoreDialog(ctk.CTkToplevel):
    def __init__(
        self,
        master,
        image_paths: list[Path],
        settings: AppSettings,
        output_folder: str,
        output_pattern: str,
        on_log: callable | None = None,
    ):
        super().__init__(master)
        self._paths = [Path(p) for p in image_paths]
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        self._on_log = on_log

        self._index = 0
        self._seed = int(settings.image_restore_seed)
        self._view = t("interactive_view_restored")
        self._token = 0
        self._closed = False
        self._current_result = None
        self._current_raw = None
        self._current_mask = None
        self._photo = None

        image_size = _read_image_size(self._paths[0])
        screen_size = (self.winfo_screenwidth(), self.winfo_screenheight())
        self._window_w, self._window_h, self._preview_w, self._preview_h = _dialog_geometry_for_image(image_size, screen_size)

        self._requests: queue.Queue[tuple[int, int, int] | None] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        self.title(t("interactive_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()
        self._center(master)
        self.wait_visibility()
        self.grab_set()
        self.lift()
        self.focus_force()

        self._worker.start()
        self._request_render()

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        top = ctk.CTkFrame(outer, fg_color="transparent")
        top.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        self._title = ctk.CTkLabel(
            top,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TITLE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._title.pack(side="left", fill="x", expand=True)

        self._counter = ctk.CTkLabel(
            top,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._counter.pack(side="right")

        self._image_label = ctk.CTkLabel(
            outer,
            text=t("interactive_loading"),
            fg_color=Colors.BG_PANEL,
            text_color=Colors.TEXT_PRIMARY,
            width=self._preview_w,
            height=self._preview_h,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._image_label.pack(fill="both", expand=True)

        self._status = ctk.CTkLabel(
            outer,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        self._status.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

        controls = ctk.CTkFrame(outer, fg_color="transparent")
        controls.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))

        self._prev_image = ctk.CTkButton(
            controls,
            text=t("interactive_prev_image"),
            width=120,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=self._previous_image,
        )
        self._prev_image.pack(side="left")

        self._next_image = ctk.CTkButton(
            controls,
            text=t("interactive_next_image"),
            width=120,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=self._next_image_clicked,
        )
        self._next_image.pack(side="left", padx=(6, 12))

        self._prev_seed = ctk.CTkButton(
            controls,
            text=t("interactive_prev_seed"),
            width=96,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=lambda: self._change_seed(-1),
        )
        self._prev_seed.pack(side="left")

        self._seed_label = ctk.CTkLabel(
            controls,
            text="",
            width=120,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._seed_label.pack(side="left", padx=6)

        self._next_seed = ctk.CTkButton(
            controls,
            text=t("interactive_next_seed"),
            width=96,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=lambda: self._change_seed(1),
        )
        self._next_seed.pack(side="left", padx=(0, 12))

        self._view_selector = ctk.CTkSegmentedButton(
            controls,
            values=[t("interactive_view_restored"), t("interactive_view_mask"), t("interactive_view_raw")],
            command=self._set_view,
            selected_color=Colors.PRIMARY,
            selected_hover_color=Colors.PRIMARY_HOVER,
            unselected_color=Colors.BG_CARD,
            unselected_hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
        )
        self._view_selector.pack(side="left")
        self._view_selector.set(self._view)

        self._save = ctk.CTkButton(
            controls,
            text=t("interactive_save"),
            width=96,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
            command=self._save_current,
            state="disabled",
        )
        self._save.pack(side="right")

        self._close_btn = ctk.CTkButton(
            controls,
            text=t("btn_close"),
            width=80,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=self._close,
        )
        self._close_btn.pack(side="right", padx=(0, 6))

        self._refresh_header()

    def _center(self, master) -> None:
        self.geometry(f"{self._window_w}x{self._window_h}")
        self.update_idletasks()
        x = master.winfo_rootx() + (master.winfo_width() - self._window_w) // 2
        y = master.winfo_rooty() + (master.winfo_height() - self._window_h) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _refresh_header(self) -> None:
        path = self._paths[self._index]
        self._title.configure(text=path.name)
        self._counter.configure(text=t("interactive_counter", current=self._index + 1, total=len(self._paths)))
        self._seed_label.configure(text=t("interactive_seed_value", seed=self._seed))
        self._prev_image.configure(state="normal" if self._index > 0 else "disabled")
        self._next_image.configure(state="normal" if self._index < len(self._paths) - 1 else "disabled")

    def _change_seed(self, delta: int) -> None:
        self._seed += int(delta)
        self._request_render()

    def _previous_image(self) -> None:
        if self._index <= 0:
            return
        self._index -= 1
        self._request_render()

    def _next_image_clicked(self) -> None:
        if self._index >= len(self._paths) - 1:
            return
        self._index += 1
        self._request_render()

    def _set_view(self, value: str) -> None:
        self._view = value
        self._show_current_view()

    def _request_render(self) -> None:
        self._token += 1
        token = self._token
        self._current_result = None
        self._current_raw = None
        self._current_mask = None
        self._save.configure(state="disabled")
        self._status.configure(text=t("interactive_rendering"), text_color=Colors.STATUS_PROCESSING)
        self._image_label.configure(text=t("interactive_loading"), image=None)
        self._refresh_header()
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put((token, self._index, self._seed))

    def _worker_loop(self) -> None:
        detector = None
        restorer = None
        prepared_by_path = {}
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return

                token, index, seed = request
                try:
                    detector, restorer, device = self._ensure_session(detector, restorer)
                    path = self._paths[index]
                    prepared = prepared_by_path.get(path)
                    if prepared is None:
                        prepared = self._prepare(path, detector, device)
                        prepared_by_path[path] = prepared
                    result = self._render(prepared, restorer, seed)
                    mask = self._mask(prepared)
                    self._after(lambda: self._finish_render(token, index, seed, prepared.img_chw_u8, mask, result))
                except Exception as exc:
                    self._after(lambda e=exc, tk=token: self._fail_render(tk, e))
        finally:
            if detector is not None:
                detector.close()
            if restorer is not None:
                restorer.close()
            try:
                import torch
                from jasna.gui.processor import _cleanup_torch
                _cleanup_torch(torch)
            except Exception:
                logger.debug("Torch cleanup failed during render worker teardown", exc_info=True)

    def _ensure_session(self, detector, restorer):
        if detector is not None and restorer is not None:
            return detector, restorer, self._device

        import torch

        from jasna._suppress_noise import install as _install_noise_filters
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.engine_paths import SD15_DIR
        from jasna.mosaic.detection_registry import build_detection_model, coerce_detection_model_name, require_detection_model_weights
        from jasna.restorer.sd15_download import bundle_present
        from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer

        _install_noise_filters()
        if not bundle_present(SD15_DIR):
            raise FileNotFoundError(t("interactive_model_missing"))

        settings = self._settings
        self._device = torch.device("cuda:0")
        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = require_detection_model_weights(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(self._device),
                fp16=settings.fp16_mode,
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
            ),
        )
        detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=self._device,
            score_threshold=settings.detection_score_threshold,
            fp16=settings.fp16_mode,
        )
        restorer = Sd15InpaintRestorer(SD15_DIR, self._device, settings.fp16_mode)
        return detector, restorer, self._device

    def _prepare(self, path: Path, detector, device):
        from jasna.image_restore import prepare_image_restore
        from jasna.media import image_io

        img = image_io.read_image_rgb_chw(path)
        return prepare_image_restore(img, detector, device=device, fp16=self._settings.fp16_mode)

    def _render(self, prepared, restorer, seed: int):
        from jasna.image_restore import clamp_strength, render_prepared_image
        from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

        freeu = dict(DEFAULT_FREEU) if bool(self._settings.image_restore_freeu) else None
        strength = clamp_strength(float(self._settings.image_restore_strength))
        with __import__("torch").cuda.device(self._device) if self._device.type == "cuda" else nullcontext():
            return render_prepared_image(
                prepared,
                restorer,
                steps=int(self._settings.image_restore_steps),
                strength=strength,
                seed=int(seed),
                freeu=freeu,
            )

    def _mask(self, prepared):
        from jasna.image_restore import mask_overlay_rgb_chw

        return mask_overlay_rgb_chw(prepared)

    def _finish_render(self, token: int, index: int, seed: int, raw, mask, result) -> None:
        if self._closed or token != self._token:
            return
        self._index = index
        self._seed = seed
        self._current_raw = raw
        self._current_mask = mask
        self._current_result = result
        self._status.configure(text=t("interactive_ready"), text_color=Colors.STATUS_COMPLETED)
        self._save.configure(state="normal")
        self._refresh_header()
        self._show_current_view()

    def _fail_render(self, token: int, exc: Exception) -> None:
        if self._closed or token != self._token:
            return
        message = str(exc)
        self._status.configure(text=message, text_color=Colors.STATUS_ERROR)
        self._save.configure(state="disabled")
        if self._on_log:
            self._on_log("ERROR", message)

    def _show_current_view(self) -> None:
        image = self._current_result
        if self._view == t("interactive_view_mask"):
            image = self._current_mask
        elif self._view == t("interactive_view_raw"):
            image = self._current_raw

        if image is None:
            return
        photo = _ctk_image_from_chw(image, self._preview_w, self._preview_h)
        old_photo = self._photo
        self._image_label.configure(text="", image=photo)
        self._photo = photo
        _ = old_photo

    def _save_current(self) -> None:
        if self._current_result is None:
            return
        from jasna.media import image_io

        path = interactive_output_path(self._paths[self._index], self._output_folder, self._output_pattern)
        path.parent.mkdir(parents=True, exist_ok=True)
        image_io.write_image_rgb_chw(path, self._current_result)
        self._status.configure(text=t("interactive_saved", path=str(path)), text_color=Colors.STATUS_COMPLETED)
        if self._on_log:
            self._on_log("INFO", f"Wrote {path}")

    def _after(self, callback: callable) -> None:
        if self._closed:
            return
        try:
            self.after(0, callback)
        except Exception:
            logger.debug("Failed to schedule callback (widget gone)", exc_info=True)

    def _close(self) -> None:
        self._closed = True
        self._requests.put(None)
        self.destroy()


def _ctk_image_from_chw(chw_uint8, max_w: int, max_h: int):
    h = int(chw_uint8.shape[1])
    w = int(chw_uint8.shape[2])
    scale = min(max_w / w, max_h / h, 1.0)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    rgb = chw_uint8.transpose(1, 2, 0)
    image = Image.fromarray(rgb, mode="RGB")
    return ctk.CTkImage(light_image=image, dark_image=image, size=size)


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
