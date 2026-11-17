from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from jasna.gui.components import grab_modal
from jasna.gui import scaling
from jasna.gui.locales import t
from jasna.gui.models import DEFAULT_OUTPUT_PATTERN, AppSettings
from jasna.gui.image_restore_worker import ImageRestoreWorker
from jasna.gui.queues import MainThreadCalls
from jasna.media.media_files import folder_output_path, unique_path
from jasna.gui.theme import Colors, Fonts, Sizing


def interactive_output_path(input_path: Path, output_folder: str, output_pattern: str) -> Path:
    input_path = Path(input_path)
    return unique_path(folder_output_path(output_folder or input_path.parent, input_path, output_pattern or DEFAULT_OUTPUT_PATTERN))


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
        on_log: Callable[[str, str], None],
        on_stopped: Callable[[], None],
    ):
        super().__init__(master)
        self._main_thread = MainThreadCalls(self, 50)
        self._paths = [Path(p) for p in image_paths]
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        self._on_log = on_log

        self._index = 0
        self._seed = int(settings.image_restore_seed)
        self._view = "restored"
        self._view_by_label = {
            t("interactive_view_restored"): "restored",
            t("interactive_view_mask"): "mask",
            t("interactive_view_raw"): "raw",
        }
        self._token = 0
        self._closed = False
        self._current_result = None
        self._current_raw = None
        self._current_mask = None
        self._photo = None

        image_size = _read_image_size(self._paths[0])
        # Logical screen: the geometry it returns feeds both geometry() and CTk widget
        # sizes, which are multiplied by the scaling factor again at render time.
        logical_screen = scaling.to_logical(self, *scaling.screen_rect(self)[2:])
        self._window_w, self._window_h, self._preview_w, self._preview_h = _dialog_geometry_for_image(image_size, logical_screen)

        self._worker = ImageRestoreWorker(
            self._paths,
            settings,
            on_rendered=lambda *rendered: self._main_thread.post(lambda: self._finish_render(*rendered)),
            on_failed=lambda token, exc: self._main_thread.post(lambda: self._fail_render(token, exc)),
            on_stopped=on_stopped,
        )

        self.title(t("interactive_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()
        self._center(master)
        grab_modal(self)

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
            values=list(self._view_by_label),
            command=lambda label: self._set_view(self._view_by_label[label]),
            selected_color=Colors.PRIMARY,
            selected_hover_color=Colors.PRIMARY_HOVER,
            unselected_color=Colors.BG_CARD,
            unselected_hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
        )
        self._view_selector.pack(side="left")
        self._view_selector.set(t("interactive_view_restored"))

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
        self.update_idletasks()
        scaling.place_centered_on_parent(
            self, master, *scaling.to_physical(self, self._window_w, self._window_h)
        )

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
        self._worker.request(token, self._index, self._seed)

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
        self._on_log("ERROR", message)

    def _show_current_view(self) -> None:
        image = self._current_result
        if self._view == "mask":
            image = self._current_mask
        elif self._view == "raw":
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
        self._on_log("INFO", f"Wrote {path}")

    def _close(self) -> None:
        self._closed = True
        self._worker.close()
        self._main_thread.close()
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
