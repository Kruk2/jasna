"""Background renderer of the interactive image restore dialog."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from jasna.gui.models import DEFAULT_OUTPUT_PATTERN, AppSettings
from jasna.gui.queues import CommandWorker
from jasna.gui.workers import run_in_background
from jasna.media.media_files import folder_output_path, unique_path

logger = logging.getLogger(__name__)

_save_lock = threading.Lock()


def interactive_output_path(input_path: Path, output_folder: str, output_pattern: str) -> Path:
    input_path = Path(input_path)
    return unique_path(folder_output_path(output_folder or input_path.parent, input_path, output_pattern or DEFAULT_OUTPUT_PATTERN))


def save_restored_image(
    source: Path,
    output_folder: str,
    output_pattern: str,
    image,
    on_done: Callable[[Path | None, Exception | None], None],
) -> None:
    """Write ``image`` to the next free output name of ``source`` on a background thread, then
    ``on_done(path, None)`` or ``on_done(None, error)`` on that thread. Saves run one at a time,
    so two quick saves never pick the same free name."""
    from jasna.media import image_io

    def write() -> Path:
        with _save_lock:
            path = interactive_output_path(source, output_folder, output_pattern)
            path.parent.mkdir(parents=True, exist_ok=True)
            image_io.write_image_rgb_chw(path, image)
            return path

    run_in_background(write, on_done, name="image-restore-save")


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


@dataclass(frozen=True)
class _Render:
    token: int
    index: int
    seed: int


class ImageRestoreWorker(CommandWorker):
    """Restores one image of ``paths`` with one seed per request, the newest request first, with
    the still-image session it loads on first use. First calls ``on_first_image_size(size)`` with
    the pixel size of ``paths[0]`` (``on_failed(0, exc)`` when it cannot be read), then per request
    ``on_rendered(token, index, seed, raw, mask, result)`` or ``on_failed(token, exc)``, and finally
    ``on_stopped()``, all on its own thread."""

    def __init__(
        self,
        paths: list[Path],
        settings: AppSettings,
        *,
        on_first_image_size: Callable[[tuple[int, int]], None],
        on_rendered: Callable[[int, int, int, object, object, object], None],
        on_failed: Callable[[int, Exception], None],
        on_stopped: Callable[[], None],
    ) -> None:
        super().__init__("image-restore")
        self._paths = paths
        self._settings = settings
        self._on_first_image_size = on_first_image_size
        self._on_rendered = on_rendered
        self._on_failed = on_failed
        self._on_stopped = on_stopped

    def request(self, token: int, index: int, seed: int) -> None:
        self._send(_Render(token, index, seed))

    def _run(self) -> None:
        from jasna.gui.video_session import build_image_session, release_session_memory

        session = None
        prepared_by_path = {}
        try:
            try:
                self._on_first_image_size(read_image_size(self._paths[0]))
            except Exception as exc:
                self._on_failed(0, exc)
            while (command := self._next_command()) is not None:
                try:
                    if session is None:
                        session = build_image_session(self._settings, log=None)
                    path = self._paths[command.index]
                    prepared = prepared_by_path.get(path)
                    if prepared is None:
                        prepared = self._prepare(path, session)
                        prepared_by_path[path] = prepared
                    result = self._render(prepared, session, command.seed)
                    mask = self._mask(prepared)
                    self._on_rendered(command.token, command.index, command.seed, prepared.img_chw_u8, mask, result)
                except Exception as exc:
                    self._on_failed(command.token, exc)
        finally:
            try:
                if session is not None:
                    session.close()
                    try:
                        release_session_memory(session.device)
                    except Exception:
                        logger.warning("Torch cleanup failed during render worker teardown", exc_info=True)
            finally:
                self._on_stopped()

    def _prepare(self, path: Path, session):
        from jasna.image_restore import prepare_image_restore
        from jasna.media import image_io

        img = image_io.read_image_rgb_chw(path)
        return prepare_image_restore(img, session.detector, device=session.device, fp16=session.fp16)

    def _render(self, prepared, session, seed: int):
        from jasna.accelerator import device_context
        from jasna.image_restore import clamp_strength, render_prepared_image
        from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

        freeu = dict(DEFAULT_FREEU) if bool(self._settings.image_restore_freeu) else None
        strength = clamp_strength(float(self._settings.image_restore_strength))
        with device_context(session.device):
            return render_prepared_image(
                prepared,
                session.restorer,
                steps=int(self._settings.image_restore_steps),
                strength=strength,
                seed=int(seed),
                freeu=freeu,
            )

    def _mask(self, prepared):
        from jasna.image_restore import mask_overlay_rgb_chw

        return mask_overlay_rgb_chw(prepared)
