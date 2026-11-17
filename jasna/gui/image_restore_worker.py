"""Background renderer of the interactive image restore dialog."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jasna.gui.models import AppSettings
from jasna.gui.queues import CommandWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Render:
    token: int
    index: int
    seed: int


class ImageRestoreWorker(CommandWorker):
    """Restores one image of ``paths`` with one seed per request, the newest request first, with
    the still-image session it loads on first use. Calls ``on_rendered(token, index, seed, raw,
    mask, result)`` or ``on_failed(token, exc)`` and finally ``on_stopped()`` on its own thread."""

    def __init__(
        self,
        paths: list[Path],
        settings: AppSettings,
        *,
        on_rendered: Callable[[int, int, int, object, object, object], None],
        on_failed: Callable[[int, Exception], None],
        on_stopped: Callable[[], None],
    ) -> None:
        super().__init__("image-restore")
        self._paths = paths
        self._settings = settings
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
