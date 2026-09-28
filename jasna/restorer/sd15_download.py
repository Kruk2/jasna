from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from jasna.engine_paths import SD15_CKPT_ENC_PATH, SD15_CKPT_PATH, SD15_HF_REPO

logger = logging.getLogger(__name__)

# Checkpoints encrypted under retired keys; they stay on the HF repo for older app versions.
RETIRED_CKPT_ENC_NAMES = ["sd15-200000.ckpt.enc"]


def bundle_present(model_dir: Path) -> bool:
    """True when a usable SD15 bundle already exists at ``model_dir``.

    Requires the (encrypted or plaintext) checkpoint plus the public UNet config
    that the loader needs.
    """
    model_dir = Path(model_dir)
    ckpt_ok = (model_dir / SD15_CKPT_ENC_PATH.name).exists() or (model_dir / SD15_CKPT_PATH.name).exists()
    return ckpt_ok and (model_dir / "unet" / "config.json").exists()


DownloadProgressCallback = Callable[[int, int | None], None]


def _progress_tqdm_class(progress_callback: DownloadProgressCallback):
    """Build a silent tqdm-compatible class for Hugging Face download progress."""

    class CallbackTqdm:
        _lock = threading.RLock()

        def __init__(
            self,
            iterable: Iterable[Any] | None = None,
            *,
            total: int | float | None = None,
            initial: int | float = 0,
            unit: str | None = None,
            **_: Any,
        ) -> None:
            self.iterable = iterable
            self.total = total
            self.n = initial
            self._emit_bytes = unit == "B"
            if self._emit_bytes:
                self._emit()

        @classmethod
        def get_lock(cls):
            return cls._lock

        @classmethod
        def set_lock(cls, lock) -> None:
            cls._lock = lock

        def __iter__(self) -> Iterator[Any]:
            if self.iterable is None:
                return
            for item in self.iterable:
                yield item
                if not self._emit_bytes:
                    self.update(1)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.close()

        def update(self, n: int | float | None = 1) -> None:
            self.n += 1 if n is None else n
            if self._emit_bytes:
                self._emit()

        def refresh(self, *args: Any, **kwargs: Any) -> bool:
            if self._emit_bytes:
                self._emit()
            return True

        def close(self) -> None:
            pass

        def clear(self, *args: Any, **kwargs: Any) -> None:
            pass

        def display(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_description(self, *args: Any, **kwargs: Any) -> None:
            pass

        def _emit(self) -> None:
            total = None if self.total is None else max(0, int(self.total))
            progress_callback(max(0, int(self.n)), total)

    return CallbackTqdm


def download_sd15_bundle(
    model_dir: Path,
    repo_id: str = SD15_HF_REPO,
    progress_callback: DownloadProgressCallback | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    logger.info("Downloading SD15 bundle %s -> %s", repo_id, model_dir)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "local_dir": str(model_dir),
        "ignore_patterns": RETIRED_CKPT_ENC_NAMES,
    }
    if progress_callback is not None:
        kwargs["tqdm_class"] = _progress_tqdm_class(progress_callback)
    snapshot_download(**kwargs)


def ensure_sd15_bundle(model_dir: Path, repo_id: str = SD15_HF_REPO) -> None:
    """Make sure the SD15 bundle is present, asking before downloading.

    The encrypted checkpoint still needs a valid license to decrypt at load
    time, so the download itself is safe to offer to anyone.
    """
    model_dir = Path(model_dir)
    if bundle_present(model_dir):
        return

    prompt = (
        f"SD15 model not found at {model_dir}.\n"
        f"Download it (~6.9 GB) from https://huggingface.co/{repo_id} ? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    if answer not in ("y", "yes"):
        raise FileNotFoundError(
            f"SD15 model bundle missing at {model_dir} and download declined. "
            f"Get it from https://huggingface.co/{repo_id}."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    download_sd15_bundle(model_dir, repo_id)
    if not bundle_present(model_dir):
        raise RuntimeError(f"SD15 bundle still incomplete after download into {model_dir}.")
