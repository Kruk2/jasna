from __future__ import annotations

from pathlib import Path

from jasna.media.image_io import IMAGE_EXTENSIONS

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
)
MEDIA_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_media(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def classify_folder(folder: str | Path) -> tuple[list[Path], list[Path]]:
    """Return ``(images, videos)`` — sorted top-level media files in ``folder``."""
    folder = Path(folder)
    entries = sorted(p for p in folder.iterdir() if p.is_file())
    images = [p for p in entries if is_image(p)]
    videos = [p for p in entries if is_video(p)]
    return images, videos


def folder_output_path(output_dir: str | Path, input_path: str | Path) -> Path:
    """Per-file output path for folder batches: ``<output_dir>/<stem>_out<ext>``."""
    input_path = Path(input_path)
    return Path(output_dir) / f"{input_path.stem}_out{input_path.suffix}"
