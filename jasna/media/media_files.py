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


def folder_media_in_processing_order(folder: str | Path) -> list[Path]:
    """Return recursive media files grouped by model type: images first, then videos."""
    folder = Path(folder)

    def sort_key(path: Path) -> tuple[int, str]:
        relative = path.relative_to(folder)
        return len(relative.parts), relative.as_posix().casefold()

    entries = sorted((p for p in folder.rglob("*") if p.is_file()), key=sort_key)
    images = [p for p in entries if is_image(p)]
    videos = [p for p in entries if is_video(p)]
    return images + videos


def folder_output_path(output_dir: str | Path, input_path: str | Path, output_pattern: str | None = None) -> Path:
    """Per-file output path for folder batches.

    Without a pattern, writes ``<output_dir>/<stem>_out<ext>``. With a pattern,
    replaces ``{original}`` with the input stem. Image outputs keep their source
    extension; video outputs use the pattern extension when one is provided.
    """
    input_path = Path(input_path)
    if not output_pattern:
        return Path(output_dir) / f"{input_path.stem}_out{input_path.suffix}"

    output_name = output_pattern.replace("{original}", input_path.stem)
    output_path = Path(output_dir) / output_name
    if is_image(input_path) or not output_path.suffix:
        output_path = output_path.with_suffix(input_path.suffix)
    return output_path
