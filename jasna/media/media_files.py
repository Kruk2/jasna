from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
)
MEDIA_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
IGNORED_RECURSIVE_MEDIA_DIRECTORIES: frozenset[str] = frozenset({".jasna-logs"})


def file_dialog_patterns(extensions: frozenset[str]) -> str:
    return " ".join(f"*{extension}" for extension in sorted(extensions))


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

    def is_ignored(path: Path) -> bool:
        try:
            return bool(
                IGNORED_RECURSIVE_MEDIA_DIRECTORIES
                & set(path.relative_to(folder).parts[:-1])
            )
        except ValueError:
            return False

    def sort_key(path: Path) -> tuple[int, str]:
        relative = path.relative_to(folder)
        return len(relative.parts), relative.as_posix().casefold()

    entries = sorted(
        (p for p in folder.rglob("*") if p.is_file() and not is_ignored(p)),
        key=sort_key,
    )
    images = [p for p in entries if is_image(p)]
    videos = [p for p in entries if is_video(p)]
    return images + videos


def folder_output_path(
    output_dir: str | Path,
    input_path: str | Path,
    output_pattern: str | None = None,
    *,
    input_root: str | Path | None = None,
    preserve_structure: bool = False,
) -> Path:
    """Per-file output path for folder batches.

    Without a pattern, writes ``<output_dir>/<stem>_out<ext>``. With a pattern,
    replaces ``{original}`` with the input stem. Image outputs keep their source
    extension; video outputs use the pattern extension when one is provided.
    When requested, the input's parent path relative to ``input_root`` is
    recreated below ``output_dir``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if preserve_structure and input_root is not None:
        try:
            relative_parent = input_path.resolve().relative_to(
                Path(input_root).resolve()
            ).parent
        except ValueError:
            # A manually added file has no meaningful path below the selected
            # folder root, so retain the established flat-output behavior.
            relative_parent = Path()
        output_dir /= relative_parent
    if not output_pattern:
        return output_dir / f"{input_path.stem}_out{input_path.suffix}"

    output_name = output_pattern.replace("{original}", input_path.stem)
    output_path = output_dir / output_name
    if is_image(input_path) or not output_path.suffix:
        output_path = output_path.with_suffix(input_path.suffix)
    return output_path


def unique_path(path: Path) -> Path:
    """Return ``path``, or ``<stem> (N)<suffix>`` for the first N that does not exist."""
    if not path.exists():
        return path
    for counter in range(1, 10000):
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique filename after 9999 attempts: {path}")
