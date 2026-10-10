"""GUI output-path construction with selected-root containment."""

from __future__ import annotations

import ntpath
import posixpath
from pathlib import Path

from jasna.media.media_files import folder_output_path


class OutputPathError(ValueError):
    """Raised when a GUI output template would leave its selected root."""


def _relative_output_name(
    input_path: str | Path,
    output_pattern: str | None,
) -> str:
    """Expand the output template and reject either OS's path escapes."""

    source = Path(input_path)
    if output_pattern:
        output_name = str(output_pattern).replace("{original}", source.stem)
    else:
        output_name = f"{source.stem}_out{source.suffix}"

    # Presets can be shared between Windows and Linux.  Validate both path
    # syntaxes so a drive, slash, or parent traversal cannot become an
    # unexpected write after a preset moves to the other platform.
    for path_module in (ntpath, posixpath):
        normalized = path_module.normpath(output_name)
        if (
            path_module.isabs(output_name)
            or (path_module is ntpath and ntpath.splitdrive(output_name)[0])
            or normalized == ".."
            or normalized.startswith(".." + path_module.sep)
        ):
            raise OutputPathError("output path escapes the selected output folder")
    return output_name


def job_output_path(
    output_dir: str | Path,
    input_path: str | Path,
    output_pattern: str | None,
    *,
    input_root: str | Path | None = None,
    preserve_structure: bool = False,
) -> Path:
    """Build one GUI output path and keep it below ``output_dir``.

    ``folder_output_path`` owns the established filename and extension rules;
    this wrapper adds the containment check needed for a user-editable GUI
    template.  Parent directories are created by the processor immediately
    before a job starts, after Stop has been checked.
    """

    try:
        _relative_output_name(input_path, output_pattern)
        root = Path(output_dir).expanduser().resolve(strict=False)
        output = folder_output_path(
            output_dir,
            input_path,
            output_pattern,
            input_root=input_root,
            preserve_structure=preserve_structure,
        )
        resolved_output = output.expanduser().resolve(strict=False)
    except OutputPathError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise OutputPathError("could not safely resolve the output path") from error

    try:
        resolved_output.relative_to(root)
    except ValueError as error:
        raise OutputPathError(
            "output path escapes the selected output folder"
        ) from error
    return output
