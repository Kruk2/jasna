from __future__ import annotations

from pathlib import Path

from jasna.gui.log_export import export_log_entries_txt


def test_export_log_entries_txt_writes_expected_content(tmp_path: Path) -> None:
    entries = [
        ("01:02:03 PM", "INFO", "hello"),
        ("01:02:04 PM", "WARNING", "warn"),
        ("01:02:05 PM", "ERROR", "boom"),
    ]

    out_path = tmp_path / "logs.txt"
    export_log_entries_txt(out_path, entries)

    assert out_path.read_text(encoding="utf-8") == (
        "01:02:03 PM INFO     hello\n"
        "01:02:04 PM WARNING  warn\n"
        "01:02:05 PM ERROR    boom\n"
    )

