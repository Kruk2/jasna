from __future__ import annotations

from pathlib import Path

from jasna.gui.log_export import export_log_entries_txt, format_log_entries


def test_export_log_entries_txt_writes_expected_content(tmp_path: Path) -> None:
    entries = [
        ("01:02:03 PM", "INFO", "hello"),
        ("01:02:04 PM", "WARNING", "warn"),
        ("01:02:05 PM", "ERROR", "boom"),
    ]

    out_path = tmp_path / "logs.txt"
    export_log_entries_txt(out_path, entries)

    expected = format_log_entries(entries)
    assert out_path.read_text(encoding="utf-8") == expected

