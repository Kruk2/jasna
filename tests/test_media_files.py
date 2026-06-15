from __future__ import annotations

from pathlib import Path

from jasna.media.media_files import (
    classify_folder,
    folder_output_path,
    is_image,
    is_video,
)


class TestPredicates:
    def test_is_image(self):
        assert is_image("a.PNG") and is_image("b.jpeg")
        assert not is_image("c.mp4")

    def test_is_video(self):
        assert is_video("a.MP4") and is_video("b.mkv")
        assert not is_video("c.png")


class TestClassifyFolder:
    def test_groups_and_sorts(self, tmp_path: Path):
        for name in ["b.png", "a.png", "c.mp4", "d.mkv", "notes.txt", "e.JPG"]:
            (tmp_path / name).write_bytes(b"x")
        (tmp_path / "subdir").mkdir()  # ignored (top-level only)
        images, videos = classify_folder(tmp_path)
        assert [p.name for p in images] == ["a.png", "b.png", "e.JPG"]
        assert [p.name for p in videos] == ["c.mp4", "d.mkv"]

    def test_empty_folder(self, tmp_path: Path):
        images, videos = classify_folder(tmp_path)
        assert images == [] and videos == []


class TestFolderOutputPath:
    def test_out_suffix_and_dir(self):
        out = folder_output_path(Path("/out"), Path("/in/photo.png"))
        assert out == Path("/out/photo_out.png")

    def test_preserves_extension(self):
        out = folder_output_path(Path("/out"), Path("/in/clip.mkv"))
        assert out == Path("/out/clip_out.mkv")
