from __future__ import annotations

from pathlib import Path

import pytest

from jasna.media.media_files import (
    classify_folder,
    file_dialog_patterns,
    folder_output_path,
    folder_media_in_processing_order,
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


class TestFolderMediaInProcessingOrder:
    def test_recurses_and_groups_images_before_videos(self, tmp_path: Path):
        nested = tmp_path / "nested"
        nested.mkdir()
        for name in [
            "clip_b.mp4",
            "photo_a.png",
            "nested/clip_a.webm",
            "nested/photo_b.JPG",
            "notes.txt",
        ]:
            (tmp_path / name).write_bytes(b"x")

        ordered = folder_media_in_processing_order(tmp_path)

        assert [p.relative_to(tmp_path).as_posix() for p in ordered] == [
            "photo_a.png",
            "nested/photo_b.JPG",
            "clip_b.mp4",
            "nested/clip_a.webm",
        ]


@pytest.mark.parametrize(
    ("source", "pattern", "output"),
    [
        ("photo.png", None, "photo_out.png"),
        ("clip.mkv", None, "clip_out.mkv"),
        ("clip.mkv", "{original}_restored.mp4", "clip_restored.mp4"),
        ("photo.jpg", "{original}_restored.mp4", "photo_restored.jpg"),
        ("clip.mkv", "{original}_restored", "clip_restored.mkv"),
        ("clip.v2.mkv", "{original}_restored", "clip.v2_restored.mkv"),
        ("a.b.png", "{original}_restored", "a.b_restored.png"),
        ("a.b.jpg", "{original}_restored.mp4", "a.b_restored.jpg"),
        ("clip.v2.mkv", "{original}.mp4", "clip.v2.mp4"),
    ],
)
def test_folder_output_path_names_videos_by_pattern_and_keeps_image_extensions(source, pattern, output):
    arguments = (Path("/out"), Path("/in") / source) + ((pattern,) if pattern else ())
    assert folder_output_path(*arguments) == Path("/out") / output


def test_file_dialog_patterns_are_sorted_globs():
    assert file_dialog_patterns(frozenset({".png", ".jpg"})) == "*.jpg *.png"


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [("out.mp4", True), ("out (3).mp4", True), ("out (x).mp4", False), ("out (1).mkv", False), ("other.mp4", False)],
)
def test_is_unique_path_of(tmp_path: Path, candidate, expected):
    from jasna.media.media_files import is_unique_path_of

    assert is_unique_path_of(tmp_path / candidate, tmp_path / "out.mp4") is expected
    assert not is_unique_path_of(tmp_path / "sub" / candidate, tmp_path / "out.mp4")
