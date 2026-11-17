from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import jasna.restorer.sd15_download as dl
from jasna.engine_paths import SD15_CKPT_ENC_PATH, SD15_CKPT_PATH


def _make_bundle(model_dir: Path) -> None:
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "unet" / "config.json").write_text("{}")
    (model_dir / SD15_CKPT_ENC_PATH.name).write_bytes(b"x")


@pytest.mark.parametrize(
    ("files", "present"),
    [
        (("unet/config.json", SD15_CKPT_ENC_PATH.name), True),
        (("unet/config.json", SD15_CKPT_PATH.name), True),
        (("unet/config.json",), False),
        ((SD15_CKPT_ENC_PATH.name,), False),
    ],
    ids=["encrypted", "plaintext", "no-checkpoint", "no-config"],
)
def test_bundle_needs_a_checkpoint_and_the_unet_config(tmp_path: Path, files, present):
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")

    assert dl.bundle_present(tmp_path) is present


def _closed_stdin(prompt):
    raise EOFError


class TestEnsureBundle:
    def test_noop_when_present(self, tmp_path: Path):
        _make_bundle(tmp_path)
        with patch.object(dl, "download_sd15_bundle") as mock_dl, patch("builtins.input") as mock_input:
            dl.ensure_sd15_bundle(tmp_path)
            mock_dl.assert_not_called()
            mock_input.assert_not_called()

    @pytest.mark.parametrize(
        ("isatty", "answer", "message"),
        [
            (True, lambda prompt: "n", "Get it from https://huggingface.co/"),
            (False, lambda prompt: "y", "Run jasna in a terminal to download it"),
            (True, _closed_stdin, "Run jasna in a terminal to download it"),
        ],
        ids=["declined", "no-terminal", "nul-stdin"],
    )
    def test_exits_with_a_message_without_downloading(self, tmp_path: Path, monkeypatch, isatty, answer, message):
        monkeypatch.setattr("sys.stdin.isatty", lambda: isatty)
        monkeypatch.setattr("builtins.input", answer)
        with patch.object(dl, "download_sd15_bundle") as mock_dl:
            with pytest.raises(SystemExit, match=f"^Error: SD15 model not found .*{message}"):
                dl.ensure_sd15_bundle(tmp_path / "missing")
            mock_dl.assert_not_called()

    def test_accepted_downloads(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        target = tmp_path / "bundle"

        def fake_download(model_dir, repo_id=dl.SD15_HF_REPO):
            _make_bundle(Path(model_dir))

        with patch("builtins.input", return_value="y"), patch.object(dl, "download_sd15_bundle", side_effect=fake_download) as mock_dl:
            dl.ensure_sd15_bundle(target)
            mock_dl.assert_called_once()
            assert dl.bundle_present(target)


def test_skips_retired_checkpoint_but_not_the_current_one(tmp_path: Path):
    from fnmatch import fnmatch

    with patch("huggingface_hub.snapshot_download") as mock_snapshot:
        dl.download_sd15_bundle(tmp_path)

    ignored = mock_snapshot.call_args.kwargs["ignore_patterns"]
    assert any(fnmatch("sd15-200000.v2.ckpt.enc", pattern) for pattern in ignored)
    assert not any(fnmatch(SD15_CKPT_ENC_PATH.name, pattern) for pattern in ignored)


def test_download_progress_reports_bytes_silently():
    events: list[tuple[int, int | None]] = []
    progress_cls = dl._progress_tqdm_class(lambda done, total: events.append((done, total)))

    byte_bar = progress_cls(total=100, initial=10, unit="B")
    byte_bar.update(15)
    byte_bar.total += 25
    byte_bar.refresh()
    file_bar = progress_cls([1, 2], total=2)

    assert list(file_bar) == [1, 2]
    assert events == [(10, 100), (25, 100), (25, 125)]
