from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import jasna.restorer.sd15_download as dl
from jasna.engine_paths import SD15_CKPT_ENC_PATH


def _make_bundle(model_dir: Path) -> None:
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "unet" / "config.json").write_text("{}")
    (model_dir / SD15_CKPT_ENC_PATH.name).write_bytes(b"x")


class TestBundlePresent:
    def test_present_with_enc_and_config(self, tmp_path: Path):
        _make_bundle(tmp_path)
        assert dl.bundle_present(tmp_path) is True

    def test_absent_without_ckpt(self, tmp_path: Path):
        (tmp_path / "unet").mkdir()
        (tmp_path / "unet" / "config.json").write_text("{}")
        assert dl.bundle_present(tmp_path) is False

    def test_absent_without_config(self, tmp_path: Path):
        (tmp_path / SD15_CKPT_ENC_PATH.name).write_bytes(b"x")
        assert dl.bundle_present(tmp_path) is False


class TestEnsureBundle:
    def test_noop_when_present(self, tmp_path: Path):
        _make_bundle(tmp_path)
        with patch.object(dl, "download_sd15_bundle") as mock_dl, patch("builtins.input") as mock_input:
            dl.ensure_sd15_bundle(tmp_path)
            mock_dl.assert_not_called()
            mock_input.assert_not_called()

    def test_declined_raises(self, tmp_path: Path):
        with patch("builtins.input", return_value="n"), patch.object(dl, "download_sd15_bundle") as mock_dl:
            with pytest.raises(FileNotFoundError):
                dl.ensure_sd15_bundle(tmp_path / "missing")
            mock_dl.assert_not_called()

    def test_accepted_downloads(self, tmp_path: Path):
        target = tmp_path / "bundle"

        def fake_download(model_dir, repo_id=dl.SD15_HF_REPO):
            _make_bundle(Path(model_dir))

        with patch("builtins.input", return_value="y"), patch.object(dl, "download_sd15_bundle", side_effect=fake_download) as mock_dl:
            dl.ensure_sd15_bundle(target)
            mock_dl.assert_called_once()
            assert dl.bundle_present(target)


class TestDownloadBundle:
    def test_without_callback_uses_default_hf_progress(self, tmp_path: Path):
        with patch("huggingface_hub.snapshot_download") as mock_snapshot:
            dl.download_sd15_bundle(tmp_path)

        kwargs = mock_snapshot.call_args.kwargs
        assert kwargs["local_dir"] == str(tmp_path)
        assert "tqdm_class" not in kwargs

    def test_callback_uses_silent_byte_progress_class(self, tmp_path: Path):
        events: list[tuple[int, int | None]] = []

        with patch("huggingface_hub.snapshot_download") as mock_snapshot:
            dl.download_sd15_bundle(tmp_path, progress_callback=lambda done, total: events.append((done, total)))

        progress_cls = mock_snapshot.call_args.kwargs["tqdm_class"]
        byte_bar = progress_cls(total=100, initial=10, unit="B")
        byte_bar.update(15)
        byte_bar.total += 25
        byte_bar.refresh()

        file_bar = progress_cls([1, 2], total=2)

        assert list(file_bar) == [1, 2]
        assert events == [(10, 100), (25, 100), (25, 125)]
