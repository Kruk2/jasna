from pathlib import Path

from jasna.gui import sd15_download_worker


def test_worker_reports_deduplicated_percent_and_success(monkeypatch, tmp_path: Path) -> None:
    def fake_download(model_dir, repo_id, progress_callback):
        assert model_dir == tmp_path
        assert repo_id == "some/repo"
        progress_callback(0, None)
        progress_callback(50, 100)
        progress_callback(50, 100)
        progress_callback(100, 100)

    monkeypatch.setattr(sd15_download_worker, "download_sd15_bundle", fake_download)

    percents: list[int] = []
    done: list[str | None] = []
    thread = sd15_download_worker.start_sd15_download(
        tmp_path, "some/repo", percents.append, done.append
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert percents == [50, 100]
    assert done == [None]


def test_worker_reports_error_text_on_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_download(model_dir, repo_id, progress_callback):
        raise RuntimeError("disk full")

    monkeypatch.setattr(sd15_download_worker, "download_sd15_bundle", fake_download)

    done: list[str | None] = []
    thread = sd15_download_worker.start_sd15_download(
        tmp_path, "some/repo", lambda _percent: None, done.append
    )
    thread.join(timeout=5)

    assert done == ["disk full"]
