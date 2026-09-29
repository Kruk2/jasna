from jasna.gui import download_worker


def test_worker_reports_deduplicated_percent_and_success() -> None:
    def fake_download(progress_callback):
        progress_callback(0, None)
        progress_callback(50, 100)
        progress_callback(50, 100)
        progress_callback(100, 100)

    percents: list[int] = []
    done: list[str | None] = []
    thread = download_worker.start_download(fake_download, percents.append, done.append)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert percents == [50, 100]
    assert done == [None]


def test_worker_reports_error_text_on_failure() -> None:
    def fake_download(progress_callback):
        raise RuntimeError("disk full")

    done: list[str | None] = []
    thread = download_worker.start_download(fake_download, lambda _percent: None, done.append)
    thread.join(timeout=5)

    assert done == ["disk full"]
