"""Downloads of the published LTX model files (``model_files.LTX_DOWNLOADS``), with a mirror fallback."""
from __future__ import annotations

import hashlib
import http.client
import logging
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import certifi

from jasna.ltx.model_files import LTX_MODEL_BRANCH, DownloadableFile
from jasna.os_utils import commit_file

logger = logging.getLogger(__name__)

LTX_DOWNLOAD_URL = f"https://huggingface.co/Kruk2/ltx-restore/resolve/{LTX_MODEL_BRANCH}"
LTX_MIRROR_URL = f"https://hf-mirror.com/Kruk2/ltx-restore/resolve/{LTX_MODEL_BRANCH}"
_NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)

DownloadProgressCallback = Callable[[int, int], None]


def download_size_text(files: list[DownloadableFile]) -> str:
    return f"{sum(f.size_bytes for f in files) / 1e9:.1f} GB"


def _open_download(url: str, offset: int, context: ssl.SSLContext):
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    response = urllib.request.urlopen(request, timeout=30, context=context)
    if response.status != 206:
        return response, 0
    if offset and not (response.headers.get("Content-Range") or "").startswith(f"bytes {offset}-"):
        response.close()  # a range other than the one asked for can't extend the partial file
        return _open_download(url, 0, context)
    return response, offset


def download_files(
    directory: Path,
    files: list[DownloadableFile],
    progress_callback: DownloadProgressCallback | None,
    *,
    mirror_first: bool,
) -> None:
    """Download ``files`` into ``directory``, verifying each sha256 before it replaces
    anything. Each file falls back to the other source when the first is unreachable;
    ``mirror_first`` tries the mirror first. An interrupted download resumes from its ``.part``
    file. ``progress_callback(done_bytes, total_bytes)`` covers all files together."""
    context = ssl.create_default_context(cafile=certifi.where())
    sources = (LTX_MIRROR_URL, LTX_DOWNLOAD_URL) if mirror_first else (LTX_DOWNLOAD_URL, LTX_MIRROR_URL)
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(f.size_bytes for f in files)
    done = 0
    for file in files:
        target = directory / file.name
        partial = target.with_name(target.name + ".part")
        manual_hint = f"Download it yourself from {sources[0]}/{file.name} and put it in {directory}"
        for attempt in range(2):
            for source, fallback in zip(sources, (*sources[1:], None)):
                url = f"{source}/{file.name}"
                offset = partial.stat().st_size if partial.is_file() else 0
                if offset >= file.size_bytes or attempt:
                    offset = 0
                logger.info("Downloading %s", url)
                try:
                    response, offset = _open_download(url, offset, context)
                    with response:
                        digest = hashlib.sha256()
                        if offset:
                            with partial.open("rb") as resumed:
                                digest = hashlib.file_digest(resumed, "sha256")
                        received = offset
                        with partial.open("ab" if offset else "wb") as out:
                            while chunk := response.read(1 << 22):
                                out.write(chunk)
                                digest.update(chunk)
                                received += len(chunk)
                                if progress_callback is not None:
                                    progress_callback(done + received, total)
                    break
                except _NETWORK_ERRORS as error:
                    if fallback is None:
                        raise RuntimeError(f"Could not download {file.name} ({error}). {manual_hint}.") from error
                    logger.warning("Downloading %s failed (%s); trying %s", url, error, fallback)
            if digest.hexdigest() == file.sha256:
                break
            partial.unlink()
            if not offset:
                raise RuntimeError(f"{file.name} is corrupted (checksum mismatch). The download was deleted. {manual_hint}.")
            # A stale .part (power loss, or the file republished under the same name) poisons a resume.
            logger.warning("Resumed download of %s failed its checksum; downloading it again from the start", file.name)
        done += received
        commit_file(partial, target)
