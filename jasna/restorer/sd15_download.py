from __future__ import annotations

import logging
from pathlib import Path

from jasna.engine_paths import SD15_CKPT_ENC_PATH, SD15_CKPT_PATH, SD15_HF_REPO

logger = logging.getLogger(__name__)


def bundle_present(model_dir: Path) -> bool:
    """True when a usable SD15 bundle already exists at ``model_dir``.

    Requires the (encrypted or plaintext) checkpoint plus the public UNet config
    that the loader needs.
    """
    model_dir = Path(model_dir)
    ckpt_ok = (model_dir / SD15_CKPT_ENC_PATH.name).exists() or (model_dir / SD15_CKPT_PATH.name).exists()
    return ckpt_ok and (model_dir / "unet" / "config.json").exists()


def download_sd15_bundle(model_dir: Path, repo_id: str = SD15_HF_REPO) -> None:
    from huggingface_hub import snapshot_download

    logger.info("Downloading SD15 bundle %s -> %s", repo_id, model_dir)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(model_dir))


def ensure_sd15_bundle(model_dir: Path, repo_id: str = SD15_HF_REPO) -> None:
    """Make sure the SD15 bundle is present, asking before downloading.

    The encrypted checkpoint still needs a valid license to decrypt at load
    time, so the download itself is safe to offer to anyone.
    """
    model_dir = Path(model_dir)
    if bundle_present(model_dir):
        return

    prompt = (
        f"SD15 model not found at {model_dir}.\n"
        f"Download it (~6.9 GB) from https://huggingface.co/{repo_id} ? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    if answer not in ("y", "yes"):
        raise FileNotFoundError(
            f"SD15 model bundle missing at {model_dir} and download declined. "
            f"Get it from https://huggingface.co/{repo_id}."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    download_sd15_bundle(model_dir, repo_id)
    if not bundle_present(model_dir):
        raise RuntimeError(f"SD15 bundle still incomplete after download into {model_dir}.")
