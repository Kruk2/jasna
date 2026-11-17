"""Command-line prompts that download the LTX model files and the DLSS 5 runtime a job needs."""

import sys

from jasna.os_utils import confirm_in_terminal
from jasna.session_config import SessionConfig


def ensure_ltx_model(config: SessionConfig, segments) -> None:
    """Offer to download the selected LTX model when the job needs it and it is missing.
    Without a terminal to ask in, fail with what to download instead of waiting."""
    from jasna.ltx.model_files import LTX_TRIAL_NOTICE, download_files, download_size_text, missing_downloads

    from jasna.segments import job_models

    if "ltx" not in job_models(config.restoration_model_name, segments):
        return
    from jasna.accelerator import supports_nvfp4
    from jasna.session_factory import LTX_FAST_NEEDS_BLACKWELL

    if config.ltx_fast and not supports_nvfp4(config.device):
        sys.exit(f"Error: {LTX_FAST_NEEDS_BLACKWELL}")
    if config.ltx_trial:
        print(LTX_TRIAL_NOTICE)
        return
    directory = config.model_path("ltx")
    missing = missing_downloads(
        directory,
        config.ltx_model,
        fast=config.ltx_fast,
        encoder=config.ltx_vae_encoder,
        decoder=config.ltx_vae_decoder,
    )
    if not missing:
        return
    from jasna.license_api import ltx_driver_mismatch, require_protection

    require_protection()
    mismatch = ltx_driver_mismatch()
    if mismatch is not None:
        sys.exit(
            f"Error: LTX can't use {mismatch.path}: it is from NVIDIA driver {mismatch.library_version}, but this "
            f"machine runs driver {mismatch.driver_version}. It usually comes from a CUDA compatibility package in "
            "the container. Remove its folder from /etc/ld.so.conf.d and run ldconfig, or use an image without it."
        )
    names = ", ".join(file.name for file in missing)
    size = download_size_text(missing)
    updating = any((directory / file.name).is_file() for file in missing)
    status = "need updating" if updating else "missing"
    message = f"LTX model files {status} in {directory}: {names}."
    if updating:
        message += " Models copied from older Jasna versions cannot be used. Your existing License code still works."
    terminal_hint = f"Error: {message} Run jasna in a terminal to download them ({size})."
    answer = confirm_in_terminal(f"{message}\nDownload ({size})? [y/N]: ")
    if answer is None:
        sys.exit(terminal_hint)
    if not answer:
        sys.exit(f"Error: {message}")
    from tqdm import tqdm

    from jasna.user_settings import user_language

    with tqdm(total=sum(file.size_bytes for file in missing), unit="B", unit_scale=True, desc="LTX download") as bar:
        download_files(directory, missing, lambda done, total: bar.update(done - bar.n), mirror_first=user_language() == "zh")


def ensure_dlss5_runtime(config: SessionConfig) -> None:
    """Offer to download the DLSS 5 runtime, which Jasna does not ship, when the job needs it."""
    if config.upscale_model != "dlss5" or sys.platform != "win32":
        return
    from jasna.upscale.dlss5 import (
        DLSS5_RELEASE_PAGE,
        DLSS5_RELEASE_SIZE,
        download_dlss5_runtime,
        dlss5_runtime_dir,
        missing_runtime_files,
    )

    directory = dlss5_runtime_dir()
    if not missing_runtime_files(directory):
        return
    size = f"{DLSS5_RELEASE_SIZE / 1e6:.0f} MB"
    message = (
        f"DLSS 5 is not part of Jasna. It runs an unofficial community runtime from {DLSS5_RELEASE_PAGE} "
        f"(NVIDIA files modified to run outside games)."
    )
    answer = confirm_in_terminal(f"{message}\nDownload it ({size}) into {directory}? [y/N]: ")
    if answer is None:
        sys.exit(f"Error: {message} Run jasna in a terminal to download it ({size}).")
    if not answer:
        sys.exit(f"Error: {message} DLSS 5 upscaling needs this download.")
    from tqdm import tqdm

    with tqdm(total=DLSS5_RELEASE_SIZE, unit="B", unit_scale=True, desc="DLSS 5 download") as bar:
        download_dlss5_runtime(directory, lambda done, total: bar.update(done - bar.n))
