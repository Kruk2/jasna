import subprocess
from unittest.mock import Mock

import pytest
import torch

from jasna import os_utils

pytestmark = pytest.mark.usefixtures("amd_driver_environment")


@pytest.mark.parametrize(
    "output,status,text",
    [
        ("32.0.21017.2051\n", None, "older than 32.0.21017.2052"),
        ("32.0.21017.2052\n", True, "32.0.21017.2052"),
        ("32.0.21017.2053\n", True, "32.0.21017.2053"),
        ("32.0.31041.5005\n", True, "32.0.31041.5005"),
        ("32.0.21017.2052\n32.0.21017.2052\n", True, "32.0.21017.2052"),
        ("32.0.21001.1\n32.0.21017.2052\n", None, "could not be checked"),
        ("", None, "could not be checked"),
        ("not a version", None, "could not be checked"),
        ("32.0.21017.2052\ninvalid", None, "could not be checked"),
    ],
)
def test_windows_amd_driver_minimum_only_warns(monkeypatch, output, status, text):
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(os_utils.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, output, ""))

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is status
    assert text in info


@pytest.mark.parametrize(
    "failure",
    [OSError("not found"), subprocess.TimeoutExpired("powershell", 10)],
)
def test_unavailable_windows_query_reports_an_unverified_driver(monkeypatch, failure):
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(os_utils.subprocess, "run", Mock(side_effect=failure))

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is None
    assert "could not be checked" in info


def test_failed_windows_query_does_not_trust_partial_output(monkeypatch):
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(
        os_utils.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, "32.0.31041.5005\n", "CIM unavailable"),
    )

    assert os_utils.check_gpu_driver_version()[0] is None


@pytest.mark.parametrize(
    "version,status,text",
    [
        ("6.12.12.60404", None, "older than AMD driver 30.10.2"),
        ("6.14.14.30100100", None, "older than AMD driver 30.10.2"),
        ("6.14.14.30100200", True, "6.14.14.30100200"),
        ("6.14.14.30200000", True, "6.14.14.30200000"),
        ("7.1.9.31600000", True, "7.1.9.31600000"),
        ("6.14.14", None, "could not be checked"),
        ("", None, "could not be checked"),
        ("invalid", None, "could not be checked"),
    ],
)
def test_linux_checks_the_loaded_driver_release_and_only_warns(monkeypatch, tmp_path, version, status, text):
    (tmp_path / "version").write_text(version + "\n")
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(os_utils, "Path", lambda _: tmp_path)

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is status
    assert text in info


@pytest.mark.parametrize(
    "module_loaded,status,text",
    [(True, True, "in-kernel amdgpu"), (False, None, "could not be checked (hidden sysfs)")],
)
def test_linux_without_module_version_trusts_the_in_kernel_driver(monkeypatch, tmp_path, module_loaded, status, text):
    if module_loaded:
        (tmp_path / "amdgpu").mkdir()
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(os_utils, "Path", lambda _: tmp_path / "amdgpu")

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is status
    assert text in info


def test_a_current_driver_does_not_hide_a_missing_gpu(monkeypatch):
    monkeypatch.setattr(os_utils, "_check_amd_driver_version", lambda: (True, "current driver"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is False
    assert "no AMD GPU is available" in info
