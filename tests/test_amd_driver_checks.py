import subprocess
from unittest.mock import Mock

import pytest
import torch

from jasna import os_utils

pytestmark = pytest.mark.usefixtures("amd_driver_environment")


@pytest.mark.parametrize(
    "output,status",
    [
        ("32.0.31041.5004\n", False),
        ("32.0.31041.5005\n", True),
        ("32.0.31041.5006\n", True),
        ("32.0.31042.1\n", True),
        ("32.0.31041.5005\n32.0.31041.5005\n", True),
        ("32.0.21001.1\n32.0.31041.5005\n", None),
        ("", None),
        ("not a version", None),
        ("32.0.31041.5005\ninvalid", None),
    ],
)
def test_windows_amd_driver_minimum(monkeypatch, output, status):
    monkeypatch.setattr(os_utils.sys, "platform", "win32")
    monkeypatch.setattr(os_utils.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, output, ""))

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is status
    if status is False:
        assert "32.0.31041.5005+" in info
        assert "reboot" in info
    elif status is None:
        assert "could not be checked" in info
    else:
        assert "32.0." in info


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
    "version,status",
    [
        ("6.12.12.60404", False),
        ("6.14.14.30100100", False),
        ("6.14.14.30100200", True),
        ("6.14.14.30200000", True),
        ("7.1.9.31600000", True),
        ("6.14.14", None),
        ("", None),
        ("invalid", None),
    ],
)
def test_linux_checks_the_loaded_driver_release(monkeypatch, tmp_path, version, status):
    driver_version = tmp_path / "version"
    driver_version.write_text(version + "\n")
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(os_utils, "Path", lambda _: driver_version)

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is status
    if status is False:
        assert "30.10.2+" in info
        assert "host driver" in info
    elif status is None:
        assert "could not be checked" in info
    else:
        assert version in info


def test_linux_without_module_version_warns_instead_of_rejecting_the_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(os_utils, "Path", lambda _: tmp_path / "missing")

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is None
    assert "distribution kernel or hidden sysfs" in info


def test_old_driver_is_rejected_before_initializing_hip(monkeypatch):
    monkeypatch.setattr(os_utils, "_check_amd_driver_version", lambda: (False, "driver too old"))
    monkeypatch.setattr(torch.cuda, "is_available", Mock(side_effect=AssertionError("initialized HIP")))

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is False
    assert "driver too old" in info


def test_a_current_driver_does_not_hide_a_missing_gpu(monkeypatch):
    monkeypatch.setattr(os_utils, "_check_amd_driver_version", lambda: (True, "current driver"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ok, info = os_utils.check_gpu_driver_version()

    assert ok is False
    assert "no AMD GPU is available" in info
