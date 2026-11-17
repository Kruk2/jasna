import platform
import subprocess
from unittest.mock import Mock

import pytest
import torch

from jasna import system_requirements
from jasna.system_requirements import _linux_amd_driver_status, _windows_amd_driver_status


@pytest.mark.parametrize(
    "returncode,output,status,text",
    [
        (0, "32.0.21017.2051\n", None, "older than 32.0.21017.2052"),
        (0, "32.0.21017.2052\n", True, "32.0.21017.2052"),
        (0, "32.0.21017.2053\n", True, "32.0.21017.2053"),
        (0, "32.0.31041.5005\n", True, "32.0.31041.5005"),
        (0, "32.0.21017.2052\n32.0.21017.2052\n", True, "32.0.21017.2052"),
        (0, "32.0.21001.1\n32.0.21017.2052\n", None, "could not be checked"),
        (0, "", None, "could not be checked"),
        (0, "not a version", None, "could not be checked"),
        (0, "32.0.21017.2052\ninvalid", None, "could not be checked"),
        (1, "32.0.31041.5005\n", None, "could not be checked"),
    ],
)
def test_windows_amd_driver_minimum_only_warns(returncode, output, status, text):
    ok, info = _windows_amd_driver_status(returncode, output)

    assert ok is status
    assert text in info


@pytest.mark.parametrize(
    "version,module_dir_exists,status,text",
    [
        ("6.12.12.60404", True, None, "older than AMD driver 30.10.2"),
        ("6.14.14.30100100", True, None, "older than AMD driver 30.10.2"),
        ("6.14.14.30100200", True, True, "6.14.14.30100200"),
        ("6.14.14.30200000", True, True, "6.14.14.30200000"),
        ("7.1.9.31600000", True, True, "7.1.9.31600000"),
        ("6.14.14", True, None, "could not be checked"),
        ("", True, None, "could not be checked"),
        ("invalid", True, None, "could not be checked"),
        (None, True, True, f"in-kernel amdgpu (Linux {platform.release()})"),
        (None, False, None, "could not be checked (hidden sysfs)"),
    ],
)
def test_linux_checks_the_loaded_driver_release_and_only_warns(version, module_dir_exists, status, text):
    ok, info = _linux_amd_driver_status(version, module_dir_exists)

    assert ok is status
    assert text in info


@pytest.mark.parametrize(
    "failure",
    [OSError("not found"), subprocess.TimeoutExpired("powershell", 10)],
)
def test_unavailable_windows_query_reports_an_unverified_driver(monkeypatch, amd_driver_environment, failure):
    monkeypatch.setattr(system_requirements.sys, "platform", "win32")
    monkeypatch.setattr(system_requirements.subprocess, "run", Mock(side_effect=failure))

    ok, info = system_requirements.check_gpu_driver_version()

    assert ok is None
    assert "could not be checked" in info


def test_a_current_driver_does_not_hide_a_missing_gpu(monkeypatch, amd_driver_environment):
    monkeypatch.setattr(system_requirements, "_check_amd_driver_version", lambda: (True, "current driver"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ok, info = system_requirements.check_gpu_driver_version()

    assert ok is False
    assert "no AMD GPU is available" in info
