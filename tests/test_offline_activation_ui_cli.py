import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def lease_api(monkeypatch, gui_language):
    """Activation with the License code alone; offline on a certified TPM, else checked online while supporter models run."""
    pytest.importorskip("jasna.protection.license_store")
    from jasna.protection import lease, license_store

    gui_language("en")
    monkeypatch.setattr(license_store, "is_licensed", lambda: False)
    activate = Mock(return_value=lease.Activated(["ltx-2.5-v2"], False, False))
    monkeypatch.setattr(lease, "activate_license_code", activate)
    return activate


@pytest.mark.parametrize("arguments", [
    ["--machine-key"],
    ["--activation-key", "key"],
    ["--cloud-workspace"],
    ["--license-email", "buyer@example.com", "--license-key", "key"],
])
def test_cli_rejects_retired_options(monkeypatch, arguments):
    from jasna.main import main

    monkeypatch.setattr(sys, "argv", ["jasna", *arguments])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize(("offline", "needs_group", "expected"), [
    (False, False, "checks your license online"), (True, False, "works offline on this computer"),
    (False, True, "sudo usermod -aG tss $USER"),
])
def test_cli_license_code_activates_and_exits_before_gpu_setup(monkeypatch, lease_api, capsys, offline, needs_group, expected):
    from jasna import main
    from jasna.protection import lease

    lease_api.return_value = lease.Activated(["ltx-2.5-v2"], offline, needs_group)
    code = "JL1-" + "A" * 43
    monkeypatch.setattr(sys, "argv", ["jasna", "--license-code", code])
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    monkeypatch.setattr(main, "_configure_runtime", lambda args: pytest.fail("Runtime setup ran"))
    main.main()
    lease_api.assert_called_once_with(code)
    output = capsys.readouterr()
    assert expected in output.out
    assert code not in output.out + output.err


def test_cli_license_code_activates_before_processing(monkeypatch, lease_api, tmp_path):
    from jasna import main

    photo = tmp_path / "photo.png"
    photo.touch()
    monkeypatch.setattr(sys, "argv", ["jasna", "--license-code", "JL1-code", "--input", str(photo), "--output", "out.png"])

    def stop_before_gpu(args):
        lease_api.assert_called_once_with("JL1-code")
        raise RuntimeError("activated before processing")

    monkeypatch.setattr(main, "_check_system", stop_before_gpu)
    with pytest.raises(RuntimeError, match="activated before processing"):
        main.main()


def test_cli_license_code_failure_is_safe_and_stops_processing(monkeypatch, lease_api, capsys):
    from jasna import main
    from jasna.protection.activation import ActivationServiceError

    lease_api.side_effect = ActivationServiceError("devices_full")
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    code = "JL1-" + "A" * 43
    monkeypatch.setattr(sys, "argv", ["jasna", "--license-code", code])
    with pytest.raises(SystemExit) as error:
        main.main()
    output = capsys.readouterr()
    assert error.value.code == 1
    assert "already in use on three devices" in output.err
    assert code not in output.out + output.err


def test_activation_worker_runs_io_and_errors_off_the_main_thread(lease_api):
    from jasna.gui.activation_worker import start_lease_activation

    result = []
    error = OSError("config volume unavailable")
    lease_api.side_effect = error
    thread = start_lease_activation("JL1-code", lambda value, exc: result.append((value, exc, threading.get_ident())))
    thread.join(timeout=3)
    assert result == [(None, error, thread.ident)]
    assert thread.ident != threading.get_ident()


def test_gui_activation_keeps_exact_ltx_entitlements(lease_api, monkeypatch, tmp_path):
    from jasna.gui import ltx_models
    from jasna.ltx import model_files
    from jasna.protection import lease, license_store

    for name in model_files.bundle_names("distilled", fast=False):
        (tmp_path / (name + ".enc")).touch()
    monkeypatch.setattr(license_store, "is_licensed", lambda: True)
    monkeypatch.setattr(lease, "saved_models", lambda: ["ltx-2.5-v2"])
    assert not model_files.model_installed(tmp_path, "distilled", fast=False)
    assert not ltx_models.license_missing(tmp_path, "distilled", False)
    monkeypatch.setattr(lease, "saved_models", lambda: ["unet-4x"])
    assert ltx_models.license_missing(tmp_path, "distilled", False)


@pytest.fixture
def controlled_activation_workers(monkeypatch):
    pytest.importorskip("jasna.protection.license_store")
    from jasna.gui import activation_worker

    pending = SimpleNamespace(info=[], activation=[])
    monkeypatch.setattr(activation_worker, "start_license_info", lambda done: pending.info.append(done))
    monkeypatch.setattr(activation_worker, "start_lease_activation", lambda code, done: pending.activation.append((code, done)))
    return pending


def test_dialog_locks_input_while_activating(tk_root, controlled_activation_workers):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t
    from jasna.protection.activation import ActivationServiceError

    pending = controlled_activation_workers
    dialog = LicenseDialog(tk_root, on_activated=lambda: pytest.fail("failed activation reported success"))
    assert dialog._activate_button.cget("state") == "disabled"
    pending.info.pop()((False, None), None)
    dialog._main_thread._run_pending()
    dialog._code.insert(0, "JL1-code")
    dialog._activate_button.invoke()
    code, done = pending.activation.pop()
    assert code == "JL1-code"
    assert dialog._code.cget("state") == "disabled"
    dialog._activate_button.invoke()
    assert pending.activation == []
    done(None, ActivationServiceError("slots_full"))
    dialog._main_thread._run_pending()
    assert dialog._activate_button.cget("state") == "normal"
    assert dialog._code.get() == "JL1-code"
    assert dialog._status.cget("text") == t("license_service_slots_full")


@pytest.mark.parametrize("stage", ["info", "activation"])
def test_dialog_ignores_worker_result_after_close(tk_root, controlled_activation_workers, caplog, stage):
    from jasna.gui.components import LicenseDialog
    from jasna.protection import lease

    caplog.set_level("DEBUG", logger="jasna.gui.components")
    pending = controlled_activation_workers
    activated = Mock()
    dialog = LicenseDialog(tk_root, on_activated=activated)
    done = pending.info.pop()
    if stage == "activation":
        done((False, None), None)
        dialog._main_thread._run_pending()
        dialog._activate_button.invoke()
        _, done = pending.activation.pop()
    dialog.destroy()
    done(lease.Activated([], False, False) if stage == "activation" else (False, None), None)
    tk_root.update()
    activated.assert_not_called()
    assert "closed activation dialog" in caplog.text


@pytest.mark.parametrize(("offline", "needs_group", "message"), [
    (False, False, "license_lease_active"), (True, False, "license_offline_active"), (False, True, "license_lease_active"),
])
def test_lease_dialog_activates_with_the_license_code_only(tk_root, lease_api, settle_activation, offline, needs_group, message):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t
    from jasna.protection import lease

    lease_api.return_value = lease.Activated(["ltx-2.5-v2"], offline, needs_group)

    callback_threads = []
    dialog = LicenseDialog(tk_root, on_activated=lambda: callback_threads.append(threading.get_ident()))
    settle_activation(dialog)
    assert dialog._activate_button.cget("state") == "normal"
    dialog._code.insert(0, "  JL1-code  ")
    dialog._activate_button.invoke()
    settle_activation(dialog)
    lease_api.assert_called_once_with("JL1-code")
    assert dialog._status.cget("text") == t(message) + ("\n" + t("license_tss_hint") if needs_group else "")
    assert dialog._code.get() == ""
    assert callback_threads == [threading.get_ident()]


@pytest.mark.parametrize(("code", "message"), [("network_error", "license_lease_network_error"),
                                                ("license_not_found", "license_service_license_not_found")])
def test_lease_dialog_explains_a_failed_activation(tk_root, lease_api, settle_activation, code, message):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t
    from jasna.protection.activation import ActivationServiceError

    lease_api.side_effect = ActivationServiceError(code)
    dialog = LicenseDialog(tk_root, on_activated=lambda: pytest.fail("failed activation reported success"))
    settle_activation(dialog)
    dialog._code.insert(0, "JL1-code")
    dialog._activate_button.invoke()
    settle_activation(dialog)
    assert dialog._status.cget("text") == t(message)


def test_lease_dialog_names_an_offline_activation_when_reopened(tk_root, lease_api, settle_activation, monkeypatch):
    from jasna.gui import activation_worker
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t
    from jasna.protection import license_store, offline

    monkeypatch.setattr(license_store, "is_licensed", lambda: True)
    monkeypatch.setattr(offline, "saved_grant", lambda: ("JT1-" + "a" * 64, b"", {}))
    assert activation_worker.read_license_info() == (True, "JT1-" + "a" * 64)
    dialog = LicenseDialog(tk_root, on_activated=lambda: None)
    settle_activation(dialog)
    assert dialog._status.cget("text") == t("license_offline_active")
