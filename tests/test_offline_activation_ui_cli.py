import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def protection_api(monkeypatch):
    pytest.importorskip("jasna.protection.license_store")
    from jasna import protection
    from jasna.gui.locales import get_locale
    from jasna.protection import license_store, signing

    machine = SimpleNamespace(machine_key=Mock(return_value="JM1-" + "1" * 64))
    monkeypatch.setattr(get_locale(), "_current_lang", "en")
    monkeypatch.setattr(protection, "machine", machine, raising=False)
    monkeypatch.setattr(license_store, "load_license", lambda: None)
    monkeypatch.setattr(license_store, "is_licensed", lambda: False)
    monkeypatch.setattr(license_store, "set_license", Mock())
    monkeypatch.setattr(signing, "decode_license", lambda key: (b"proof", {}))
    monkeypatch.setattr(license_store, "_decode", lambda key: signing.decode_license(key))
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JM1-" + "1" * 64, raising=False)
    return machine, license_store


@pytest.mark.parametrize("cloud", [False, True])
def test_cli_machine_key_exits_before_runtime_setup(monkeypatch, protection_api, capsys, cloud):
    from jasna import main

    machine, _ = protection_api
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    monkeypatch.setattr(main, "_configure_runtime", lambda args: pytest.fail("Runtime setup ran"))
    monkeypatch.setattr(sys, "argv", ["jasna", "--machine-key"] + (["--cloud-workspace"] if cloud else []))
    main.main()
    machine.machine_key.assert_called_once_with(cloud_workspace=cloud)
    assert capsys.readouterr().out.strip() == "JM1-" + "1" * 64


def test_cli_activation_only_and_processing_share_the_early_save(monkeypatch, protection_api, tmp_path):
    from jasna import main

    _, store = protection_api
    monkeypatch.setattr(sys, "argv", ["jasna", "--activation-key", "  activation  "])
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    main.main()
    store.set_license.assert_called_once_with("activation")
    store.set_license.reset_mock()

    photo = tmp_path / "photo.png"
    photo.touch()
    monkeypatch.setattr(sys, "argv", [
        "jasna", "--activation-key", "activation", "--input", str(photo), "--output", "out.png",
    ])

    def stop_before_gpu(args):
        store.set_license.assert_called_once_with("activation")
        raise RuntimeError("saved before image processing")

    monkeypatch.setattr(main, "_check_system", stop_before_gpu)
    with pytest.raises(RuntimeError, match="saved before image processing"):
        main.main()


@pytest.mark.parametrize("arguments", [
    ["--cloud-workspace"],
    ["--machine-key", "--activation-key", "key"],
    ["--machine-key", "--license-code", "code"],
    ["--license-code", "code", "--activation-key", "key"],
    ["--license-email", "buyer@example.com", "--license-key", "key"],
])
def test_cli_rejects_conflicting_or_retired_options(monkeypatch, arguments):
    from jasna.main import main

    monkeypatch.setattr(sys, "argv", ["jasna", *arguments])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_cli_rejection_keeps_machine_diagnostic_without_printing_key(monkeypatch, protection_api, capsys):
    from jasna.main import main
    from jasna.protection import LicenseError

    _, store = protection_api
    store.set_license.side_effect = LicenseError("Activation belongs to another Machine key.")
    monkeypatch.setattr(sys, "argv", ["jasna", "--activation-key", "private-activation"])
    with pytest.raises(SystemExit) as exc:
        main()
    output = capsys.readouterr()
    assert exc.value.code == 1
    assert "another Machine key" in output.err
    assert "private-activation" not in output.err + output.out


@pytest.mark.skipif(sys.platform != "linux", reason="System identity is currently available on Linux")
def test_machine_key_command_imports_no_torch_cuda_or_gui(tmp_path):
    pytest.importorskip("jasna.protection.license_store")
    script = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name.startswith('jasna.gui') or name == 'customtkinter':
        raise AssertionError('Heavy startup import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = ['jasna', '--machine-key']
runpy.run_module('jasna', run_name='__main__')
"""
    environment = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path)}
    environment.pop("JASNA_MAIN_PID", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("JM1-")
    assert len(result.stdout.strip()) == 68
    assert not (tmp_path / "jasna" / "workspace.id").exists()


@pytest.fixture
def root():
    import customtkinter as ctk

    window = ctk.CTk()
    yield window
    window.destroy()


def _wait(root, ready):
    deadline = time.monotonic() + 3
    while not ready() and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    assert ready()


def test_dialog_copies_only_readonly_system_machine_key(root, protection_api):
    from jasna.gui.components import LicenseDialog

    machine, _ = protection_api
    dialog = LicenseDialog(root, on_activated=lambda: None)
    _wait(root, lambda: dialog._machine.get() != "")
    assert dialog._machine.cget("state") == "readonly"
    dialog._machine.insert(0, "altered")
    dialog._copy.invoke()
    assert root.clipboard_get() == "JM1-" + "1" * 64
    machine.machine_key.assert_called_once_with(cloud_workspace=False)

    assert "_cloud_check" not in vars(dialog)
    assert "_cloud_workspace" not in vars(dialog)


def test_dialog_logs_machine_identity_error(root, protection_api, caplog):
    from jasna.gui.components import LicenseDialog

    machine, _ = protection_api
    machine.machine_key.side_effect = RuntimeError("machine identity is unavailable or unsafe")
    dialog = LicenseDialog(root, on_activated=lambda: None)
    _wait(root, lambda: "unavailable or unsafe" in dialog._status.cget("text"))
    assert any(
        record.levelname == "WARNING" and "unavailable or unsafe" in record.getMessage()
        for record in caplog.records
    )


def test_dialog_saves_only_activation_key_and_calls_ui_on_main_thread(root, protection_api):
    from jasna.gui.components import LicenseDialog

    _, store = protection_api
    callback_threads = []
    dialog = LicenseDialog(root, on_activated=lambda: callback_threads.append(threading.get_ident()))
    _wait(root, lambda: dialog._activate_button.cget("state") == "normal")
    dialog._key.insert(0, "  activation  ")
    dialog._activate_button.invoke()
    _wait(root, lambda: callback_threads != [])
    store.set_license.assert_called_once_with("activation")
    assert callback_threads == [threading.get_ident()]


@pytest.mark.parametrize("language,expected_url", [
    ("en", "https://jasna-activation.kruk2.workers.dev"),
    ("zh", "https://47.238.135.110"),
    ("ja", "https://jasna-activation.kruk2.workers.dev"),
])
def test_dialog_opens_activation_page_without_sending_credentials(root, protection_api, monkeypatch, language, expected_url):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import get_locale

    monkeypatch.setattr(get_locale(), "_current_lang", language)
    opened = Mock()
    monkeypatch.setattr("jasna.gui.components.webbrowser.open", opened)
    dialog = LicenseDialog(root, on_activated=lambda: None)
    dialog._key.insert(0, "private-activation")
    dialog._activation_page_button.invoke()
    opened.assert_called_once_with(expected_url)


def test_saved_cloud_activation_does_not_switch_gui_identity(root, monkeypatch, protection_api):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t
    from jasna.protection import signing

    machine, store = protection_api
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    monkeypatch.setattr(signing, "decode_license", lambda key: (b"proof", {}))
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64, raising=False)
    dialog = LicenseDialog(root, on_activated=lambda: None)
    _wait(root, lambda: dialog._machine.get().startswith("JM1-"))
    assert dialog._status.cget("text") == t("license_cloud_cli_only")
    assert dialog._key.get() == ""
    machine.machine_key.assert_called_once_with(cloud_workspace=False)


def test_activation_worker_runs_io_and_errors_off_the_main_thread(protection_api):
    from jasna.gui.activation_worker import start_activation

    _, store = protection_api
    result = []
    error = OSError("config volume unavailable")
    store.set_license.side_effect = error
    thread = start_activation("activation", lambda value, exc: result.append((value, exc, threading.get_ident())))
    thread.join(timeout=3)
    assert result == [(None, error, thread.ident)]
    assert thread.ident != threading.get_ident()


def test_gui_status_does_not_offer_a_saved_cloud_activation(protection_api, monkeypatch):
    from jasna.gui.activation_worker import start_license_info
    from jasna.protection import signing

    machine, store = protection_api
    identity = "JW1-" + "2" * 64
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    approved = Mock(return_value=True)
    monkeypatch.setattr(store, "is_licensed", approved)
    monkeypatch.setattr(signing, "license_identity", lambda proof: identity)
    result = []
    thread = start_license_info(lambda value, error: result.append((value, error)))
    thread.join(timeout=3)
    assert result == [(("stored-cloud-activation", False, identity), None)]
    approved.assert_not_called()
    machine.machine_key.assert_not_called()
    store.set_license.assert_not_called()


@pytest.mark.parametrize("frozen", [False, True])
def test_saved_cloud_activation_blocks_encrypted_ltx_gui_paths(protection_api, monkeypatch, tmp_path, frozen):
    from dataclasses import replace
    from jasna.gui import ltx_models, video_session
    from jasna.gui.locales import t
    from jasna.gui.models import AppSettings
    from jasna.ltx import model_files
    from jasna.protection import protected_tensors, signing
    from jasna.mosaic import detection_registry

    _, store = protection_api
    directory = tmp_path / "ltx-restore"
    directory.mkdir()
    monkeypatch.setattr(model_files, "LTX_DOWNLOADS", {
        name: replace(entry, size_bytes=0) for name, entry in model_files.LTX_DOWNLOADS.items()
    })
    for name in model_files.bundle_names("distilled", fast=False):
        (directory / (name + ".enc")).touch()
        if frozen:
            (directory / name).touch()
    monkeypatch.setattr(model_files, "is_frozen", lambda: frozen)
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    monkeypatch.setattr(store, "is_licensed", lambda: True)
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64)
    accepted = Mock()
    monkeypatch.setattr(protected_tensors, "check_ltx_license", accepted)
    model_files.LtxModelFiles.from_dir(directory, "distilled", fast=False)
    accepted.assert_called_once()
    assert ltx_models.license_missing(directory, "distilled", False)

    built = Mock()
    monkeypatch.setattr("jasna.engine_paths.model_weights_dir", lambda: tmp_path)
    monkeypatch.setattr(detection_registry, "require_detection_model_weights", lambda name: tmp_path / "det.onnx")
    monkeypatch.setattr("jasna.session_factory.build_restoration_session", built)
    settings = replace(AppSettings(), restoration_model="ltx", ltx_trial=False)
    with pytest.raises(ValueError, match=t("license_cloud_cli_only")):
        video_session.build_video_session(settings, log=lambda message: None)
    built.assert_not_called()
    store.set_license.assert_not_called()


@pytest.mark.parametrize("mode", ["plaintext", "trial", "free"])
def test_gui_cloud_restriction_keeps_plaintext_trial_and_free_paths(protection_api, monkeypatch, tmp_path, mode):
    from dataclasses import replace
    from jasna.gui import ltx_models, video_session
    from jasna.gui.models import AppSettings
    from jasna.ltx import model_files
    from jasna.protection import signing
    from jasna.mosaic import detection_registry
    from jasna.session_config import LTX_DEFAULT_MODEL

    _, store = protection_api
    directory = tmp_path / "ltx-restore"
    directory.mkdir()
    for name in model_files.bundle_names(LTX_DEFAULT_MODEL, fast=False):
        (directory / (name + ".enc")).touch()
        if mode == "plaintext":
            (directory / name).touch()
    monkeypatch.setattr(model_files, "is_frozen", lambda: False)
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    monkeypatch.setattr(store, "is_licensed", lambda: True)
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64)
    if mode == "plaintext":
        assert not ltx_models.license_missing(directory, LTX_DEFAULT_MODEL, False)
    built = Mock()
    monkeypatch.setattr("jasna.engine_paths.model_weights_dir", lambda: tmp_path)
    monkeypatch.setattr(detection_registry, "require_detection_model_weights", lambda name: tmp_path / "det.onnx")
    monkeypatch.setattr("jasna.session_factory.build_restoration_session", built)
    settings = replace(AppSettings(), restoration_model="basicvsrpp" if mode == "free" else "ltx", ltx_trial=mode == "trial")
    assert video_session.build_video_session(settings, log=lambda message: None) is built.return_value
    built.assert_called_once()
    store.set_license.assert_not_called()


def test_gui_standard_machine_activation_keeps_exact_ltx_entitlements(protection_api, monkeypatch, tmp_path):
    from jasna.gui import ltx_models
    from jasna.ltx import model_files
    from jasna.protection import ModelNotLicensedError, protected_tensors

    _, store = protection_api
    for name in model_files.bundle_names("distilled", fast=False):
        (tmp_path / (name + ".enc")).touch()
    monkeypatch.setattr(store, "load_license", lambda: "stored-standard-activation")
    monkeypatch.setattr(store, "is_licensed", lambda: True)
    accepted = Mock()
    monkeypatch.setattr(protected_tensors, "check_ltx_license", accepted)
    assert not model_files.model_installed(tmp_path, "distilled", fast=False)
    assert not ltx_models.license_missing(tmp_path, "distilled", False)
    accepted.assert_called_once()
    accepted.side_effect = ModelNotLicensedError("LTX not included")
    assert ltx_models.license_missing(tmp_path, "distilled", False)


@pytest.mark.parametrize("plaintext", [False, True])
def test_gui_saved_cloud_unet_grant_only_allows_plaintext_source(protection_api, monkeypatch, tmp_path, plaintext):
    from dataclasses import replace
    from jasna.gui import video_session
    from jasna.gui.locales import t
    from jasna.gui.models import AppSettings
    from jasna.protection import signing

    _, store = protection_api
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    monkeypatch.setattr(store, "is_licensed", lambda: True)
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64)
    monkeypatch.setattr("jasna.engine_paths.unet4x_plaintext_available", lambda: plaintext)
    monkeypatch.setattr("jasna.mosaic.detection_registry.require_detection_model_weights", lambda name: tmp_path / "det.onnx")
    built = Mock()
    monkeypatch.setattr("jasna.session_factory.build_restoration_session", built)
    settings = replace(AppSettings(), secondary_restoration="unet-4x")
    if plaintext:
        assert video_session.build_video_session(settings, log=lambda message: None) is built.return_value
        built.assert_called_once()
    else:
        with pytest.raises(ValueError, match=t("license_cloud_cli_only")):
            video_session.build_video_session(settings, log=lambda message: None)
        built.assert_not_called()


@pytest.mark.parametrize("plaintext,frozen", [(False, False), (True, False), (True, True)])
def test_gui_saved_cloud_sd_grant_only_allows_plaintext_source(protection_api, monkeypatch, tmp_path, plaintext, frozen):
    from jasna.gui import video_session
    from jasna.gui.locales import t
    from jasna.gui.models import AppSettings
    from jasna.protection import signing
    from jasna.restorer import sd15_inpaint_restorer
    from jasna.engine_paths import SD15_CKPT_PATH

    _, store = protection_api
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    monkeypatch.setattr(store, "is_licensed", lambda: True)
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64)
    monkeypatch.setattr("jasna.engine_paths.SD15_DIR", tmp_path)
    if plaintext:
        (tmp_path / SD15_CKPT_PATH.name).touch()
    monkeypatch.setattr(sd15_inpaint_restorer, "is_frozen", lambda: frozen)
    monkeypatch.setattr("jasna.restorer.sd15_download.bundle_present", lambda directory: True)
    built = Mock()
    detector = Mock()
    monkeypatch.setattr(sd15_inpaint_restorer, "Sd15InpaintRestorer", built)
    monkeypatch.setattr("jasna.session_factory.build_compiled_detection_model", detector)
    monkeypatch.setattr("jasna.mosaic.detection_registry.resolve_detection_model", lambda *args: ("rfdetr-v6", "det.onnx", None))
    if plaintext and not frozen:
        result = video_session.build_image_session(AppSettings(), log=None)
        assert result[:2] == (detector.return_value, built.return_value)
        built.assert_called_once()
    else:
        with pytest.raises(ValueError, match=t("license_cloud_cli_only")):
            video_session.build_image_session(AppSettings(), log=None)
        built.assert_not_called()
        detector.assert_not_called()


def test_gui_activation_rejects_cloud_key_before_save(protection_api, monkeypatch):
    from jasna.gui.activation_worker import start_activation
    from jasna.gui.locales import t
    from jasna.protection import signing

    _, store = protection_api
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64)
    result = []
    thread = start_activation("cloud-activation", lambda value, error: result.append((value, error)))
    thread.join(timeout=3)
    assert len(result) == 1 and result[0][0] is None
    assert str(result[0][1]) == t("license_cloud_cli_only")
    store.set_license.assert_not_called()


def test_gui_online_activation_rejects_cloud_identity_before_request(protection_api, monkeypatch):
    from jasna.gui.activation_worker import start_online_activation
    from jasna.gui.locales import t
    from jasna.protection import activation

    request = Mock()
    monkeypatch.setattr(activation, "activate_license_code", request)
    result = []
    thread = start_online_activation("code", "JW1-" + "2" * 64,
                                    lambda value, error: result.append((value, error)))
    thread.join(timeout=3)
    assert len(result) == 1 and result[0][0] is None
    assert str(result[0][1]) == t("license_cloud_cli_only")
    request.assert_not_called()


@pytest.fixture
def controlled_activation_workers(monkeypatch):
    pytest.importorskip("jasna.protection.license_store")
    from jasna.gui import activation_worker

    pending = SimpleNamespace(info=[], machine=[], activation=[], online=[], devices=[])
    monkeypatch.setattr(activation_worker, "start_license_info", lambda done: pending.info.append(done))
    monkeypatch.setattr(activation_worker, "start_machine_key", lambda done: pending.machine.append(done))
    monkeypatch.setattr(activation_worker, "start_activation", lambda key, done: pending.activation.append((key, done)))
    monkeypatch.setattr(activation_worker, "start_online_activation", lambda code, machine, done: pending.online.append((code, machine, done)))
    monkeypatch.setattr(activation_worker, "start_device_count", lambda key, done: pending.devices.append((key, done)))
    return pending


@pytest.mark.parametrize("licensed,cloud", [(False, False), (True, False), (True, True)])
def test_device_count_is_requested_only_when_opened_with_active_desktop_license(root, controlled_activation_workers, licensed, cloud):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t

    pending = controlled_activation_workers
    dialog = LicenseDialog(root, on_activated=lambda: None)
    assert pending.devices == []
    identity = ("JW1-" if cloud else "JM1-") + "1" * 64
    pending.info.pop()(("saved-grant", licensed, identity), None)
    _wait(root, lambda: len(pending.machine) == 1)
    if licensed and not cloud:
        key, done = pending.devices.pop()
        assert key == "saved-grant"
        assert dialog._devices.cget("text") == t("license_devices_checking")
        done((2, 3), None)
        _wait(root, lambda: dialog._devices.cget("text") == t("license_devices_count", used=2, limit=3))
        root.update_idletasks()
        assert dialog._devices.winfo_y() + dialog._devices.winfo_height() <= dialog.winfo_height()
        assert dialog._status.cget("text") == t("license_active")
        dialog._select_method(t("license_manual"))
        root.update()
        assert pending.devices == []
    else:
        assert pending.devices == []
        assert dialog._devices.winfo_manager() == ""


@pytest.mark.parametrize("closed", [False, True])
def test_device_count_failure_preserves_activation_and_ignores_closed_window(root, controlled_activation_workers, closed):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t

    pending = controlled_activation_workers
    dialog = LicenseDialog(root, on_activated=lambda: None)
    pending.info.pop()(("saved-grant", True, "JM1-" + "1" * 64), None)
    _wait(root, lambda: len(pending.devices) == 1)
    _, done = pending.devices.pop()
    if closed:
        dialog.destroy()
    done(None, OSError("private-request-body"))
    if closed:
        root.update()
    else:
        _wait(root, lambda: dialog._devices.cget("text") == t("license_devices_unavailable"))
        assert dialog._status.cget("text") == t("license_active")
        assert dialog._key.get() == "saved-grant"


@pytest.mark.parametrize("language,prefer_hong_kong", [("en", False), ("zh", True)])
def test_device_count_worker_uses_background_thread_and_locale(monkeypatch, protection_api, language, prefer_hong_kong):
    from jasna.gui.activation_worker import start_device_count
    from jasna.gui.locales import get_locale
    from jasna.protection import activation

    monkeypatch.setattr(get_locale(), "_current_lang", language)
    threads = []
    def count(key, *, prefer_hong_kong):
        threads.append(threading.get_ident())
        return (2, 3)
    lookup = Mock(side_effect=count)
    monkeypatch.setattr(activation, "get_device_count", lookup)
    result = []
    worker = start_device_count("saved-grant", lambda value, error: result.append((value, error)))
    worker.join(timeout=3)
    assert result == [((2, 3), None)]
    assert threads[0] != threading.get_ident()
    lookup.assert_called_once_with("saved-grant", prefer_hong_kong=prefer_hong_kong)


@pytest.mark.parametrize("already_activated", [False, True])
def test_activation_success_does_not_start_a_device_count_or_keep_a_stale_count(root, controlled_activation_workers, already_activated):
    from jasna.gui.components import LicenseDialog

    pending = controlled_activation_workers
    dialog = LicenseDialog(root, on_activated=lambda: None)
    identity = "JM1-" + "1" * 64
    pending.info.pop()(("saved-grant" if already_activated else None, already_activated, identity), None)
    _wait(root, lambda: len(pending.machine) == 1)
    pending.machine.pop()(identity, None)
    _wait(root, lambda: not dialog._busy)
    dialog._activation_done(identity, None)
    if already_activated:
        _, done = pending.devices.pop()
        done((2, 3), None)
        root.update()
    assert pending.devices == []
    assert dialog._devices.winfo_manager() == ""


@pytest.mark.parametrize("error", [None, OSError("config volume unavailable")])
def test_dialog_serializes_machine_read_and_activation(root, controlled_activation_workers, error):
    from jasna.gui.components import LicenseDialog

    pending = controlled_activation_workers
    activated = Mock()
    dialog = LicenseDialog(root, on_activated=activated)
    pending.info.pop()((None, False, None), None)
    _wait(root, lambda: len(pending.machine) == 1)
    dialog._key.insert(0, "activation")
    assert dialog._activate_button.cget("state") == "disabled"
    dialog._activate_button.invoke()
    assert pending.activation == []

    done = pending.machine.pop()
    done("JM1-" + "1" * 64, None)
    _wait(root, lambda: dialog._activate_button.cget("state") == "normal")
    dialog._activate_button.invoke()
    assert dialog._method.cget("state") == "disabled"
    assert pending.machine == []

    key, done = pending.activation.pop()
    assert key == "activation"
    done("JM1-" + "1" * 64 if error is None else None, error)
    _wait(root, lambda: dialog._activate_button.cget("state") == "normal")
    assert dialog._method.cget("state") == "normal"
    if error is None:
        assert dialog._machine.get() == "JM1-" + "1" * 64
        activated.assert_called_once()
    else:
        assert dialog._machine.get() == "JM1-" + "1" * 64
        assert "config volume unavailable" in dialog._status.cget("text")
        activated.assert_not_called()


def test_dialog_restores_controls_after_machine_read_failure(root, controlled_activation_workers):
    from jasna.gui.components import LicenseDialog

    pending = controlled_activation_workers
    dialog = LicenseDialog(root, on_activated=lambda: None)
    pending.info.pop()((None, False, None), None)
    _wait(root, lambda: len(pending.machine) == 1)
    done = pending.machine.pop()
    done(None, OSError("system identity unavailable"))
    _wait(root, lambda: dialog._method.cget("state") == "normal")
    assert dialog._activate_button.cget("state") == "normal"
    assert dialog._copy.cget("state") == "disabled"
    assert "system identity unavailable" in dialog._status.cget("text")
    assert pending.machine == []
    assert dialog._online_activate_button.cget("state") == "disabled"


def test_dialog_ignores_worker_result_after_close(root, controlled_activation_workers, caplog):
    from jasna.gui.components import LicenseDialog

    caplog.set_level("DEBUG", logger="jasna.gui.components")
    dialog = LicenseDialog(root, on_activated=lambda: None)
    done = controlled_activation_workers.info.pop()
    dialog.destroy()
    done((None, False, None), None)
    root.update()
    assert controlled_activation_workers.machine == []
    assert "closed activation dialog" in caplog.text


@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("language,prefer_hong_kong", [("en", False), ("zh", True)])
def test_cli_online_activation_exits_before_gpu_setup(monkeypatch, protection_api, capsys, cloud, language, prefer_hong_kong):
    from jasna import main
    from jasna.gui.locales import get_locale
    from jasna.protection import activation

    machine, _ = protection_api
    monkeypatch.setattr(get_locale(), "_current_lang", language)
    request = Mock(return_value="private-returned-key")
    monkeypatch.setattr(activation, "activate_license_code", request)
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    monkeypatch.setattr(main, "_configure_runtime", lambda args: pytest.fail("Runtime setup ran"))
    code = "JL1-" + "A" * 43
    monkeypatch.setattr(sys, "argv", ["jasna", "--license-code", code] + (["--cloud-workspace"] if cloud else []))
    main.main()
    machine.machine_key.assert_called_once_with(cloud_workspace=cloud)
    request.assert_called_once_with(code, "JM1-" + "1" * 64, prefer_hong_kong=prefer_hong_kong)
    output = capsys.readouterr()
    assert "Jasna activated" in output.out
    assert code not in output.out + output.err
    assert "private-returned-key" not in output.out + output.err


def test_cli_online_activation_is_saved_before_processing(monkeypatch, protection_api, tmp_path):
    from jasna import main
    from jasna.protection import activation

    request = Mock(return_value="activation")
    monkeypatch.setattr(activation, "activate_license_code", request)
    code = "JL1-" + "A" * 43
    photo = tmp_path / "photo.png"
    photo.touch()
    monkeypatch.setattr(sys, "argv", [
        "jasna", "--license-code", code, "--input", str(photo), "--output", "out.png",
    ])

    def stop_before_gpu(args):
        request.assert_called_once_with(code, "JM1-" + "1" * 64, prefer_hong_kong=False)
        raise RuntimeError("saved before processing")

    monkeypatch.setattr(main, "_check_system", stop_before_gpu)
    with pytest.raises(RuntimeError, match="saved before processing"):
        main.main()


def test_cli_online_failure_is_safe_and_stops_processing(monkeypatch, protection_api, capsys):
    from jasna import main
    from jasna.protection import activation

    monkeypatch.setattr(activation, "activate_license_code", Mock(side_effect=activation.ActivationServiceError("slots_full")))
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    code = "JL1-" + "A" * 43
    monkeypatch.setattr(sys, "argv", ["jasna", "--license-code", code])
    with pytest.raises(SystemExit) as error:
        main.main()
    output = capsys.readouterr()
    assert error.value.code == 1
    assert "Three identities" in output.err
    assert code not in output.out + output.err


def test_dialog_defaults_to_online_and_can_use_manual(root, protection_api):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t

    dialog = LicenseDialog(root, on_activated=lambda: None)
    _wait(root, lambda: dialog._online_activate_button.cget("state") == "normal")
    assert dialog._method.get() == t("license_online")
    assert dialog._online_frame.winfo_manager() == "pack"
    assert dialog._manual_frame.winfo_manager() == ""
    dialog._method.set(t("license_manual"))
    dialog._select_method(t("license_manual"))
    assert dialog._manual_frame.winfo_manager() == "pack"
    assert dialog._online_frame.winfo_manager() == ""
    dialog._key.insert(0, "activation")
    dialog._activate_button.invoke()
    _, store = protection_api
    _wait(root, lambda: store.set_license.called)
    store.set_license.assert_called_once_with("activation")


@pytest.mark.parametrize("error_code", [None, "slots_full", "cloud_not_enabled", "network_error"])
def test_dialog_online_activation_uses_displayed_identity_and_locks_requests(root, controlled_activation_workers, error_code):
    from jasna.gui.components import LicenseDialog
    from jasna.gui.locales import t
    from jasna.protection.activation import ActivationServiceError

    pending = controlled_activation_workers
    activated = Mock()
    dialog = LicenseDialog(root, on_activated=activated)
    pending.info.pop()((None, False, None), None)
    _wait(root, lambda: len(pending.machine) == 1)
    code = "JL1-" + "A" * 43
    dialog._code.insert(0, code)
    dialog._online_activate_button.invoke()
    assert pending.online == []
    done = pending.machine.pop()
    identity = "JM1-" + "1" * 64
    done(identity, None)
    _wait(root, lambda: dialog._online_activate_button.cget("state") == "normal")
    dialog._online_activate_button.invoke()
    sent_code, sent_machine, done = pending.online.pop()
    assert (sent_code, sent_machine) == (code, identity)
    assert dialog._code.cget("state") == "disabled"
    assert dialog._key.cget("state") == "disabled"
    assert dialog._method.cget("state") == "disabled"
    dialog._activate_button.invoke()
    dialog._online_activate_button.invoke()
    assert pending.activation == [] and pending.online == [] and pending.machine == []
    if error_code is None:
        done(("returned-activation", identity), None)
    else:
        done(None, ActivationServiceError(error_code))
    _wait(root, lambda: dialog._online_activate_button.cget("state") == "normal")
    if error_code is None:
        assert dialog._code.get() == ""
        assert dialog._key.get() == "returned-activation"
        activated.assert_called_once()
    else:
        assert dialog._code.get() == code
        assert dialog._status.cget("text") == t(f"license_service_{error_code}")
        activated.assert_not_called()


def test_dialog_ignores_online_result_after_close(root, controlled_activation_workers, caplog):
    from jasna.gui.components import LicenseDialog

    caplog.set_level("DEBUG", logger="jasna.gui.components")
    pending = controlled_activation_workers
    activated = Mock()
    dialog = LicenseDialog(root, on_activated=activated)
    pending.info.pop()((None, False, None), None)
    _wait(root, lambda: len(pending.machine) == 1)
    done = pending.machine.pop()
    identity = "JM1-" + "1" * 64
    done(identity, None)
    _wait(root, lambda: dialog._online_activate_button.cget("state") == "normal")
    dialog._code.insert(0, "JL1-" + "A" * 43)
    dialog._online_activate_button.invoke()
    _, _, done = pending.online.pop()
    dialog.destroy()
    done(("returned-activation", identity), None)
    root.update()
    activated.assert_not_called()
    assert "closed activation dialog" in caplog.text


@pytest.mark.parametrize("language,prefer_hong_kong", [("en", False), ("zh", True), ("ja", False)])
def test_online_worker_runs_network_off_tk_thread(protection_api, monkeypatch, language, prefer_hong_kong):
    from jasna.gui.activation_worker import start_online_activation
    from jasna.gui.locales import get_locale
    from jasna.protection import activation

    network_threads, results = [], []
    monkeypatch.setattr(get_locale(), "_current_lang", language)

    def request(code, machine_key, *, prefer_hong_kong):
        network_threads.append((threading.get_ident(), prefer_hong_kong))
        return "returned-activation"

    monkeypatch.setattr(activation, "activate_license_code", request)
    thread = start_online_activation("code", "JM1-" + "1" * 64,
        lambda value, error: results.append((value, error, threading.get_ident())))
    thread.join(timeout=3)
    assert network_threads == [(thread.ident, prefer_hong_kong)]
    assert thread.ident != threading.get_ident()
    assert results == [(("returned-activation", "JM1-" + "1" * 64), None, thread.ident)]
