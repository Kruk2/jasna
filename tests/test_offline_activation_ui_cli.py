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
    from jasna import protection
    from jasna.protection import license_store, signing

    machine = SimpleNamespace(machine_key=Mock(return_value="JM1-" + "1" * 64))
    monkeypatch.setattr(protection, "machine", machine, raising=False)
    monkeypatch.setattr(license_store, "load_license", lambda: None)
    monkeypatch.setattr(license_store, "is_licensed", lambda: False)
    monkeypatch.setattr(license_store, "set_license", Mock())
    monkeypatch.setattr(signing, "decode_license", lambda key: (b"proof", {}))
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


def test_cli_activation_only_and_processing_share_the_early_save(monkeypatch, protection_api):
    from jasna import main

    _, store = protection_api
    monkeypatch.setattr(sys, "argv", ["jasna", "--activation-key", "  activation  "])
    monkeypatch.setattr(main, "_check_system", lambda args: pytest.fail("GPU check ran"))
    main.main()
    store.set_license.assert_called_once_with("activation")
    store.set_license.reset_mock()

    monkeypatch.setattr(sys, "argv", [
        "jasna", "--activation-key", "activation", "--input", "photo.png", "--output", "out.png",
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


def test_dialog_copies_readonly_machine_key_and_cloud_is_explicit(root, protection_api):
    from jasna.gui.components import LicenseDialog

    machine, _ = protection_api
    dialog = LicenseDialog(root, on_activated=lambda: None)
    _wait(root, lambda: dialog._machine.get() != "")
    assert dialog._machine.cget("state") == "readonly"
    dialog._machine.insert(0, "altered")
    dialog._copy.invoke()
    assert root.clipboard_get() == "JM1-" + "1" * 64
    machine.machine_key.assert_called_once_with(cloud_workspace=False)

    machine.machine_key.return_value = "JW1-" + "2" * 64
    dialog._cloud_check.toggle()
    _wait(root, lambda: dialog._machine.get().startswith("JW1-"))
    assert machine.machine_key.call_args.kwargs == {"cloud_workspace": True}


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


def test_saved_cloud_activation_does_not_create_a_seed(root, monkeypatch, protection_api):
    from jasna.gui.components import LicenseDialog
    from jasna.protection import signing

    machine, store = protection_api
    monkeypatch.setattr(store, "load_license", lambda: "stored-cloud-activation")
    monkeypatch.setattr(signing, "decode_license", lambda key: (b"proof", {}))
    monkeypatch.setattr(signing, "license_identity", lambda proof: "JW1-" + "2" * 64, raising=False)
    dialog = LicenseDialog(root, on_activated=lambda: None)
    _wait(root, lambda: dialog._machine.get().startswith("JW1-"))
    assert dialog._cloud_workspace.get() is True
    assert dialog._key.get() == "stored-cloud-activation"
    machine.machine_key.assert_not_called()


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


@pytest.fixture
def controlled_activation_workers(monkeypatch):
    from jasna.gui import activation_worker

    pending = SimpleNamespace(info=[], machine=[], activation=[])
    monkeypatch.setattr(activation_worker, "start_license_info", lambda done: pending.info.append(done))
    monkeypatch.setattr(activation_worker, "start_machine_key", lambda cloud, done: pending.machine.append((cloud, done)))
    monkeypatch.setattr(activation_worker, "start_activation", lambda key, done: pending.activation.append((key, done)))
    return pending


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

    cloud, done = pending.machine.pop()
    assert cloud is False
    done("JM1-" + "1" * 64, None)
    _wait(root, lambda: dialog._activate_button.cget("state") == "normal")
    dialog._activate_button.invoke()
    assert dialog._cloud_check.cget("state") == "disabled"
    dialog._cloud_check.toggle()
    assert dialog._cloud_workspace.get() is False
    assert pending.machine == []

    key, done = pending.activation.pop()
    assert key == "activation"
    done("JW1-" + "2" * 64 if error is None else None, error)
    _wait(root, lambda: dialog._activate_button.cget("state") == "normal")
    assert dialog._cloud_check.cget("state") == "normal"
    if error is None:
        assert dialog._cloud_workspace.get() is True
        assert dialog._machine.get() == "JW1-" + "2" * 64
        activated.assert_called_once()
    else:
        assert dialog._cloud_workspace.get() is False
        assert dialog._machine.get() == "JM1-" + "1" * 64
        assert "config volume unavailable" in dialog._status.cget("text")
        activated.assert_not_called()


def test_dialog_restores_controls_after_machine_read_failure(root, controlled_activation_workers):
    from jasna.gui.components import LicenseDialog

    pending = controlled_activation_workers
    dialog = LicenseDialog(root, on_activated=lambda: None)
    pending.info.pop()((None, False, None), None)
    _wait(root, lambda: len(pending.machine) == 1)
    _, done = pending.machine.pop()
    done(None, OSError("system identity unavailable"))
    _wait(root, lambda: dialog._cloud_check.cget("state") == "normal")
    assert dialog._activate_button.cget("state") == "normal"
    assert dialog._copy.cget("state") == "disabled"
    assert "system identity unavailable" in dialog._status.cget("text")
    dialog._cloud_check.toggle()
    assert pending.machine[0][0] is True
    assert dialog._activate_button.cget("state") == "disabled"


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
