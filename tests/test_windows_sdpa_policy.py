from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from jasna.mosaic import windows_sdpa_policy as policy


def fake_torch(monkeypatch, *, arch="gfx1100", hip="7.15.26333",
               version="2.12.0+rocm10.0.0", platform="win32"):
    monkeypatch.setattr(policy.sys, "platform", platform)
    monkeypatch.delenv(policy.POLICY_ENV, raising=False)
    flags = dict(flash=True, mem_efficient=True, cudnn=True, math=True)
    calls = []
    backend = SimpleNamespace()
    for name in flags:
        setattr(backend, name + "_sdp_enabled", lambda name=name: flags[name])
        def setter(enabled, name=name):
            calls.append((name, enabled))
            flags[name] = enabled
        setattr(backend, "enable_" + name + "_sdp", setter)
    torch = SimpleNamespace(
        __version__=version, version=SimpleNamespace(hip=hip),
        backends=SimpleNamespace(cuda=backend),
        cuda=SimpleNamespace(get_device_properties=Mock(
            return_value=SimpleNamespace(gcnArchName=arch))),
    )
    identity = Mock(return_value=dict(runtime_version=71526333, torch_hip="7.15.26333",
                                     runtime_dll_sha256=policy._HIP_SHA256))
    monkeypatch.setattr(policy, "_runtime_identity", identity)
    return torch, flags, calls, identity


@pytest.mark.parametrize("fp16", [True, False])
def test_exact_verified_identity_selects_gpu_math_independently_of_fp16(monkeypatch, fp16):
    torch, flags, calls, identity = fake_torch(monkeypatch, arch="gfx1100:sramecc-:xnack-")
    record = policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=fp16, torch_module=torch)
    assert record["resolved"] == "math"
    assert record["fp16"] is fp16
    assert flags == dict(flash=False, mem_efficient=False, cudnn=False, math=True)
    assert len(calls) == 4
    identity.assert_called_once()


@pytest.mark.parametrize("platform,device,hip", [
    ("linux", "cuda", "7.15.26333"), ("win32", "cpu", "7.15.26333"),
    ("win32", "cuda", None), ("darwin", "mps", None),
])
def test_other_platforms_and_cpu_never_probe_or_change_backends(monkeypatch, platform, device, hip):
    torch, flags, calls, identity = fake_torch(monkeypatch, platform=platform, hip=hip)
    monkeypatch.setenv(policy.POLICY_ENV, "math")
    record = policy.configure_windows_amd_sdpa(SimpleNamespace(type=device), fp16=True, torch_module=torch)
    assert record["resolved"] == "unchanged"
    assert calls == []
    identity.assert_not_called()
    torch.cuda.get_device_properties.assert_not_called()


@pytest.mark.parametrize("override", [dict(version="2.9.1+rocm7.2.1"),
                                      dict(hip="7.16.26354"), dict(arch="gfx1201")])
def test_auto_keeps_unknown_profiles_unchanged_and_math_rejects(monkeypatch, override):
    torch, flags, calls, identity = fake_torch(monkeypatch, **override)
    assert policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=True,
                                            torch_module=torch)["resolved"] == "unchanged"
    monkeypatch.setenv(policy.POLICY_ENV, "math")
    with pytest.raises(RuntimeError, match="profile not verified"):
        policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=True, torch_module=torch)
    assert calls == []
    identity.assert_not_called()


@pytest.mark.parametrize("key,value", [("runtime_version", 71626354),
    ("runtime_dll_sha256", "0" * 64), ("torch_hip", "7.2.53211")])
def test_actual_runtime_identity_is_required(monkeypatch, key, value):
    torch, flags, calls, identity = fake_torch(monkeypatch)
    identity.return_value[key] = value
    result = policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=True, torch_module=torch)
    assert result["resolved"] == "unchanged"
    assert calls == []
    monkeypatch.setenv(policy.POLICY_ENV, "math")
    with pytest.raises(RuntimeError, match="binary identity"):
        policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=True, torch_module=torch)


def test_explicit_default_does_not_reset_external_bootstrap(monkeypatch):
    torch, flags, calls, identity = fake_torch(monkeypatch)
    flags.update(flash=False, mem_efficient=False, cudnn=False)
    monkeypatch.setenv(policy.POLICY_ENV, "default")
    result = policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=False, torch_module=torch)
    assert result["flags"] == flags
    assert result["resolved"] == "unchanged"
    assert calls == []
    identity.assert_not_called()


def test_invalid_policy_and_failed_flag_readback_refuse_forward(monkeypatch):
    torch, flags, calls, identity = fake_torch(monkeypatch)
    monkeypatch.setenv(policy.POLICY_ENV, "typo")
    with pytest.raises(ValueError, match="auto, math or default"):
        policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=True, torch_module=torch)
    monkeypatch.setenv(policy.POLICY_ENV, "auto")
    torch.backends.cuda.enable_flash_sdp = lambda _: None
    with pytest.raises(RuntimeError, match="readback failed"):
        policy.configure_windows_amd_sdpa(SimpleNamespace(type="cuda"), fp16=True, torch_module=torch)


@pytest.mark.parametrize("fp16", [True, False])
def test_runner_applies_policy_before_wrapper_constructor_and_keeps_precision(monkeypatch, fp16):
    import torch
    from jasna.mosaic import rfdetr_torch_runner as runner

    order = []
    monkeypatch.setattr(policy, "configure_windows_amd_sdpa",
                        lambda device, **kw: order.append(("policy", kw["fp16"])) or {"resolved": "math"})
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"model": {"class_embed.weight": torch.zeros(2, 4)}})
    core = SimpleNamespace(to=lambda _: core, eval=lambda: core)
    def wrapper(**kw):
        assert order == [("policy", fp16)]
        order.append(("constructor", fp16))
        return SimpleNamespace(model=SimpleNamespace(model=core))
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSegMedium=wrapper))
    result = runner.RfDetrTorchRunner(Path("synthetic.pt"), [(4, 3, 128, 128)], torch.device("cpu"),
                                    fp16=fp16, resolution=128, variant="medium")
    assert result.fp16 is fp16
    assert result.sdpa_policy["resolved"] == "math"
    assert order == [("policy", fp16), ("constructor", fp16)]
