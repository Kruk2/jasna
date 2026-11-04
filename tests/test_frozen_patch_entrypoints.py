import importlib
import importlib.util
import logging
import inspect
import linecache
import os
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import sysconfig
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from factories import cli_args, run_cli


def test_frozen_detection_import_does_not_load_late_runtime_helpers():
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", """
import sys
sys.path.insert(0, sys.argv[1])
before = set(sys.modules)
from jasna._frozen import is_frozen
assert not is_frozen()
loaded = set(sys.modules) - before
assert not loaded.intersection({'hashlib', 'sysconfig', 'linecache', 'pathlib', 'importlib.util'})
assert not any(name == 'torch' or name.startswith('torch.') for name in loaded)
assert not any(name == 'cryptography' or name.startswith('cryptography.') for name in loaded)
""", str(repository)],
        check=True, timeout=30,
    )


def test_frozen_config_scan_skips_source(caplog, monkeypatch, torch_config):
    import torch._inductor.config as inductor_config
    from torch._inductor import codecache
    import torch.jit
    from torch.utils import _config_module
    from jasna import _frozen

    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(_frozen, "register_ltx_sources", lambda root: None)
    monkeypatch.setattr(_frozen, "use_prebuilt_triton_modules", lambda directory: None)
    monkeypatch.setenv("TRITON_BACKENDS_IN_TREE", "1")
    torch_config(inductor_config, compile_threads=inductor_config.compile_threads)
    monkeypatch.setattr(codecache, "torch_key", codecache.torch_key_cache(lambda: b"source"))
    with (
        patch.object(torch.jit, "interface"),
        patch.object(
            _config_module,
            "get_assignments_with_compile_ignored_comments",
            side_effect=KeyError("torch._inductor.config_comms"),
        ) as scan,
        caplog.at_level(logging.DEBUG, logger="jasna._frozen"),
    ):
        _frozen.patch_frozen_torch()
        assert _config_module.get_assignments_with_compile_ignored_comments(object()) == set()

    scan.assert_not_called()
    assert not caplog.records


def test_source_build_keeps_config_scan(monkeypatch):
    import torch.jit
    from torch.utils import _config_module
    from jasna import _frozen

    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: False)
    scan = _config_module.get_assignments_with_compile_ignored_comments
    with patch.object(torch.jit, "interface"):
        _frozen.patch_frozen_torch()
    assert _config_module.get_assignments_with_compile_ignored_comments is scan


def test_importing_pipeline_does_not_patch_frozen_torch():
    orig = sys.modules.pop("jasna.pipeline", None)
    try:
        with patch("jasna._frozen.patch_frozen_torch") as spy:
            importlib.import_module("jasna.pipeline")
        spy.assert_not_called()
    finally:
        if orig is not None:
            sys.modules["jasna.pipeline"] = orig
        else:
            sys.modules.pop("jasna.pipeline", None)


@pytest.fixture
def root_log_level():
    """The CLI sets the root log level for its whole process; keep it from reaching later tests."""
    level = logging.getLogger().level
    yield
    logging.getLogger().setLevel(level)


def test_cli_main_patches_frozen_torch(tmp_path, root_log_level):
    with (patch("jasna._frozen.patch_frozen_torch") as spy,
          patch("jasna.accelerator.device_context", return_value=nullcontext())):
        run_cli(cli_args(tmp_path))
    spy.assert_called()


def test_gui_run_gui_patches_frozen_torch():
    from jasna.gui import app as gui_app

    class _Stop(Exception):
        pass

    with patch("jasna._frozen.patch_frozen_torch") as spy:
        with (patch.object(gui_app, "JasnaApp", side_effect=_Stop),
              patch.object(gui_app.scaling, "activate_static_dpi")):
            with pytest.raises(_Stop):
                gui_app.run_gui()
    spy.assert_called()


@pytest.mark.parametrize("module", (
    "jasna.ltx.transformer", "jasna.ltx.kernels",
    "jasna.models.ltx_vae.transformer", "jasna.models.ltx_vae.na_triton",
    "jasna.models.ltx_vae.video_encoder", "jasna.models.ltx_vae.ops",
    "jasna.models.unet4x",
))
def test_frozen_ltx_sources_support_inspect_without_a_jasna_directory(monkeypatch, tmp_path, module):
    from jasna import _frozen

    (tmp_path / "jasna").write_bytes(b"executable")
    monkeypatch.setattr(linecache, "cache", {})
    source = "def probe(value):\n    return value + 1\n"
    for name in _frozen.LTX_TRACED_MODULES:
        path = tmp_path / "python_source" / Path(*name.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        linecache.cache[str(path)] = (len(source), None, source.splitlines(keepends=True), str(path))
        path.write_text("def probe(value):\n    return 'replaced'\n")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(
        origin=str(tmp_path / (name.replace(".", "/") + ".py"))))
    _frozen.register_ltx_sources(tmp_path)
    filename = str(tmp_path / Path(*module.split(".")).with_suffix(".py"))
    namespace = {}
    exec(compile(source, filename, "exec"), namespace)
    assert inspect.getsource(namespace["probe"]) == source


def test_disabled_frozen_compilation_does_not_import_triton():
    code = """
import os
import sys
from jasna import _frozen
os.environ['TORCH_COMPILE_DISABLE'] = '1'
_frozen.is_frozen = lambda: True
_frozen.patch_frozen_torch()
assert 'triton' not in sys.modules
assert 'torch._inductor' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_every_runtime_triton_kernel_module_ships_its_source():
    """Triton compiles a kernel from its Python source, so the frozen build must ship every module that
    defines one; modules only the AMD build scripts compile ahead of time are exempt."""
    pytest.importorskip("triton")
    import pkgutil

    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITCallable

    import jasna.ltx
    import jasna.models
    from jasna._frozen import LTX_TRACED_MODULES

    amd_build_only = {"jasna.models.ltx_vae.layout_triton"}
    kernel_modules = set()
    for package in (jasna.ltx, jasna.models):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            module = importlib.import_module(info.name)
            if any(isinstance(value, (JITCallable, Autotuner, Heuristics))
                   and getattr(value, "fn", value).__module__ == info.name for value in vars(module).values()):
                kernel_modules.add(info.name)
    assert "jasna.ltx.kernels" in kernel_modules
    assert kernel_modules - amd_build_only <= set(LTX_TRACED_MODULES)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX user accounts")
@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no numeric user IDs")
@pytest.mark.parametrize("module", (
    "torch._inductor.codecache",
    pytest.param("torch_tensorrt.dynamo._defaults", marks=pytest.mark.skipif(
        importlib.util.find_spec("torch_tensorrt") is None, reason="Torch-TensorRT is not installed")),
))
def test_torch_imports_for_a_user_id_without_an_account_name(module):
    code = f"""
import importlib
import os
import pwd
for name in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
    os.environ.pop(name, None)
def unknown(uid):
    raise KeyError(uid)
pwd.getpwuid = unknown
from jasna import _frozen
_frozen.name_a_nameless_user()
importlib.import_module({module!r})
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no numeric user IDs")
def test_nameless_user_fallback_looks_the_account_up_only_when_asked(monkeypatch):
    import pwd
    from jasna import _frozen

    lookups = []

    def unknown(uid):
        lookups.append(uid)
        raise KeyError(uid)

    monkeypatch.setattr(pwd, "getpwuid", unknown)
    monkeypatch.setenv("HOME", "/srv/jasna")
    _frozen.name_a_nameless_user()
    assert lookups == []
    account = pwd.getpwuid(os.getuid())
    assert (account.pw_name, account.pw_dir) == ("jasna", "/srv/jasna")
    with pytest.raises(KeyError):
        pwd.getpwuid(os.getuid() + 1)


def test_bundled_triton_import_needs_no_external_sdk_command():
    pytest.importorskip("triton")
    code = """
import subprocess
import sys
from jasna._frozen import _import_bundled_triton
def denied(*args, **kwargs):
    raise AssertionError('external SDK command')
subprocess.check_output = denied
_import_bundled_triton()
assert 'triton' in sys.modules
assert subprocess.check_output is denied
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_prebuilt_triton_helper_avoids_a_customer_c_compiler(monkeypatch, tmp_path):
    from jasna import _frozen

    triton = pytest.importorskip("triton")
    monkeypatch.setattr(triton.knobs.build, "impl", None)
    module = tmp_path / ("cuda_utils" + sysconfig.get_config_var("EXT_SUFFIX"))
    module.write_bytes(b"prebuilt fixture")
    _frozen.use_prebuilt_triton_modules(tmp_path)
    assert triton.runtime.build._build("cuda_utils", "source.c", str(tmp_path), [], [], [], []) == str(module)
    with pytest.raises(FileNotFoundError):
        triton.runtime.build._build("missing", "source.c", str(tmp_path), [], [], [], [])


def test_frozen_inductor_runs_in_process_and_keeps_runtime_cache_paths(monkeypatch, torch_config):
    import torch._inductor.config as inductor_config
    from torch._inductor import codecache
    from jasna import _frozen

    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(_frozen, "register_ltx_sources", lambda root: None)
    monkeypatch.setattr(_frozen, "use_prebuilt_triton_modules", lambda directory: None)
    torch_config(inductor_config, compile_threads=8)
    monkeypatch.setattr(codecache, "torch_key", codecache.torch_key_cache(lambda: b"source"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/private/process/inductor")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/private/process/triton")
    _frozen.patch_frozen_torch()
    assert inductor_config.compile_threads == 1
    assert codecache.torch_key() != b"source"
    assert _frozen.os.environ["TRITON_BACKENDS_IN_TREE"] == "1"
    assert _frozen.os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/private/process/inductor"
    assert _frozen.os.environ["TRITON_CACHE_DIR"] == "/private/process/triton"


@pytest.mark.skipif(sys.platform != "win32", reason="Triton looks for the CUDA Toolkit only on Windows")
def test_frozen_startup_without_cuda_toolkit_shows_no_triton_warning(tmp_path):
    pytest.importorskip("triton")
    repository = Path(__file__).resolve().parents[1]
    environment = {name: value for name, value in os.environ.items() if name not in ("CUDA_PATH", "CUDA_HOME")}
    environment["ProgramFiles(x86)"] = environment["ProgramW6432"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", """
import sys, sysconfig, warnings
platlib_paths = sysconfig.get_paths
sysconfig.get_paths = lambda *args, **kwargs: {**platlib_paths(*args, **kwargs), "platlib": sys.argv[2]}
sys.path.insert(0, sys.argv[1])
from jasna import _frozen
_frozen.is_frozen = lambda: True
_frozen.register_ltx_sources = lambda root: None
_frozen.use_prebuilt_triton_modules = lambda directory: None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _frozen.patch_frozen_torch()
    from triton.windows_utils import find_cuda
    assert find_cuda() == (None, [], [])
print([str(warning.message) for warning in caught if "CUDA" in str(warning.message)])
""", str(repository), str(tmp_path)],
        env=environment, capture_output=True, text=True, check=True, timeout=300,
    )
    assert result.stdout.strip() == "[]", result.stdout + result.stderr


@pytest.mark.parametrize("platform", ("linux", "win32"))
def test_frozen_build_compiles_in_process_with_prebuilt_triton_modules(monkeypatch, tmp_path, platform, torch_config):
    import torch._inductor.config as inductor_config
    import torch.jit
    from jasna import _frozen, os_utils

    triton = pytest.importorskip("triton")
    registered_roots = []
    monkeypatch.setattr(_frozen, "_patched", False)
    monkeypatch.setattr(_frozen, "is_frozen", lambda: True)
    monkeypatch.setattr(_frozen, "register_ltx_sources", registered_roots.append)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os_utils, "get_application_executable", lambda: str(tmp_path / "jasna"))
    monkeypatch.delenv("TRITON_BACKENDS_IN_TREE", raising=False)
    torch_config(inductor_config, compile_threads=inductor_config.compile_threads)
    monkeypatch.setattr(triton.knobs.build, "impl", None)
    directory = tmp_path / "triton_prebuilt"
    directory.mkdir()
    expected = directory / ("cuda_utils" + sysconfig.get_config_var("EXT_SUFFIX"))
    expected.write_bytes(b"prebuilt fixture")
    from torch._inductor import codecache
    monkeypatch.setattr(codecache, "torch_key", codecache.torch_key_cache(lambda: b"source"))
    with patch.object(torch.jit, "interface"):
        _frozen.patch_frozen_torch()

    assert inductor_config.compile_threads == 1
    assert codecache.torch_key() != b"source"
    assert os.environ["TRITON_BACKENDS_IN_TREE"] == "1"
    module = triton.runtime.build._build("cuda_utils", "src.c", str(tmp_path), [], [], [], [])
    assert module == str(expected)
    assert registered_roots == [tmp_path]
