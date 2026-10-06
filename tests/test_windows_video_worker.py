"""CPU-only source-runtime command preparation and existing launcher entry tests."""
import ast
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

PRODUCT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRODUCT))
from jasna.gui import windows_video_worker as worker
from jasna.gui.windows_guarded_attempt import GuardedAttemptConfig, WindowsGuardedAttempt


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/run_jasna_unified.py").touch()
        self.python = self.root / "python.exe"
        self.python.touch()
        self.request = self.root / "unique-request.json"
        self.request.write_text("{}", encoding="utf-8")
        self.runtime = worker.WindowsUnifiedWorkerRuntime(self.root, self.root / "runtime", self.python)

    def test_explicit_launcher_and_existing_environment_builder(self):
        original = {"JASNA_MAIN_PID":"parent", "UNCHANGED":"keep"}
        with patch.object(worker.sys, "platform", "win32"), patch.object(worker, "is_frozen", return_value=False), patch.object(worker, "build_runtime_environment", return_value=dict(original)) as build:
            command, environment = self.runtime.prepare(self.request, original)
        self.assertEqual(command, [str(self.python), "-B", str(self.root / "scripts/run_jasna_unified.py"),
            "--_product-child", "--runtime-root", str(self.root / "runtime"), "--repo-root", str(self.root),
            "--", "--isolated-video-job", str(self.request)])
        build.assert_called_once_with(self.root / "runtime", self.root, python_executable=self.python,
            platform="win32", base_environment=original)
        self.assertNotIn("JASNA_MAIN_PID", environment)
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(environment["JASNA_WINDOWS_WORKER_GPU_IDENTITY"], "1")
        self.assertEqual(original, {"JASNA_MAIN_PID":"parent", "UNCHANGED":"keep"})

    def test_no_silent_frozen_or_other_platform_fallback(self):
        for platform, frozen in (("linux", False), ("win32", True)):
            with self.subTest(platform=platform, frozen=frozen), patch.object(worker.sys, "platform", platform), patch.object(worker, "is_frozen", return_value=frozen), patch.object(worker, "build_runtime_environment") as build:
                with self.assertRaises(RuntimeError):
                    self.runtime.prepare(self.request, {})
                build.assert_not_called()

    def test_paths_and_layout_fail_closed(self):
        with self.assertRaises(ValueError):
            worker.WindowsUnifiedWorkerRuntime(Path("relative"), self.root, self.python)
        with self.assertRaises(FrozenInstanceError):
            self.runtime.repo_root = self.root / "other"
        with patch.object(worker.sys, "platform", "win32"), patch.object(worker, "is_frozen", return_value=False), patch.object(worker, "build_runtime_environment", side_effect=RuntimeError("bad ABI")):
            with self.assertRaisesRegex(RuntimeError, "bad ABI"):
                self.runtime.prepare(self.request, {})
            with self.assertRaises(ValueError):
                self.runtime.prepare(self.root / "missing.json", {})

    def test_guard_requires_an_explicit_runtime_and_delegates_without_media(self):
        backend = WindowsGuardedAttempt(GuardedAttemptConfig(self.python, self.root))
        with self.assertRaisesRegex(RuntimeError, "not explicitly configured"):
            backend.prepare_request(self.request, {})
        backend = WindowsGuardedAttempt(GuardedAttemptConfig(self.python, self.root), worker_runtime=self.runtime)
        with patch.object(worker.WindowsUnifiedWorkerRuntime, "prepare", return_value=(["worker"], {"env":"ok"})) as prepare:
            self.assertEqual(backend.prepare_request(self.request, {}), (["worker"], {"env":"ok"}))
            prepare.assert_called_once_with(self.request, {})

    def test_existing_launcher_parser_accepts_exact_worker_command(self):
        spec = importlib.util.spec_from_file_location("worker_launcher_cpu", PRODUCT / "scripts/run_jasna_unified.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        args, trailing = launcher.build_parser().parse_known_args(["--_product-child", "--runtime-root",
            str(self.runtime.runtime_root), "--repo-root", str(self.root), "--", "--isolated-video-job", str(self.request)])
        self.assertTrue(args._product_child)
        self.assertEqual(trailing, ["--", "--isolated-video-job", str(self.request)])
        # Actual _product_child keeps the validated launcher's loaded-runtime
        # call, cwd and shared jasna entry; fake native validation/runpy only.
        with patch.object(launcher, "validate_loaded_runtime") as validate, patch.object(launcher.os, "chdir") as chdir, patch.object(launcher.runpy, "run_module") as run, patch.object(launcher.sys, "argv", []):
            self.assertEqual(launcher._product_child(self.runtime.runtime_root, self.root, trailing[1:]), 0)
            validate.assert_called_once_with(self.runtime.runtime_root, self.root)
            chdir.assert_called_once_with(self.root)
            run.assert_called_once_with("jasna", run_name="__main__", alter_sys=True)
            self.assertEqual(launcher.sys.argv[1:], trailing[1:])

    def test_processor_legacy_command_and_request_protocol_are_retained(self):
        tree = ast.parse((PRODUCT / "jasna/gui/processor.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Processor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_process_isolated_video_job")
        names = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
        for name in ("build_video_job_request", "write_video_job_request", "video_job_command", "backend.prepare_request"):
            self.assertIn(name, names)

    def test_identity_report_opt_in_and_token_are_required_before_native_import(self):
        with patch.dict(worker.os.environ, {}, clear=True), patch.dict(sys.modules, {"torch": None}):
            worker.report_windows_worker_gpu_identity(None)
        for token in ("", "A" * 32, "a" * 31, "a" * 33):
            with self.subTest(token=token), patch.object(worker.sys, "platform", "win32"), patch.dict(worker.os.environ,
                    {"JASNA_WINDOWS_WORKER_GPU_IDENTITY":"1", "JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN":token}, clear=True), patch.dict(sys.modules, {"torch":None}):
                with self.assertRaisesRegex(RuntimeError, "guarded attempt token"):
                    worker.report_windows_worker_gpu_identity(None)

    def test_worker_identity_closes_reader_before_emitting_exact_protocol(self):
        from jasna.windows_global_vram import WindowsGpuIdentity
        from unittest.mock import Mock
        identity = WindowsGpuIdentity("luid_0x00000000_0x00000001_phys_", 0)
        reader = Mock(identity=identity)
        hip = object()
        destination = object()
        events = []
        def emit(stream, event):
            reader.close.assert_called_once_with()
            self.assertIs(stream, destination)
            events.append(event)
        with patch.object(worker.sys, "platform", "win32"), patch.dict(worker.os.environ,
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY":"1", "JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN":"a"*32}, clear=True), patch.dict(sys.modules,
                {"torch":SimpleNamespace(version=SimpleNamespace(hip="7.16")),
                 "jasna.media.hip_kernel":SimpleNamespace(hip_runtime=lambda:hip)}), patch("jasna.windows_global_vram.WindowsGlobalVramReader", return_value=reader) as create, patch("jasna.gui.video_job_process._emit_event", side_effect=emit):
            worker.report_windows_worker_gpu_identity(destination)
        create.assert_called_once_with(0, hip)
        self.assertEqual(events, [dict(type="windows_gpu_identity", attempt_token="a"*32,
            adapter_marker=identity.adapter_marker, node_index=0)])

    def test_worker_identity_does_not_emit_after_reader_close_failure(self):
        from unittest.mock import Mock
        reader = Mock()
        reader.close.side_effect = RuntimeError("PDH close failed")
        with patch.object(worker.sys, "platform", "win32"), patch.dict(worker.os.environ,
                {"JASNA_WINDOWS_WORKER_GPU_IDENTITY":"1", "JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN":"a"*32}, clear=True), patch.dict(sys.modules,
                {"torch":SimpleNamespace(version=SimpleNamespace(hip="7.16")),
                 "jasna.media.hip_kernel":SimpleNamespace(hip_runtime=lambda:None)}), patch("jasna.windows_global_vram.WindowsGlobalVramReader", return_value=reader), patch("jasna.gui.video_job_process._emit_event") as emit:
            with self.assertRaisesRegex(RuntimeError, "PDH close failed"):
                worker.report_windows_worker_gpu_identity(None)
            emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
