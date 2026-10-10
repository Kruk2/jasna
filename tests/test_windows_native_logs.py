"""Pure-stdlib policy checks: no PyAV, GPU, GUI, codecs or subprocesses."""
import ast
import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'jasna/windows_native_logs.py'


def load_policy():
    spec = importlib.util.spec_from_file_location('native_logs_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admitted():
    return {'JASNA_WINDOWS_NATIVE_FFMPEG_LOGS': '1',
            'JASNA_ISOLATED_VIDEO_JOB': '1',
            'JASNA_WINDOWS_WORKER_GPU_IDENTITY': '1',
            'JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN': '0123456789abcdef' * 2}


def fake_av():
    calls = []
    log = SimpleNamespace(ERROR=16, WARNING=24,
        set_level=lambda value: calls.append(('python', value)),
        get_level=lambda: None,
        set_libav_level=lambda value: calls.append(('native_level', value)),
        restore_default_callback=lambda: calls.append(('native_callback',)))
    stats = lambda *args: calls.append(('unsafe_stats',))
    return SimpleNamespace(__version__='18.1.0', logging=log,
        filter=SimpleNamespace(stats=stats), loudnorm=SimpleNamespace(stats=stats,
            set_level=log.set_level, get_level=log.get_level)), calls


class NativeLogsTests(unittest.TestCase):
    def setUp(self):
        self.module = load_policy()
        patcher = patch.dict(sys.modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def install(self, av):
        sys.modules.update({'av': av, 'av.filter': av.filter, 'av.filter.loudnorm': av.loudnorm})
        return self.module.install_worker_native_logs(environ=admitted(), platform='win32')

    def test_default_off_never_imports_native_runtime(self):
        original = builtins.__import__
        seen = []
        def checked(name, *args, **kwargs):
            seen.append(name)
            if name.split('.')[0] in {'av', 'torch', 'tkinter', 'customtkinter'}:
                raise AssertionError('forbidden native import')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=checked):
            module = load_policy()
            for value in ('', '0', 'FALSE', ' off ', 'no'):
                self.assertIsNone(module.install_worker_native_logs(
                    environ={module.ENV: value}, platform='linux'))
        self.assertNotIn('av', seen)

    def test_admission_rejects_without_importing_av(self):
        bad = []
        for key in admitted():
            env = admitted(); del env[key]
            if key != self.module.ENV:
                bad.append((env, 'win32'))
        for value in ('', 'x' * 32, 'a' * 31, 'A' * 32, 'a' * 32 + '\n'):
            env = admitted(); env['JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN'] = value
            bad.append((env, 'win32'))
        env = admitted(); env[self.module.ENV] = 'maybe'; bad.append((env, 'win32'))
        bad.extend((admitted(), platform) for platform in ('linux', 'darwin'))
        with patch.dict(sys.modules, {'av': None}):
            for env, platform in bad:
                with self.subTest(env=env, platform=platform), self.assertRaises(RuntimeError):
                    self.module.install_worker_native_logs(environ=env, platform=platform)

    def test_install_native_only_and_idempotent(self):
        av, calls = fake_av()
        policy = self.install(av)
        self.assertIs(self.install(av), policy)
        self.assertEqual(calls, [('native_level', 24), ('native_callback',)])
        self.assertEqual(policy.snapshot(), dict(mode='native_process_lifetime', level=24,
            redirected_level_requests=0, python_callback_setter_used=False))

    def test_late_level_calls_never_reinstall_python_callback(self):
        av, calls = fake_av(); policy = self.install(av)
        for setter in (av.logging.set_level, av.logging.set_libav_level):
            for requested, expected in ((None, 24), (-8, 16), (0, 16), (8, 16),
                    (16, 16), (20, 20), (24, 24), (40, 24), (64, 24)):
                setter(requested)
                self.assertEqual(av.logging.get_level(), expected)
                self.assertEqual(calls[-1], ('native_level', expected))
        self.assertFalse(any(call[0] == 'python' for call in calls))
        self.assertEqual(policy.snapshot()['redirected_level_requests'], 18)

    def test_bad_levels_leave_native_state_unchanged(self):
        av, calls = fake_av(); policy = self.install(av)
        before = list(calls)
        for value in (True, False, 16.0, '16', -9, 65, object()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                av.logging.set_level(value)
        self.assertEqual(calls, before)
        self.assertEqual(policy.snapshot()['redirected_level_requests'], 0)

    def test_wrong_version_constants_and_missing_api_fail_before_mutation(self):
        for kind in ('version', 'constants', 'set_level', 'get_level',
                     'set_libav_level', 'restore_default_callback'):
            av, calls = fake_av()
            if kind == 'version': av.__version__ = '18.0.0'
            elif kind == 'constants': av.logging.ERROR = 40
            else: setattr(av.logging, kind, None)
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): self.install(av)
            self.assertEqual(calls, [])
            self.assertIsNone(self.module._POLICY)

    def test_unexpected_loudnorm_alias_rejected_before_mutation(self):
        av, calls = fake_av()
        av.loudnorm.set_level = lambda value: None
        with self.assertRaisesRegex(RuntimeError, 'unexpected PyAV loudnorm'): self.install(av)
        self.assertEqual(calls, [])

    def test_binding_conflicts_fail_closed(self):
        for name in ('set_level', 'set_libav_level', 'get_level', 'restore_default_callback'):
            self.module = load_policy(); av, calls = fake_av(); policy = self.install(av)
            setter = av.logging.set_level
            setattr(av.logging, name, lambda *args: None)
            before = list(calls)
            for operation in (policy.assert_active, policy.snapshot, lambda: self.install(av),
                              lambda: setter(16)):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'replaced'):
                    operation()
            self.assertEqual(calls, before)

    def test_loudnorm_stats_rejected_and_cached_level_alias_redirected(self):
        av, calls = fake_av(); policy = self.install(av)
        for stats in (av.filter.stats, av.loudnorm.stats):
            with self.assertRaisesRegex(RuntimeError, 'unsupported'): stats('unused', None)
        av.loudnorm.set_level(40)
        self.assertEqual(av.loudnorm.get_level(), 24)
        self.assertFalse(any(call[0] in {'python', 'unsafe_stats'} for call in calls))
        av.filter.stats = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, 'loudnorm bindings'): policy.assert_active()

    def test_failed_native_initialization_does_not_publish_policy(self):
        av, calls = fake_av(); original = av.logging.set_level
        def fail(): raise RuntimeError('native API failure')
        av.logging.restore_default_callback = fail
        with self.assertRaisesRegex(RuntimeError, 'native API failure'): self.install(av)
        self.assertIsNone(self.module._POLICY)
        self.assertIs(av.logging.set_level, original)
        # Worker entry must treat this as fatal; no unsafe rollback callback.
        self.assertEqual(calls, [('native_level', 24)])



if __name__ == '__main__': unittest.main(verbosity=2)
