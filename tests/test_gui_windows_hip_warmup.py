"""Execute the actual source function with fake Torch; never initialize Tk/GPU."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_local_app = Path(__file__).with_name('app.py')
SOURCE = Path(os.environ.get('JASNA_GUI_APP_UNDER_TEST',
    _local_app if _local_app.is_file() else Path(__file__).resolve().parents[1] / 'jasna/gui/app.py'))


def load_warmup(platform, torch):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_warm_up_cuda']
    if len(definitions) != 1:
        raise RuntimeError('ambiguous source function')
    namespace = {'sys': SimpleNamespace(platform=platform)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    def call():
        with patch.dict(sys.modules, {'torch': torch}):
            namespace['_warm_up_cuda']()
    return call


class WarmupTests(unittest.TestCase):
    def fake(self, hip='7.2', available=True, fail=False):
        calls = []
        def sync():
            calls.append('sync')
            if fail:
                raise RuntimeError('injected synchronize failure')
        def zeros(*args, **kwargs):
            self.assertEqual(args, (1,))
            self.assertEqual(kwargs, {'device': 'cuda'})
            calls.append('zeros')
        return SimpleNamespace(version=SimpleNamespace(hip=hip), zeros=zeros,
            cuda=SimpleNamespace(is_available=lambda:available, synchronize=sync)), calls

    def test_windows_hip_synchronizes_after_allocation(self):
        torch, calls = self.fake()
        load_warmup('win32', torch)()
        self.assertEqual(calls, ['zeros', 'sync'])

    def test_no_gpu_does_not_allocate_or_synchronize(self):
        torch, calls = self.fake(available=False)
        load_warmup('win32', torch)()
        self.assertEqual(calls, [])

    def test_other_backends_are_unchanged(self):
        for platform, hip in (('linux','7.2'), ('win32',None), ('linux',None), ('darwin',None)):
            with self.subTest(platform=platform, hip=hip):
                torch, calls = self.fake(hip=hip)
                load_warmup(platform, torch)()
                self.assertEqual(calls, ['zeros'])

    def test_sync_failure_is_not_silently_swallowed(self):
        torch, calls = self.fake(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            load_warmup('win32', torch)()
        self.assertEqual(calls, ['zeros', 'sync'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
