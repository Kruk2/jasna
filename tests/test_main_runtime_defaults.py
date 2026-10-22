import os
from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize('platform', ('win32', 'linux'))
@pytest.mark.parametrize('existing', (None, '', 'custom'))
def test_entry_defaults_precede_frozen_torch_initialization(monkeypatch, platform, existing):
    pytest.importorskip("jasna.protection.runtime")
    from jasna import _frozen, protection

    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(_frozen, 'is_frozen', lambda: True)
    for name in ('OMP_WAIT_POLICY', 'CUDA_MODULE_LOADING'):
        if existing is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, existing)

    class RuntimeReached(Exception):
        pass

    def initialize():
        actual_cuda = os.environ.get('CUDA_MODULE_LOADING')
        assert actual_cuda == ('LAZY' if existing is None else existing)
        expected_omp = 'passive' if existing is None else existing
        actual_omp = os.environ.get('OMP_WAIT_POLICY')
        assert actual_omp == expected_omp
        raise RuntimeReached

    monkeypatch.setattr(protection, 'initialize_runtime', initialize)
    entry = Path(__file__).resolve().parents[1] / 'jasna/__main__.py'
    with pytest.raises(RuntimeReached):
        runpy.run_path(str(entry), run_name='__main__')
