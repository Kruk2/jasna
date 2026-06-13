import jasna._frozen as _frozen
from jasna._frozen import is_frozen


def test_dev_run_is_not_frozen(monkeypatch):
    monkeypatch.delattr(_frozen.sys, "frozen", raising=False)
    monkeypatch.delitem(_frozen.__dict__, "__compiled__", raising=False)
    assert is_frozen() is False


def test_pyinstaller_sets_sys_frozen(monkeypatch):
    monkeypatch.delitem(_frozen.__dict__, "__compiled__", raising=False)
    monkeypatch.setattr(_frozen.sys, "frozen", True, raising=False)
    assert is_frozen() is True


def test_nuitka_injects_compiled_global(monkeypatch):
    monkeypatch.delattr(_frozen.sys, "frozen", raising=False)
    monkeypatch.setitem(_frozen.__dict__, "__compiled__", object())
    assert is_frozen() is True
