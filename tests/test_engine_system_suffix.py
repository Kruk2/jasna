import os

import pytest

from jasna.engine_paths import engine_system_suffix


@pytest.mark.parametrize(("os_name", "suffix"), [("nt", ".win"), ("posix", ".linux")])
def test_engine_names_carry_the_operating_system(monkeypatch, os_name, suffix):
    monkeypatch.setattr(os, "name", os_name)
    assert engine_system_suffix() == suffix
