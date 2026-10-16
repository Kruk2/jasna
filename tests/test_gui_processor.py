from jasna.gui.processor import ProgressGate
from jasna.progressbar import StageProgress


def _stage(name: str) -> StageProgress:
    return StageProgress(name, 1, 4, 0.0, 0.0)


def test_progress_gate_throttles_updates_but_never_a_new_ltx_pass() -> None:
    gate = ProgressGate(0.1)
    assert gate.admits(10.0, _stage("encode"))
    assert not gate.admits(10.05, _stage("encode"))
    assert gate.admits(10.06, _stage("denoise"))
    assert gate.admits(10.2, _stage("denoise"))
    assert gate.admits(10.25, None)
    assert not gate.admits(10.3, None)
