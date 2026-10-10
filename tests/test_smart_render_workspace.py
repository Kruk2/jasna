from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

from jasna.smart_render_workspace import (
    SmartRenderWorkspace,
    WORKSPACE_ALGORITHM_VERSION,
    workspace_signature,
)


def _plan():
    index = SimpleNamespace(
        time_base=Fraction(1, 30),
        start_pts=0,
        end_pts=180,
        pts=(0, 60, 120),
        max_b_frames=2,
        uses_b_references=False,
    )
    spans = (
        SimpleNamespace(
            kind="copy",
            start_pts=0,
            end_pts=60,
            effect_ranges=(),
        ),
        SimpleNamespace(
            kind="render",
            start_pts=60,
            end_pts=120,
            effect_ranges=((75, 90),),
        ),
    )
    return SimpleNamespace(index=index, spans=spans)


def _signature(tmp_path: Path, **processing):
    source = tmp_path / "source.mp4"
    if not source.exists():
        source.write_bytes(b"source-video")
    model = tmp_path / "model.pt"
    if not model.exists():
        model.write_bytes(b"model")
    return workspace_signature(
        source=source,
        output=tmp_path / "restored.mp4",
        plan=_plan(),
        processing={"threshold": 0.5, **processing},
        model_files={"detection": model},
        codec="h264",
        encoder_settings={"cq": 22},
        resolved_projection="flat",
    )


def test_completed_fragment_is_reused_only_while_identity_matches(tmp_path: Path) -> None:
    workspace = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=_signature(tmp_path),
    )
    fragment = workspace.fragment_path(1, ".ts")
    fragment.write_bytes(b"valid-fragment")
    workspace.mark_complete(1, fragment)

    reopened = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=_signature(tmp_path),
    )
    assert reopened.reusable_fragment(1) == fragment.resolve()

    fragment.write_bytes(b"tampered")
    assert reopened.reusable_fragment(1) is None


def test_running_span_returns_to_pending_after_restart(tmp_path: Path) -> None:
    signature = _signature(tmp_path)
    workspace = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=signature,
    )
    workspace.mark_running(1)

    reopened = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=signature,
    )

    assert reopened.manifest["spans"][1]["status"] == "pending"
    assert reopened.manifest["spans"][1]["artifact"] is None


def test_signature_change_uses_a_different_workspace(tmp_path: Path) -> None:
    first = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=_signature(tmp_path, padding=1),
    )
    second = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=_signature(tmp_path, padding=2),
    )

    assert first.path != second.path


def test_algorithm_version_bump_invalidates_v2_fragments(tmp_path: Path) -> None:
    current_signature = _signature(tmp_path)
    legacy_signature = {
        **current_signature,
        "algorithm_version": "jasna-smart-render-workspace-v2",
    }
    legacy_workspace = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=legacy_signature,
    )
    legacy_fragment = legacy_workspace.fragment_path(1, ".ts")
    legacy_fragment.write_bytes(b"v2-fragment")
    legacy_workspace.mark_complete(1, legacy_fragment)

    current_workspace = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=current_signature,
    )

    assert WORKSPACE_ALGORITHM_VERSION == "jasna-smart-render-workspace-v4"
    assert current_workspace.path != legacy_workspace.path
    assert current_workspace.reusable_fragment(1) is None


def test_invalid_manifest_is_preserved_and_replaced(tmp_path: Path) -> None:
    signature = _signature(tmp_path)
    workspace = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=signature,
    )
    workspace.manifest_path.write_text("{broken", encoding="utf-8")

    reopened = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=signature,
    )

    assert json.loads(reopened.manifest_path.read_text(encoding="utf-8"))
    assert list(reopened.path.glob("manifest.invalid-*.json"))


def test_cleanup_removes_only_valid_workspace(tmp_path: Path) -> None:
    workspace = SmartRenderWorkspace.open(
        tmp_path,
        output=tmp_path / "restored.mp4",
        signature=_signature(tmp_path),
    )
    path = workspace.path

    workspace.cleanup()

    assert not path.exists()
