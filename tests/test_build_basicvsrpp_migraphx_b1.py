import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_basicvsrpp_migraphx_b1.py"
SPEC = importlib.util.spec_from_file_location("build_basicvsrpp_b1", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_offline_compiler_requires_explicit_checkpoint_and_new_destination():
    with pytest.raises(SystemExit):
        builder.build_parser().parse_args([])
    args = builder.build_parser().parse_args([
        "--checkpoint", "model.pth", "--output-dir", "new-artifacts"
    ])
    assert args.checkpoint == Path("model.pth")
    assert args.output_dir == Path("new-artifacts")


def test_compiler_semantic_snapshot_matches_product_contract(tmp_path):
    from jasna.restorer.basicvsrpp_migraphx_b1 import _SEMANTIC_SOURCE_HASHES

    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    snapshot = builder.snapshot_sources(checkpoint, _SEMANTIC_SOURCE_HASHES)
    assert {name: snapshot[name] for name in _SEMANTIC_SOURCE_HASHES} == _SEMANTIC_SOURCE_HASHES
    assert snapshot[str(checkpoint)] == builder.sha256_file(checkpoint)


def test_compiler_rejects_unreviewed_source_before_snapshotting_checkpoint(tmp_path):
    from jasna.restorer.basicvsrpp_migraphx_b1 import _SEMANTIC_SOURCE_HASHES

    expected = dict(_SEMANTIC_SOURCE_HASHES)
    expected[next(iter(expected))] = "0" * 64
    with pytest.raises(RuntimeError, match="reviewed product contract"):
        builder.snapshot_sources(tmp_path / "missing.pth", expected)
