from dataclasses import asdict, replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from factories import session_config
from jasna import job_resume
from jasna.job_resume import ResumeJob, ResumeMismatchError, atomic_write, find_resume_output, job_fingerprint, write_json
from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import TrackPlan, Window, plans_from_json, plans_to_json
from jasna.ltx.restore import LatentStore
from jasna.segments import SegmentRange, SegmentRestoration


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    config = session_config(restoration_model_name="ltx")
    metadata = SimpleNamespace(num_frames=120, time_base=Fraction(1, 30), duration=4.0, codec_name="hevc")
    return config, source, output, metadata


def test_plans_round_trip_preserves_crops_float64_polygons_and_weights(tmp_path):
    crop = CropFrame((1, 2, 640, 480), (384, 512), (64, 64, 0, 0))
    integer = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    floating = integer.astype(np.float64) + 0.1234567890123456
    plans = [TrackPlan(7, 12, ((integer,), (floating,)), (0.1, 1.0), (Window(3, 7, 12, 2, (crop, crop)),), 104)]
    path = tmp_path / "plans.json"
    write_json(path, plans_to_json(plans))
    (restored,) = plans_from_json(job_resume.read_json(path))
    assert restored.windows == plans[0].windows
    assert restored.presence == plans[0].presence
    assert (restored.track_id, restored.start, restored.window_hop) == (7, 12, 104)
    for expected, actual in zip((integer, floating), (restored.polygons[0][0], restored.polygons[1][0])):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_atomic_write_does_not_publish_interrupted_file(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic_write(target, "w") as stream:
            stream.write("new")
            raise RuntimeError("interrupted")
    assert target.read_text() == "old"


def test_latents_ignore_part_files_and_load_only_weights(tmp_path):
    store = LatentStore(tmp_path, "reference")
    (tmp_path / "000003.reference.pt.part").write_bytes(b"unfinished")
    assert not store.has(3)
    store.put(3, torch.tensor([1.0, 2.0]))
    with patch("torch.load", wraps=torch.load) as load:
        assert torch.equal(store.get(3), torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert load.call_args.kwargs["weights_only"] is True
    assert store.has(3)


def test_fingerprint_ignores_progress_and_working_directory(job, tmp_path):
    config, source, output, metadata = job
    changed = replace(config, disable_progress=True, working_dir=tmp_path / "scratch")
    assert job_fingerprint(config, source, output, metadata, ()) == job_fingerprint(changed, source, output, metadata, ())


@pytest.mark.parametrize("change", ["seed", "model", "settings", "ranges", "input", "version", "output", "metadata"])
def test_fingerprint_detects_changes(job, monkeypatch, change):
    config, source, output, metadata = job
    original = job_fingerprint(config, source, output, metadata, ())
    ranges = ()
    if change == "seed":
        config = replace(config, ltx_seed=42)
    elif change == "model":
        config = replace(config, ltx_model="distilled2")
    elif change == "settings":
        config = replace(config, encoder_settings={"cq": 12})
    elif change == "ranges":
        ranges = (SegmentRange(1, 2),)
    elif change == "input":
        source.write_bytes(b"changed source")
    elif change == "version":
        monkeypatch.setattr(job_resume, "__version__", "different")
    elif change == "output":
        output = output.with_suffix(".mkv")
    else:
        metadata = SimpleNamespace(num_frames=120, time_base=Fraction(1, 24), duration=5.0)
    assert job_fingerprint(config, source, output, metadata, ranges) != original


def test_mismatch_preserves_every_file_until_confirmed(job):
    config, source, output, metadata = job
    original = ResumeJob(config, source, output, metadata, ())
    original.prepare()
    latent = original.directory / "latent.pt"
    latent.write_bytes(b"saved")
    changed = ResumeJob(replace(config, ltx_seed=7), source, output, metadata, ())
    with pytest.raises(ResumeMismatchError) as raised:
        changed.prepare()
    assert latent.read_bytes() == b"saved"
    raised.value.discard()
    changed.prepare()
    assert changed.matches()
    changed.finish()
    assert not changed.root.exists()


@pytest.mark.parametrize("renamed", [False, True])
def test_find_saved_output_before_file_conflict_even_after_auto_rename(job, renamed):
    config, source, output, metadata = job
    saved = output.with_stem(output.stem + " (1)") if renamed else output
    checkpoint = ResumeJob(config, source, saved, metadata, ())
    checkpoint.prepare()
    saved.touch()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(config, source, output, ()) == saved
        assert find_resume_output(replace(config, ltx_seed=42), source, output, ()) == saved


def test_resume_lookup_handles_brackets_and_only_follows_actual_auto_rename_names(job):
    config, source, output, metadata = job
    output = output.with_stem("[group] output")
    unrelated = output.with_stem(output.stem + " other")
    ResumeJob(config, source, unrelated, metadata, ()).prepare()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(config, source, output, ()) is None
        saved = output.with_stem(output.stem + " (1)")
        ResumeJob(config, source, saved, metadata, ()).prepare()
        assert find_resume_output(config, source, output, ()) == saved


def test_find_auto_split_checkpoint_with_effective_codec_and_ranges(job):
    config, source, output, metadata = job
    config = replace(config, ltx_auto_split_percent=12.0, codec="av1")
    segments = (SegmentRange(1, 2, SegmentRestoration("ltx", 0)),)
    from jasna.session_factory import segment_job_config

    effective = segment_job_config(config, metadata.codec_name, source)
    checkpoint = ResumeJob(effective, source, output, metadata, segments)
    checkpoint.prepare()
    write_json(checkpoint.root / "split.json", {
        "fingerprint": job_fingerprint(config, source, output, metadata, ()),
        "segments": [asdict(segment) for segment in segments],
    })
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(config, source, output, ()) == output


@pytest.mark.parametrize("answer", ["yes", "no", None])
def test_cli_asks_before_replacing_progress(job, answer):
    from jasna.main import _run_resumable_video

    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, ())
    checkpoint.prepare()
    error = ResumeMismatchError(checkpoint.root)
    pipeline = MagicMock()
    pipeline.run.side_effect = [error, None]
    with patch("sys.stdin.isatty", return_value=answer is not None), patch("builtins.input", return_value=answer) as prompt:
        if answer == "yes":
            _run_resumable_video(lambda *_: pipeline, source, output)
            assert pipeline.run.call_count == 2
            assert not checkpoint.root.exists()
        else:
            with pytest.raises(ResumeMismatchError):
                _run_resumable_video(lambda *_: pipeline, source, output)
            assert checkpoint.matches()
        assert prompt.called is (answer is not None)


@pytest.mark.parametrize("render_started", [False, True])
def test_auto_split_reuses_saved_detection_and_rejects_changed_settings(job, render_started):
    from jasna.media.splice import KeyframeIndex
    from jasna.mosaic_scan import MosaicSamples
    from jasna.session_factory import ltx_auto_split

    config, source, output, metadata = job
    config = replace(config, ltx_auto_split_percent=12.0, vr_mode="off")
    metadata.duration = 6.0
    metadata.num_frames = 180
    metadata.video_width, metadata.video_height = 1920, 1080
    metadata.sample_aspect_ratio = Fraction(1, 1)
    samples = MosaicSamples(times=(1.0, 3.0), scores=(0.9, 0.9), sizes=(0.05, 0.3), stride=1.0)
    with (
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=KeyframeIndex((0, 60, 120), Fraction(1,30), 0, 180)),
        patch("jasna.mosaic_scan.scan_mosaic_sizes", return_value=samples) as scan,
    ):
        first = ltx_auto_split(config, MagicMock(), source, output, progress_callback=None)
        second = ltx_auto_split(config, MagicMock(), source, output, progress_callback=None)
        assert first == second
        assert first is not None
        scan.assert_called_once()
        checkpoint = ResumeJob(first.config, source, output, metadata, first.segments)
        if render_started:
            checkpoint.prepare()
        saved_split = (checkpoint.root / "split.json").read_bytes()
        with pytest.raises(ResumeMismatchError):
            ltx_auto_split(replace(config, ltx_seed=42), MagicMock(), source, output, progress_callback=None)
        scan.assert_called_once()
        assert (checkpoint.root / "split.json").read_bytes() == saved_split
        if render_started:
            assert checkpoint.matches()
