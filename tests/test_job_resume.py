import os
from dataclasses import asdict, replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from factories import session_config
from jasna import job_resume
from jasna.job_resume import ResumeJob, ResumeMismatchError, atomic_write, find_resume_output, job_fingerprints, write_json
from jasna.ltx.camera import CropFrame
from jasna.ltx.model_files import bundle_names
from jasna.ltx.plan import TrackPlan, Window, plans_from_arrays, plans_to_arrays
from jasna.ltx.restore import LatentStore
from jasna.segments import SegmentRange, SegmentRestoration


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    detector = tmp_path / "det.onnx"
    detector.write_bytes(b"detector")
    models = tmp_path / "ltx"
    models.mkdir()
    config = session_config(
        restoration_model_name="ltx", detection_model_path=detector, restoration_model_path=models
    )
    for name in bundle_names(config.ltx_model, fast=config.ltx_fast):
        (models / name).write_bytes(b"weights")
    metadata = SimpleNamespace(num_frames=120, time_base=Fraction(1, 30), duration=4.0, codec_name="hevc")
    return config, source, output, metadata


def _save_work(checkpoint: ResumeJob) -> None:
    (checkpoint.directory / "ltx").mkdir(parents=True, exist_ok=True)
    (checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt").write_bytes(b"latent")


def test_plans_round_trip_preserves_crops_float64_polygons_and_weights(tmp_path):
    crop = CropFrame((1, 2, 640, 480), (384, 512), (64, 64, 0, 0))
    integer = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    floating = integer.astype(np.float64) + 0.1234567890123456
    plans = [TrackPlan(7, 12, ((integer,), (floating, integer)), (0.1, 1.0), (Window(3, 7, 12, 2, (crop, crop)),), 104)]
    path = tmp_path / "plans.npz"
    np.savez(path, **plans_to_arrays([[], plans]))
    with np.load(path, allow_pickle=False) as arrays:
        empty, (restored,) = plans_from_arrays(arrays)
    assert empty == []
    assert restored.windows == plans[0].windows
    assert restored.presence == plans[0].presence
    assert (restored.track_id, restored.start, restored.window_hop) == (7, 12, 104)
    assert [len(frame) for frame in restored.polygons] == [1, 2]
    for expected, actual in zip((integer, floating, integer), (*restored.polygons[0], *restored.polygons[1])):
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
    assert job_fingerprints(config, source, output, metadata, ()) == job_fingerprints(changed, source, output, metadata, ())


def test_changed_lut_contents_redo_the_output_even_with_same_size_and_timestamp(job, tmp_path):
    config, source, output, metadata = job
    lut = tmp_path / "colors.cube"
    lut.write_bytes(b"0 0 0\n1 1 1\n")
    config = replace(config, lut_path=str(lut))
    checkpoint = ResumeJob(config, source, output, metadata, ())
    checkpoint.prepare()
    _save_work(checkpoint)
    stat = lut.stat()
    lut.write_bytes(b"1 1 1\n0 0 0\n")
    os.utime(lut, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = ResumeJob(config, source, output, metadata, ())
    assert changed.fingerprint != checkpoint.fingerprint
    assert changed.latents == checkpoint.latents


@pytest.mark.parametrize("asset, encrypted", [
    ("detector", False), ("basicvsrpp", False),
    ("transformer", False), ("vae", False), ("decoder", False),
    ("transformer", True), ("vae", True), ("decoder", True),
])
def test_model_replaced_at_the_same_path_changes_the_fingerprint(job, tmp_path, monkeypatch, asset, encrypted):
    config, source, output, metadata = job
    ranges = ()
    if asset == "detector":
        path = config.detection_model_path
    elif asset == "basicvsrpp":
        path = tmp_path / "basic.pth"
        path.write_bytes(b"weights")
        monkeypatch.setattr("jasna.engine_paths.default_restoration_model_path", lambda name: path)
        ranges = (SegmentRange(0, 1, SegmentRestoration("basicvsrpp", None)), SegmentRange(1, 2))
    else:
        index = {"transformer": 0, "vae": 1, "decoder": 2}[asset]
        path = config.restoration_model_path / bundle_names(config.ltx_model, fast=config.ltx_fast)[index]
        if encrypted:
            path = path.rename(path.with_name(path.name + ".enc"))
    checkpoint = ResumeJob(config, source, output, metadata, ranges)
    checkpoint.prepare()
    _save_work(checkpoint)
    stat = path.stat()
    path.write_bytes(b"updated")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    changed = ResumeJob(config, source, output, metadata, ranges)
    assert changed.fingerprint != checkpoint.fingerprint
    if asset == "basicvsrpp":
        assert changed.latents == checkpoint.latents
    else:
        with pytest.raises(ResumeMismatchError):
            changed.prepare()


def test_frozen_fingerprint_uses_encrypted_bundle_even_when_plaintext_exists(job, monkeypatch):
    config, source, output, metadata = job
    for path in tuple(config.restoration_model_path.iterdir()):
        path.with_name(path.name + ".enc").write_bytes(b"encrypted")
    monkeypatch.setattr("jasna.ltx.model_files.is_frozen", lambda: True)
    original = job_fingerprints(config, source, output, metadata, ())
    transformer = config.restoration_model_path / bundle_names(config.ltx_model, fast=config.ltx_fast)[0]
    transformer.write_bytes(b"unused plaintext")
    assert job_fingerprints(config, source, output, metadata, ()) == original
    transformer.with_name(transformer.name + ".enc").write_bytes(b"changed encrypted")
    assert job_fingerprints(config, source, output, metadata, ()) != original


def test_fingerprint_ignores_ltx_files_when_every_range_uses_basicvsrpp(job, tmp_path, monkeypatch):
    config, source, output, metadata = job
    basic = tmp_path / "basic.pth"
    basic.write_bytes(b"weights")
    monkeypatch.setattr("jasna.engine_paths.default_restoration_model_path", lambda name: basic)
    ranges = (SegmentRange(0, 1, SegmentRestoration("basicvsrpp", None)),)
    original = job_fingerprints(config, source, output, metadata, ranges)
    for path in config.restoration_model_path.iterdir():
        path.unlink()
    assert job_fingerprints(config, source, output, metadata, ranges) == original


def test_removed_model_rejects_checkpoint_without_deleting_saved_progress(job):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, ())
    checkpoint.prepare()
    _save_work(checkpoint)
    config.detection_model_path.unlink()
    with pytest.raises(ResumeMismatchError):
        ResumeJob(config, source, output, metadata, ()).prepare()
    assert checkpoint.matches()


def test_trial_fingerprint_ignores_installed_ltx_models(job):
    config, source, output, metadata = job
    config = replace(config, ltx_trial=True)
    original = job_fingerprints(config, source, output, metadata, ())
    for path in config.restoration_model_path.iterdir():
        path.unlink()
    assert job_fingerprints(config, source, output, metadata, ()) == original


@pytest.mark.parametrize("change", ["seed", "model", "settings", "ranges", "input", "version", "output", "metadata"])
def test_fingerprint_detects_changes(job, monkeypatch, change):
    config, source, output, metadata = job
    original = job_fingerprints(config, source, output, metadata, ())
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
    assert job_fingerprints(config, source, output, metadata, ranges) != original


def test_mismatch_preserves_every_file_until_confirmed(job):
    config, source, output, metadata = job
    original = ResumeJob(config, source, output, metadata, ())
    original.prepare()
    latent = original.directory / "ltx" / "000000.0-1-0.reference.pt"
    latent.parent.mkdir()
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
        assert find_resume_output(replace(config, ltx_seed=42), source, output, ()) is None


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
        "fingerprint": job_fingerprints(config, source, output, metadata, ())[1],
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


def test_auto_split_scans_past_a_checkpoint_without_saved_work(job):
    from jasna.media.splice import KeyframeIndex
    from jasna.mosaic_scan import MosaicSamples
    from jasna.session_factory import ltx_auto_split

    config, source, output, metadata = job
    config = replace(config, ltx_auto_split_percent=12.0, vr_mode="off")
    metadata.duration = 6.0
    metadata.num_frames = 180
    metadata.video_width, metadata.video_height = 1920, 1080
    metadata.sample_aspect_ratio = Fraction(1, 1)
    ResumeJob(replace(config, ltx_seed=42), source, output, metadata, ()).prepare()
    samples = MosaicSamples(times=(1.0, 3.0), scores=(0.9, 0.9), sizes=(0.05, 0.3), stride=1.0)
    with (
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=KeyframeIndex((0, 60, 120), Fraction(1,30), 0, 180)),
        patch("jasna.mosaic_scan.scan_mosaic_sizes", return_value=samples) as scan,
    ):
        assert ltx_auto_split(config, MagicMock(), source, output, progress_callback=None) is not None
    scan.assert_called_once()


def test_changed_settings_replace_a_checkpoint_without_saved_work(job):
    config, source, output, metadata = job
    ResumeJob(config, source, output, metadata, ()).prepare()
    changed = ResumeJob(replace(config, ltx_seed=7), source, output, metadata, ())
    changed.prepare()
    assert changed.matches()


@pytest.mark.parametrize("work", ["plans", "latent", "fragment"])
def test_changed_settings_ask_once_work_is_saved(job, work):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, ())
    checkpoint.prepare()
    saved = {
        "plans": checkpoint.directory / "ltx" / "0-1.plans.npz",
        "latent": checkpoint.directory / "ltx" / "000000.0-1-0.final.pt",
        "fragment": checkpoint.directory / "segments" / "0000-raw.nut",
    }[work]
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(b"saved")
    with pytest.raises(ResumeMismatchError):
        ResumeJob(replace(config, ltx_seed=7), source, output, metadata, ()).prepare()


def test_partial_files_are_not_saved_work(job):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, ())
    checkpoint.prepare()
    for partial in (
        checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt.part",
        checkpoint.directory / "segments" / "0000-partial.nut",
    ):
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"unfinished")
    ResumeJob(replace(config, ltx_seed=7), source, output, metadata, ()).prepare()


def test_outputs_with_the_same_name_keep_separate_checkpoints_in_a_working_directory(job, tmp_path):
    config, source, _output, metadata = job
    config = replace(config, working_dir=tmp_path / "scratch")
    first, second = tmp_path / "a" / "clip.mp4", tmp_path / "b" / "clip.mp4"
    assert job_resume.resume_directory(config, first) != job_resume.resume_directory(config, second)
    renamed = first.with_stem("clip (1)")
    ResumeJob(config, source, renamed, metadata, ()).prepare()
    ResumeJob(config, source, second, metadata, ()).prepare()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(config, source, first, ()) == renamed
        assert find_resume_output(config, source, second, ()) == second


@pytest.mark.parametrize("change", [
    {"codec": "av1"}, {"encoder_settings": {"cq": 12}}, {"lut_path": "lut"}, {"sharpen_strength": 0.5},
    {"fmp4": True}, {"max_clip_size": 120}, {"compile_basicvsrpp": False}, {"secondary_restoration": "unet-4x"},
])
def test_output_only_settings_keep_the_latent_fingerprint(job, tmp_path, change):
    config, source, output, metadata = job
    if "lut_path" in change:
        (tmp_path / "lut").write_bytes(b"0 0 0\n")
        change = {"lut_path": str(tmp_path / "lut")}
    job_key, latents = job_fingerprints(config, source, output, metadata, ())
    changed_job, changed_latents = job_fingerprints(replace(config, **change), source, output, metadata, ())
    assert changed_latents == latents
    assert changed_job != job_key


@pytest.mark.parametrize("change", [
    {"ltx_seed": 1}, {"ltx_model": "distilled2"}, {"ltx_window_hop": 104}, {"ltx_large_canvas": False},
    {"detection_score_threshold": 0.5}, {"batch_size": 8}, {"fp16": False}, {"retarget_high_fps": True},
    {"vr_mode": "off"},
])
def test_scan_encode_and_denoise_settings_change_the_latent_fingerprint(job, change):
    config, source, output, metadata = job
    for name in bundle_names("distilled2", fast=False):
        (config.restoration_model_path / name).write_bytes(b"weights")
    latents = job_fingerprints(config, source, output, metadata, ())[1]
    assert job_fingerprints(replace(config, **change), source, output, metadata, ())[1] != latents


def test_output_only_change_keeps_latents_and_redoes_fragments(job):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, ())
    checkpoint.prepare()
    _save_work(checkpoint)
    fragment = checkpoint.directory / "segments" / "0000-raw.nut"
    fragment.parent.mkdir()
    fragment.write_bytes(b"old codec")
    changed = replace(config, codec="av1", encoder_settings={"cq": 30})
    ResumeJob(changed, source, output, metadata, ()).prepare()
    assert (checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt").exists()
    assert not fragment.exists()
    assert ResumeJob(changed, source, output, metadata, ()).matches()
    output.touch()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(changed, source, output, ()) == output
