import os
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from factories import session_config
from jasna import job_resume
from jasna.job_resume import ResumeJob, ResumeMismatchError, changed_inputs, fingerprints, job_inputs
from jasna.media.splice import KeyframeIndex, build_splice_plan
from jasna.os_utils import atomic_write
from jasna.session_factory import find_resume_output
from jasna.ltx.camera import CropFrame
from jasna.ltx.model_files import bundle_names
from jasna.ltx.plan import MAX_GAP, TrackPlan, Window, plans_from_arrays, plans_to_arrays
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


def _plan(segments):
    """The plan of ``segments`` in the 4 s fixture video, None for a whole-video job."""
    if not segments:
        return None
    return build_splice_plan(segments, KeyframeIndex((0, 30, 60, 90), Fraction(1, 30), 0, 120), duration=4.0)


def _save_work(checkpoint: ResumeJob) -> None:
    (checkpoint.directory / "ltx").mkdir(parents=True, exist_ok=True)
    (checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt").write_bytes(b"latent")


def test_plans_round_trip_preserves_crops_float64_polygons_and_weights(tmp_path):
    crop = CropFrame((1, 2, 640, 480), (384, 512), (64, 64, 0, 0))
    integer = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    floating = integer.astype(np.float64) + 0.1234567890123456
    plans = [TrackPlan(7, 12, ((integer,), (floating, integer)), (0.1, 1.0), (Window(3, 7, 12, 2, (crop, crop), 3),), 104)]
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


def job_fingerprints(config, source, output, metadata, splice_plan):
    return fingerprints(job_inputs(config, source, output, metadata, splice_plan))


def test_fingerprint_ignores_progress_working_directory_and_decoder(job, tmp_path):
    config, source, output, metadata = job
    changed = replace(config, disable_progress=True, working_dir=tmp_path / "scratch", decode_backend="pyav-sw")
    assert job_fingerprints(config, source, output, metadata, None) == job_fingerprints(changed, source, output, metadata, None)


def test_changed_lut_contents_redo_the_output_even_with_same_size_and_timestamp(job, tmp_path):
    config, source, output, metadata = job
    lut = tmp_path / "colors.cube"
    lut.write_bytes(b"0 0 0\n1 1 1\n")
    config = replace(config, lut_path=str(lut))
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    _save_work(checkpoint)
    stat = lut.stat()
    lut.write_bytes(b"1 1 1\n0 0 0\n")
    os.utime(lut, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = ResumeJob(config, source, output, metadata, None)
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
    checkpoint = ResumeJob(config, source, output, metadata, _plan(ranges))
    checkpoint.prepare()
    _save_work(checkpoint)
    stat = path.stat()
    path.write_bytes(b"updated")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    changed = ResumeJob(config, source, output, metadata, _plan(ranges))
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
    original = job_fingerprints(config, source, output, metadata, None)
    transformer = config.restoration_model_path / bundle_names(config.ltx_model, fast=config.ltx_fast)[0]
    transformer.write_bytes(b"unused plaintext")
    assert job_fingerprints(config, source, output, metadata, None) == original
    transformer.with_name(transformer.name + ".enc").write_bytes(b"changed encrypted")
    assert job_fingerprints(config, source, output, metadata, None) != original


def test_fingerprint_ignores_ltx_files_when_every_range_uses_basicvsrpp(job, tmp_path, monkeypatch):
    config, source, output, metadata = job
    basic = tmp_path / "basic.pth"
    basic.write_bytes(b"weights")
    monkeypatch.setattr("jasna.engine_paths.default_restoration_model_path", lambda name: basic)
    ranges = (SegmentRange(0, 1, SegmentRestoration("basicvsrpp", None)),)
    original = job_fingerprints(config, source, output, metadata, _plan(ranges))
    for path in config.restoration_model_path.iterdir():
        path.unlink()
    assert job_fingerprints(config, source, output, metadata, _plan(ranges)) == original


def test_removed_model_rejects_checkpoint_without_deleting_saved_progress(job):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    _save_work(checkpoint)
    config.detection_model_path.unlink()
    with pytest.raises(ResumeMismatchError):
        ResumeJob(config, source, output, metadata, None).prepare()
    assert checkpoint.matches()


def test_trial_fingerprint_ignores_installed_ltx_models(job):
    config, source, output, metadata = job
    config = replace(config, ltx_trial=True)
    original = job_fingerprints(config, source, output, metadata, None)
    for path in config.restoration_model_path.iterdir():
        path.unlink()
    assert job_fingerprints(config, source, output, metadata, None) == original


@pytest.mark.parametrize("change", ["seed", "model", "settings", "ranges", "input", "version", "output", "metadata"])
def test_fingerprint_detects_changes(job, monkeypatch, change):
    config, source, output, metadata = job
    original = job_fingerprints(config, source, output, metadata, None)
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
    assert job_fingerprints(config, source, output, metadata, _plan(ranges)) != original


def test_another_ltx_encoder_changes_the_saved_latents_key(job):
    config, source, output, metadata = job
    (config.model_path("ltx") / "vae-encoder-half.safetensors").write_bytes(b"half encoder")
    latents = job_fingerprints(config, source, output, metadata, None)[1]

    assert job_fingerprints(replace(config, ltx_vae_encoder="half"), source, output, metadata, None)[1] != latents


def test_latents_follow_the_detection_gap_only_where_ltx_tracks_use_it(job):
    config, source, output, metadata = job
    job_key, latents = job_fingerprints(config, source, output, metadata, None)
    for gap, keeps_latents in ((0, True), (MAX_GAP, True), (MAX_GAP + 3, False)):
        changed = job_fingerprints(replace(config, max_detection_gap=gap), source, output, metadata, None)
        assert changed[0] != job_key
        assert (changed[1] == latents) is keeps_latents


@pytest.mark.parametrize("saved, current, expected", [
    ({"a": 1, "b": {"c": 2, "d": [1, 2]}}, {"a": 1, "b": {"c": 3, "d": [1, 2]}}, ["b.c: 2 -> 3"]),
    ({"a": 1}, {"a": 1, "b": "new"}, ["b: None -> new"]),
    ({"a": 1}, {"a": 1}, []),
    (None, {"a": 1}, ["unknown (saved by an older Jasna)"]),
])
def test_changed_inputs_names_each_difference(saved, current, expected):
    assert changed_inputs(saved, current) == expected


def test_mismatch_log_names_the_changed_model_file(job, caplog):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    _save_work(checkpoint)
    stat = config.detection_model_path.stat()
    os.utime(config.detection_model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ResumeMismatchError), caplog.at_level("WARNING", logger="jasna.job_resume"):
        ResumeJob(config, source, output, metadata, None).prepare()
    assert "changed: latents.models.detection: " in caplog.text
    assert "latents.settings" not in caplog.text


def test_mismatch_preserves_every_file_until_confirmed(job):
    config, source, output, metadata = job
    original = ResumeJob(config, source, output, metadata, None)
    original.prepare()
    latent = original.directory / "ltx" / "000000.0-1-0.reference.pt"
    latent.parent.mkdir()
    latent.write_bytes(b"saved")
    changed = ResumeJob(replace(config, ltx_seed=7), source, output, metadata, None)
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
    checkpoint = ResumeJob(config, source, saved, metadata, None)
    checkpoint.prepare()
    saved.touch()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(config, source, output, ()) == saved
        assert find_resume_output(replace(config, ltx_seed=42), source, output, ()) is None


def test_resume_lookup_handles_brackets_and_only_follows_actual_auto_rename_names(job):
    config, source, output, metadata = job
    output = output.with_stem("[group] output")
    unrelated = output.with_stem(output.stem + " other")
    ResumeJob(config, source, unrelated, metadata, None).prepare()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(config, source, output, ()) is None
        saved = output.with_stem(output.stem + " (1)")
        ResumeJob(config, source, saved, metadata, None).prepare()
        assert find_resume_output(config, source, output, ()) == saved


def _video_metadata(metadata):
    metadata.video_width, metadata.video_height = 1920, 1080
    metadata.sample_aspect_ratio = Fraction(1, 1)
    return metadata


def _splice_patches(metadata):
    return (
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=KeyframeIndex((0, 30, 60, 90), Fraction(1, 30), 0, 120)),
    )


def test_find_auto_split_checkpoint_with_effective_codec_and_ranges(job):
    from jasna.session_factory import segment_job_config

    config, source, output, metadata = job
    metadata = _video_metadata(metadata)
    config = replace(config, ltx_auto_split_percent=12.0, codec="av1", vr_mode="off")
    segments = (SegmentRange(1, 2, SegmentRestoration("ltx", 0)),)
    ResumeJob(config, source, output, metadata, None).save_split(segments)
    checkpoint = ResumeJob(segment_job_config(config, "hevc", source), source, output, metadata, _plan(segments))
    checkpoint.prepare()
    validate, metadata_patch, keyframes = _splice_patches(metadata)
    with validate, metadata_patch, keyframes:
        assert find_resume_output(config, source, output, ()) == output
        assert find_resume_output(replace(config, ltx_seed=3), source, output, ()) is None


def test_find_ranges_checkpoint_with_effective_codec(job):
    from jasna.session_factory import segment_job_config

    config, source, output, metadata = job
    config = replace(config, codec="av1")
    segments = (SegmentRange(1, 2),)
    ResumeJob(segment_job_config(config, "hevc", source), source, output, metadata, _plan(segments)).prepare()
    validate, metadata_patch, keyframes = _splice_patches(metadata)
    with validate, metadata_patch, keyframes:
        assert find_resume_output(config, source, output, segments) == output
        assert find_resume_output(config, source, output, (SegmentRange(1, 3),)) is None


def test_auto_split_that_could_not_split_finds_its_whole_video_checkpoint(job):
    from jasna.media.splice import SmartRenderCompatibilityError

    config, source, output, metadata = job
    config = replace(config, ltx_auto_split_percent=12.0, vr_mode="off")
    metadata = _video_metadata(metadata)
    ResumeJob(config, source, output, metadata, None).prepare()
    with (
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render", side_effect=SmartRenderCompatibilityError("VFR")),
    ):
        assert find_resume_output(config, source, output, ()) == output


@pytest.mark.parametrize("answer", ["yes", "no", None])
def test_cli_asks_before_replacing_progress(job, answer):
    from jasna.job_resume import run_with_resume_prompt
    from jasna.main import _ask_to_replace_progress

    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    error = ResumeMismatchError(checkpoint.root)
    run = MagicMock(side_effect=[error, "done"])
    with patch("sys.stdin.isatty", return_value=answer is not None), patch("builtins.input", return_value=answer) as prompt:
        if answer == "yes":
            assert run_with_resume_prompt(run, _ask_to_replace_progress) == "done"
            assert run.call_count == 2
            assert not checkpoint.root.exists()
        else:
            with pytest.raises(ResumeMismatchError):
                run_with_resume_prompt(run, _ask_to_replace_progress)
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
        checkpoint = ResumeJob(first.config, source, output, metadata, first.splice_plan)
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
    ResumeJob(replace(config, ltx_seed=42), source, output, metadata, None).prepare()
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
    ResumeJob(config, source, output, metadata, None).prepare()
    changed = ResumeJob(replace(config, ltx_seed=7), source, output, metadata, None)
    changed.prepare()
    assert changed.matches()


@pytest.mark.parametrize("work", ["plans", "latent", "fragment"])
def test_changed_settings_ask_once_work_is_saved(job, work):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    saved = {
        "plans": checkpoint.directory / "ltx" / "0-1.plans.npz",
        "latent": checkpoint.directory / "ltx" / "000000.0-1-0.final.pt",
        "fragment": checkpoint.directory / "segments" / "0000-raw.nut",
    }[work]
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(b"saved")
    with pytest.raises(ResumeMismatchError):
        ResumeJob(replace(config, ltx_seed=7), source, output, metadata, None).prepare()


def test_partial_files_are_not_saved_work(job):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    partials = (
        checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt.part",
        checkpoint.directory / "segments" / "0000-partial.nut",
    )
    for partial in partials:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"unfinished")

    ResumeJob(replace(config, ltx_seed=7), source, output, metadata, None).prepare()

    assert not any(partial.exists() for partial in partials)


def test_outputs_with_the_same_name_keep_separate_checkpoints_in_a_working_directory(job, tmp_path):
    config, source, _output, metadata = job
    config = replace(config, working_dir=tmp_path / "scratch")
    first, second = tmp_path / "a" / "clip.mp4", tmp_path / "b" / "clip.mp4"
    assert job_resume.resume_directory(config, first) != job_resume.resume_directory(config, second)
    renamed = first.with_stem("clip (1)")
    ResumeJob(config, source, renamed, metadata, None).prepare()
    ResumeJob(config, source, second, metadata, None).prepare()
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
    job_key, latents = job_fingerprints(config, source, output, metadata, None)
    changed_job, changed_latents = job_fingerprints(replace(config, **change), source, output, metadata, None)
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
    latents = job_fingerprints(config, source, output, metadata, None)[1]
    assert job_fingerprints(replace(config, **change), source, output, metadata, None)[1] != latents


def test_output_only_change_keeps_latents_and_redoes_fragments(job):
    config, source, output, metadata = job
    checkpoint = ResumeJob(config, source, output, metadata, None)
    checkpoint.prepare()
    _save_work(checkpoint)
    fragment = checkpoint.directory / "segments" / "0000-raw.nut"
    fragment.parent.mkdir()
    fragment.write_bytes(b"old codec")
    changed = replace(config, codec="av1", encoder_settings={"cq": 30})
    ResumeJob(changed, source, output, metadata, None).prepare()
    assert (checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt").exists()
    assert not fragment.exists()
    assert ResumeJob(changed, source, output, metadata, None).matches()
    output.touch()
    with patch("jasna.media.probe.get_video_meta_data", return_value=metadata):
        assert find_resume_output(changed, source, output, ()) == output


def test_output_only_change_asks_before_redoing_fragments_without_ltx_work(job):
    config, source, output, metadata = job
    config = replace(config, restoration_model_name="basicvsrpp")
    plan = _plan((SegmentRange(1, 2),))
    checkpoint = ResumeJob(config, source, output, metadata, plan)
    checkpoint.prepare()
    fragment = checkpoint.directory / "segments" / "0000-raw.nut"
    fragment.parent.mkdir()
    fragment.write_bytes(b"restored")
    with pytest.raises(ResumeMismatchError):
        ResumeJob(replace(config, codec="av1"), source, output, metadata, plan).prepare()
    assert fragment.read_bytes() == b"restored"


def test_changed_plan_keeps_latents_and_redoes_fragments(job):
    config, source, output, metadata = job
    segments = (SegmentRange(1, 2),)
    plan = _plan(segments)
    checkpoint = ResumeJob(config, source, output, metadata, plan)
    checkpoint.prepare()
    _save_work(checkpoint)
    fragment = checkpoint.directory / "segments" / "0000-raw.nut"
    fragment.parent.mkdir()
    fragment.write_bytes(b"old plan")
    replanned = build_splice_plan(segments, KeyframeIndex((0, 45, 90), Fraction(1, 30), 0, 120), duration=4.0)
    changed = ResumeJob(config, source, output, metadata, replanned)
    assert changed.fingerprint != checkpoint.fingerprint
    changed.prepare()
    assert (checkpoint.directory / "ltx" / "000000.0-1-0.reference.pt").exists()
    assert not fragment.exists()
