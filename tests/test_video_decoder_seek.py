"""Test VideoReader seek behavior on sample videos."""
import pytest
import torch
from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import VideoReader

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SAMPLE_VIDEOS = [
    "assets/test_clip1_1080p.mp4",
    "assets/test_clip1_2160p.mp4",
]


def _first_available_video():
    from pathlib import Path
    for v in SAMPLE_VIDEOS:
        if Path(v).exists():
            return v
    pytest.skip("No sample video found")


@pytest.fixture
def video_path():
    return _first_available_video()


@pytest.fixture
def metadata(video_path):
    return get_video_meta_data(video_path)


class TestSeekBehavior:
    def test_seek_pts_are_sequential(self, video_path, metadata):
        """After a seek, PTS values should be sequential (not repeating)."""
        device = torch.device("cuda:0")
        seek_frame = metadata.num_frames // 3
        seek_ts = seek_frame / metadata.video_fps
        all_pts = []
        with VideoReader(video_path, batch_size=24, device=device, metadata=metadata) as reader:
            for batch, pts in reader.frames(seek_ts=seek_ts):
                all_pts.extend(pts)
                if len(all_pts) >= 72:
                    break
        for i in range(1, len(all_pts)):
            assert all_pts[i] > all_pts[i - 1], (
                f"PTS not increasing at index {i}: {all_pts[i-1]} -> {all_pts[i]}"
            )

    def test_two_readers_seek_same_frame(self, video_path, metadata):
        """Two readers seeking to the same frame should produce same PTS."""
        device = torch.device("cuda:0")
        seek_ts = (metadata.num_frames // 2) / metadata.video_fps
        with (
            VideoReader(video_path, batch_size=24, device=device, metadata=metadata) as r1,
            VideoReader(video_path, batch_size=24, device=device, metadata=metadata) as r2,
        ):
            gen1 = r1.frames(seek_ts=seek_ts)
            gen2 = r2.frames(seek_ts=seek_ts)
            pts1 = [pts for _ in range(3) for pts in next(gen1)[1]]
            pts2 = [pts for _ in range(3) for pts in next(gen2)[1]]
        assert pts1 == pts2

    def test_seek_honors_nonzero_stream_start_time(self, video_path, tmp_path):
        """Seek targets must be offset by the stream's start_time, and the first
        returned frame must not precede the requested timestamp."""
        import subprocess
        from jasna.os_utils import resolve_executable

        shifted = tmp_path / "shifted.mp4"
        subprocess.run(
            [resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
             "-i", video_path, "-t", "4", "-c", "copy",
             "-output_ts_offset", "1.5", str(shifted)],
            check=True,
        )
        shifted_meta = get_video_meta_data(str(shifted))
        device = torch.device("cuda:0")
        seek_ts = 2.5
        with VideoReader(str(shifted), batch_size=8, device=device, metadata=shifted_meta) as reader:
            _, pts = next(iter(reader.frames(seek_ts=seek_ts)))

        first_seconds = float(pts[0] * shifted_meta.time_base)
        start_seconds = float(shifted_meta.start_pts * shifted_meta.time_base)
        assert start_seconds >= 1.4, f"fixture lost its start offset ({start_seconds})"
        assert first_seconds >= start_seconds + seek_ts - 0.001, (
            f"first frame at {first_seconds:.3f}s precedes start_time+seek_ts "
            f"({start_seconds + seek_ts:.3f}s)"
        )
        assert first_seconds < start_seconds + seek_ts + 0.5

    @pytest.mark.parametrize("backend", ["auto", "vali", "pyav-hw", "pyav-sw"])
    @pytest.mark.parametrize("seek_ts", [None, 0.0])
    def test_every_backend_reads_exactly_the_shown_frames_of_a_mid_gop_cut(self, tmp_path, backend, seek_ts):
        if backend == "vali" and torch.version.hip is not None:
            pytest.skip("VALI decodes with NVDEC")
        import subprocess

        import av
        from jasna.os_utils import resolve_executable

        full = tmp_path / "full.mp4"
        cut = tmp_path / "cut.mp4"
        ffmpeg = [resolve_executable("ffmpeg"), "-y", "-loglevel", "error"]
        subprocess.run(
            [*ffmpeg, "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=12:duration=4",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "3", "-g", "24", str(full)],
            check=True,
        )
        subprocess.run([*ffmpeg, "-ss", "1.5", "-i", str(full), "-c", "copy", str(cut)], check=True)
        with av.open(str(cut)) as container:
            shown_pts = [frame.pts for frame in container.decode(video=0)]
        metadata = get_video_meta_data(str(cut))
        with VideoReader(
            str(cut), batch_size=8, device=torch.device("cuda:0"), metadata=metadata, decode_backend=backend
        ) as reader:
            pts = [int(p) for _, batch_pts in reader.frames(seek_ts=seek_ts) for p in batch_pts]
        assert pts == shown_pts
