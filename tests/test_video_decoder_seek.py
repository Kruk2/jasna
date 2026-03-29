"""Test NvidiaVideoReader seek behavior on sample videos."""
import time
import pytest
import torch
from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader

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
    def test_sequential_read_speed(self, video_path, metadata):
        """Baseline: read first 5 batches sequentially from start."""
        device = torch.device("cuda:0")
        with NvidiaVideoReader(video_path, batch_size=24, device=device, metadata=metadata) as reader:
            t0 = time.monotonic()
            frames_read = 0
            for batch, pts in reader.frames():
                frames_read += len(pts)
                if frames_read >= 120:
                    break
            elapsed = time.monotonic() - t0
            fps = frames_read / elapsed
            print(f"\nSequential: {frames_read} frames in {elapsed:.2f}s = {fps:.0f} fps")
            assert fps > 30, f"Sequential read too slow: {fps:.0f} fps"

    def test_seek_then_read_speed(self, video_path, metadata):
        """Seek to mid-video then read 5 batches."""
        device = torch.device("cuda:0")
        seek_frame = metadata.num_frames // 2
        seek_ts = seek_frame / metadata.video_fps
        with NvidiaVideoReader(video_path, batch_size=24, device=device, metadata=metadata) as reader:
            t0 = time.monotonic()
            frames_read = 0
            first_batch_time = None
            for batch, pts in reader.frames(seek_ts=seek_ts):
                if first_batch_time is None:
                    first_batch_time = time.monotonic() - t0
                frames_read += len(pts)
                if frames_read >= 120:
                    break
            elapsed = time.monotonic() - t0
            fps = frames_read / elapsed
            print(f"\nSeek to {seek_frame}: first batch {first_batch_time:.2f}s, "
                  f"{frames_read} frames in {elapsed:.2f}s = {fps:.0f} fps")
            assert fps > 30, f"Seek+read too slow: {fps:.0f} fps"

    def test_seek_pts_are_sequential(self, video_path, metadata):
        """After a seek, PTS values should be sequential (not repeating)."""
        device = torch.device("cuda:0")
        seek_frame = metadata.num_frames // 3
        seek_ts = seek_frame / metadata.video_fps
        all_pts = []
        with NvidiaVideoReader(video_path, batch_size=24, device=device, metadata=metadata) as reader:
            for batch, pts in reader.frames(seek_ts=seek_ts):
                all_pts.extend(pts)
                if len(all_pts) >= 72:
                    break
        print(f"\nPTS after seek to frame {seek_frame}: {all_pts[:10]}...")
        for i in range(1, len(all_pts)):
            assert all_pts[i] > all_pts[i - 1], (
                f"PTS not increasing at index {i}: {all_pts[i-1]} -> {all_pts[i]}"
            )

    def test_two_readers_seek_same_frame(self, video_path, metadata):
        """Two readers seeking to the same frame should produce same PTS."""
        device = torch.device("cuda:0")
        seek_frame = metadata.num_frames // 2
        with (
            NvidiaVideoReader(video_path, batch_size=24, device=device, metadata=metadata) as r1,
            NvidiaVideoReader(video_path, batch_size=24, device=device, metadata=metadata) as r2,
        ):
            t0 = time.monotonic()
            frames1 = 0
            frames2 = 0
            pts1 = []
            pts2 = []
            seek_ts = seek_frame / metadata.video_fps
            gen1 = r1.frames(seek_ts=seek_ts)
            gen2 = r2.frames(seek_ts=seek_ts)
            for _ in range(3):
                b1, p1 = next(gen1)
                pts1.extend(p1)
                frames1 += len(p1)
                b2, p2 = next(gen2)
                pts2.extend(p2)
                frames2 += len(p2)
            elapsed = time.monotonic() - t0
            combined_fps = (frames1 + frames2) / elapsed
            print(f"\nTwo readers seek to {seek_frame}: {frames1}+{frames2} frames in {elapsed:.2f}s = {combined_fps:.0f} fps combined")
            assert pts1 == pts2, f"PTS mismatch between readers"

    def test_seek_does_not_repeat_on_subsequent_batches(self, video_path, metadata):
        """Verify that batch 2+ after a seek continues forward, not re-seeking."""
        device = torch.device("cuda:0")
        seek_frame = metadata.num_frames // 2
        seek_ts = seek_frame / metadata.video_fps
        batch_times = []
        with NvidiaVideoReader(video_path, batch_size=24, device=device, metadata=metadata) as reader:
            for batch, pts in reader.frames(seek_ts=seek_ts):
                batch_times.append(time.monotonic())
                if len(batch_times) >= 5:
                    break

        intervals = [batch_times[i] - batch_times[i-1] for i in range(1, len(batch_times))]
        first_interval = batch_times[0]  # includes seek
        print(f"\nBatch intervals after seek: {[f'{i:.3f}s' for i in intervals]}")
        avg_interval = sum(intervals) / len(intervals)
        assert avg_interval < 0.5, (
            f"Subsequent batches too slow ({avg_interval:.2f}s avg), "
            f"suggests re-seeking on each batch"
        )
