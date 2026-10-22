"""AMD decode and encode on private streams while another thread churns the
default stream's allocator, as the restorer does (issue #252 showed up as
whole frames of foreign activations)."""
from __future__ import annotations

import threading

import av
import numpy as np
import pytest
import torch

from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import VideoReader
from jasna.media.video_encoder import VideoEncoder

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and {"h264_amf", "hevc_amf"} <= set(av.codecs_available)
    ),
    reason="needs an AMD GPU with the AMF codecs",
)

WIDTH, HEIGHT, FRAMES = 640, 368, 96


def _write_source(path) -> None:
    # Smooth moving luma and flat chroma survive 4:2:0 coding, so a clean round
    # trip stays far above the threshold and a foreign-memory frame falls far below.
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "10"}
        for i in range(FRAMES):
            luma = 128 + 80 * np.sin((x + 6 * i) / 23.0) * np.cos((y - 3 * i) / 31.0)
            planes = np.full((HEIGHT * 3 // 2, WIDTH), 128, dtype=np.uint8)
            planes[:HEIGHT] = luma.astype(np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(planes, format="yuv420p")))
        container.mux(stream.encode(None))


def _decode_rgb(path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _churn_default_stream(stop: threading.Event, device: torch.device) -> None:
    torch.cuda.set_device(device)
    while not stop.is_set():
        for _ in range(8):
            torch.full((3, HEIGHT, WIDTH), 0x3C, dtype=torch.uint8, device=device).mul_(3)
        torch.cuda.current_stream(device).synchronize()


def test_amd_decode_encode_round_trip_survives_default_stream_churn(tmp_path, monkeypatch):
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "auto")
    device = torch.device("cuda:0")
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _write_source(source)
    metadata = get_video_meta_data(str(source))

    stop = threading.Event()
    churn = threading.Thread(target=_churn_default_stream, args=(stop, device), daemon=True)
    churn.start()
    try:
        with (
            VideoReader(str(source), 4, device, metadata) as reader,
            VideoEncoder(
                str(output), device, metadata, codec="hevc", encoder_settings={"cq": 18}
            ) as encoder,
        ):
            assert encoder.stream != torch.cuda.current_stream(device)
            for batch, pts in reader.frames():
                for frame, frame_pts in zip(batch, pts):
                    encoder.encode(frame.clone(), frame_pts)
    finally:
        stop.set()
        churn.join()

    expected = _decode_rgb(source)
    restored = _decode_rgb(output)
    assert len(restored) == FRAMES
    worst = min(_psnr(a, b) for a, b in zip(expected, restored))
    assert worst > 35, f"worst frame PSNR {worst:.1f} dB"
