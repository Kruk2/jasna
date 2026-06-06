"""Unit test for NvidiaVideoReader._handoff cross-stream synchronization.

The decoder produces each batch on its private (non-blocking) CUDA stream and
hands it to consumers running on the default stream. _handoff is the event
handshake that (a) orders the consumer stream after the decode and (b) calls
record_stream so the caching allocator will not recycle the batch buffer while
the consumer is still reading it. Mirrors the encoder's sync tests.
"""
from unittest.mock import MagicMock, patch

import torch

from jasna.media.video_decoder import NvidiaVideoReader


def _make_reader() -> NvidiaVideoReader:
    # __init__ touches no GPU; stream is normally created in __enter__.
    reader = NvidiaVideoReader(
        file="dummy.mp4",
        batch_size=4,
        device=torch.device("cuda:0"),
        metadata=object(),
    )
    reader.stream = MagicMock()
    return reader


def test_handoff_orders_consumer_stream_and_records_buffer() -> None:
    reader = _make_reader()
    fake_event = object()
    fake_consumer_stream = MagicMock()
    batch_tensor = MagicMock()

    with (
        patch("jasna.media.video_decoder.torch.cuda.Event", return_value=fake_event),
        patch(
            "jasna.media.video_decoder.torch.cuda.current_stream",
            return_value=fake_consumer_stream,
        ) as mock_current_stream,
    ):
        reader._handoff(batch_tensor)

    # decode completion captured on the private decode stream
    reader.stream.record_event.assert_called_once_with(fake_event)
    # consumer (default) stream waits for the decode before reading the batch
    mock_current_stream.assert_called_once_with(reader.device)
    fake_consumer_stream.wait_event.assert_called_once_with(fake_event)
    # allocator must not recycle the batch buffer until the consumer is done
    batch_tensor.record_stream.assert_called_once_with(fake_consumer_stream)
