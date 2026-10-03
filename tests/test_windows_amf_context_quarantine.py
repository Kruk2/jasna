from types import SimpleNamespace
from unittest.mock import Mock

import av
import pytest

from jasna.gpu_context_errors import (
    NativeGpuContextUnusableError, native_context_failure,
    is_windows_amf_host_transfer_failure,
)

_MESSAGE = "[AVHWFramesContext] Convert(amf::AMF_MEMORY_HOST) failed with error -1313558101"


@pytest.mark.parametrize("errno", [1313558101, -1313558101])
def test_specific_unknown_transfer_contract_is_quarantined(errno):
    error = av.error.UnknownError(errno, _MESSAGE)
    assert is_windows_amf_host_transfer_failure(error, platform="win32", amd=True, decoder_name="h264_amf")


@pytest.mark.parametrize("platform,amd,decoder,message,errno", [
    ("linux", True, "hevc_amf", _MESSAGE, 1313558101),
    ("win32", False, "h264_amf", _MESSAGE, 1313558101),
    ("win32", True, "h264", _MESSAGE, 1313558101),
    ("win32", True, "h264_amf", "Unknown error occurred", 1313558101),
    ("win32", True, "h264_amf", _MESSAGE, 1094995529),
])
def test_generic_unknown_bad_input_and_other_backends_not_misclassified(platform, amd, decoder, message, errno):
    assert not is_windows_amf_host_transfer_failure(av.error.UnknownError(errno, message),
                platform=platform, amd=amd, decoder_name=decoder)


def test_decoder_raises_typed_fault_with_original_av_cause(monkeypatch):
    from jasna.accelerator import AcceleratorVendor
    from jasna.media import video_decoder as decoder
    error = av.error.UnknownError(1313558101, _MESSAGE)
    reader = SimpleNamespace(file="synthetic.mp4", vendor=AcceleratorVendor.AMD,
                             _decoder_ctx=SimpleNamespace(name="h264_amf", decode=Mock(side_effect=error)))
    monkeypatch.setattr(decoder.sys, "platform", "win32")
    with pytest.raises(NativeGpuContextUnusableError) as caught:
        decoder.VideoReader._decode_packet(reader, object(), 0)
    assert caught.value.__cause__ is error


def test_wrapped_context_fault_is_preserved_and_cycles_terminate():
    fault = NativeGpuContextUnusableError("synthetic transfer failure")
    wrapped = RuntimeError("pipeline reader failed")
    wrapped.__cause__ = fault
    fault.__context__ = wrapped
    assert native_context_failure(wrapped) is fault
    plain = RuntimeError("ordinary media error")
    plain.__cause__ = plain
    assert native_context_failure(plain) is None


def test_actual_processor_stops_before_next_job_and_rejects_same_process_restart(tmp_path, monkeypatch):
    from jasna.gui.models import AppSettings, JobItem, JobStatus
    from jasna.gui.processor import Processor

    monkeypatch.setattr("jasna.gui.processor._cleanup_torch", lambda _: None)
    first, second = [JobItem(path=tmp_path / (name + ".mp4")) for name in ("first", "second")]
    logs, completed = [], []
    processor = Processor(on_log=lambda level, message: logs.append((level, message)), on_complete=completed.append)
    processor._jobs = [first, second]
    processor._settings = AppSettings(pre_scan_policy="off")
    processor._output_folder = str(tmp_path / "output")
    processor._output_pattern = "{original}_restored.mp4"
    pipeline = Mock(side_effect=NativeGpuContextUnusableError("synthetic AMF transfer failure"))
    processor._run_pipeline = pipeline
    processor._run()
    assert first.status is JobStatus.ERROR
    assert second.status is JobStatus.PENDING
    assert pipeline.call_count == 1
    assert completed == [False]
    assert processor._native_context_quarantined
    assert "rebuild decoder, encoder and model resources" in processor.restart_required_reason()
    assert "memory was exhausted" not in processor.restart_required_reason().lower()
    with pytest.raises(RuntimeError, match="restart"):
        processor.start([second], processor._settings, processor._output_folder, processor._output_pattern)


def test_cleanup_failure_does_not_lose_original_fault_or_completion(tmp_path, monkeypatch):
    from jasna.gui.models import AppSettings, JobItem, JobStatus
    from jasna.gui.processor import Processor

    cleanup = Mock()
    monkeypatch.setattr("jasna.gui.processor._cleanup_torch", cleanup)
    done = []
    job = JobItem(path=tmp_path / "synthetic.mp4")
    processor = Processor(on_complete=done.append)
    processor._jobs = [job]
    processor._settings = AppSettings(pre_scan_policy="off")
    processor._output_folder = str(tmp_path)
    processor._output_pattern = "{original}_restored.mp4"
    processor._run_pipeline = Mock(side_effect=NativeGpuContextUnusableError("original native fault"))
    processor._close_video_session = Mock(side_effect=RuntimeError("cleanup after invalid context"))
    processor._run()
    assert job.status is JobStatus.ERROR
    assert done == [False]
    cleanup.assert_not_called()
