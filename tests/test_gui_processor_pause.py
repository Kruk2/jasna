"""Unit tests for Processor pause timing logic."""

import time
import threading
from unittest.mock import Mock, patch
import pytest

from jasna.gui.processor import Processor, ProgressUpdate, JobStatus


class TestProcessorPauseTiming:
    """Test pause timing logic in Processor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Processor()
        self.mock_progress = Mock()
        self.mock_log = Mock()
        self.mock_complete = Mock()
        
        self.processor._on_progress = self.mock_progress
        self.processor._on_log = self.mock_log
        self.processor._on_complete = self.mock_complete
        
        # Mock job data
        self.jobs = [
            Mock(path=Mock(stem="test_video", parent=Mock()), filename="test_video.mp4")
        ]
        self.settings = Mock()
        self.settings.file_conflict = "auto_rename"
        self.settings.working_directory = ""
        
    def test_pause_timing_basic(self):
        """Test basic pause and resume timing."""
        # Mock time.monotonic for deterministic testing
        mock_time = Mock()
        mock_time.side_effect = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
        
        with patch('jasna.gui.processor.time.monotonic', mock_time):
            # Start processor
            self.processor.start(
                jobs=self.jobs,
                settings=self.settings,
                output_folder="output",
                output_pattern="{original}_restored.mp4",
                disable_basicvsrpp_tensorrt=False,
            )
            
            # Pause at 0.1s
            self.processor.pause()
            
            # Resume at 0.2s (paused for 0.1s)
            self.processor.pause()
            
            # Stop at 0.3s (ran for 0.1s total)
            self.processor.stop()
            self.processor.join(timeout=1.0)
            
            # Check timing
            real_time = self.processor.get_real_processing_time()
            
            # Should be 0.1s (total run time) - 0.1s (pause time) = 0.0s
            # But since we started timing at 0.0, it should be 0.1s
            assert real_time == 0.1
        
    def test_pause_timing_multiple_pauses(self):
        """Test timing with multiple pause/resume cycles."""
        self.processor.start(
            jobs=self.jobs,
            settings=self.settings,
            output_folder="output",
            output_pattern="{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
        )
        
        # First pause/resume cycle
        self.processor.pause()
        time.sleep(0.05)
        self.processor.pause()
        
        # Run a bit
        time.sleep(0.05)
        
        # Second pause/resume cycle
        self.processor.pause()
        time.sleep(0.03)
        self.processor.pause()
        
        # Run a bit more
        time.sleep(0.05)
        
        self.processor.stop()
        self.processor.join(timeout=1.0)
        
        real_time = self.processor.get_real_processing_time()
        
        # Should exclude both pause durations
        assert real_time > 0
        assert real_time < 0.2  # Less than total wall clock time
        
    def test_pause_timing_reset_between_runs(self):
        """Test that timing is reset between processing runs."""
        # First run
        self.processor.start(
            jobs=self.jobs,
            settings=self.settings,
            output_folder="output",
            output_pattern="{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
        )
        
        self.processor.pause()
        time.sleep(0.05)
        self.processor.pause()
        
        self.processor.stop()
        self.processor.join(timeout=1.0)
        
        first_run_time = self.processor.get_real_processing_time()
        
        # Second run - should start fresh
        self.processor.start(
            jobs=self.jobs,
            settings=self.settings,
            output_folder="output",
            output_pattern="{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
        )
        
        time.sleep(0.05)
        self.processor.stop()
        self.processor.join(timeout=1.0)
        
        second_run_time = self.processor.get_real_processing_time()
        
        # Second run should not include first run's pause time
        assert second_run_time < first_run_time
        
    def test_pause_state_tracking(self):
        """Test pause state is tracked correctly."""
        assert not self.processor.is_paused()
        
        self.processor.pause()
        assert self.processor.is_paused()
        
        self.processor.pause()
        assert not self.processor.is_paused()
        
        self.processor.pause()
        assert self.processor.is_paused()
        
    def test_pause_timing_no_double_counting(self):
        """Test that pause time is not double-counted between pause() and progress_callback."""
        # Mock the progress callback to simulate pipeline progress
        progress_calls = []
        
        def mock_progress_callback(progress_pct, fps, eta_seconds, frames_done, total):
            progress_calls.append(time.monotonic())
            # Simulate processing work
            time.sleep(0.01)
            
        # Mock the pipeline to use our progress callback
        with patch('jasna.gui.processor.Pipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock pipeline.run to call our progress callback
            def run_with_progress():
                for i in range(10):
                    mock_progress_callback(10 * i, 30.0, 10.0, i * 100, 1000)
                    time.sleep(0.01)
                    
            mock_pipeline.run.side_effect = run_with_progress
            
            self.processor.start(
                jobs=self.jobs,
                settings=self.settings,
                output_folder="output",
                output_pattern="{original}_restored.mp4",
                disable_basicvsrpp_tensorrt=False,
            )
            
            # Pause during processing
            self.processor.pause()
            time.sleep(0.05)
            self.processor.pause()
            
            self.processor.stop()
            self.processor.join(timeout=1.0)
            
        # Check that real processing time is reasonable
        real_time = self.processor.get_real_processing_time()
        
        # Should be roughly the time spent in progress calls minus pause time
        assert real_time > 0
        assert real_time < 0.2  # Should exclude the 0.05s pause
        
    def test_get_real_processing_time_before_start(self):
        """Test get_real_processing_time before processor is started."""
        assert self.processor.get_real_processing_time() == 0.0
        
    def test_pause_timing_with_no_pauses(self):
        """Test timing when no pauses occur."""
        self.processor.start(
            jobs=self.jobs,
            settings=self.settings,
            output_folder="output",
            output_pattern="{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
        )
        
        time.sleep(0.1)
        
        self.processor.stop()
        self.processor.join(timeout=1.0)
        
        real_time = self.processor.get_real_processing_time()
        
        # Should be close to the actual sleep time
        assert real_time > 0.08  # Allow some tolerance
        assert real_time < 0.15
        
    def test_pause_timing_precision(self):
        """Test that pause timing is reasonably precise."""
        self.processor.start(
            jobs=self.jobs,
            settings=self.settings,
            output_folder="output",
            output_pattern="{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
        )
        
        # Pause for exactly 100ms
        self.processor.pause()
        time.sleep(0.1)
        self.processor.pause()
        
        # Run for 50ms
        time.sleep(0.05)
        
        self.processor.stop()
        self.processor.join(timeout=1.0)
        
        real_time = self.processor.get_real_processing_time()
        
        # Should be close to 50ms (excluding the 100ms pause)
        assert real_time > 0.04
        assert real_time < 0.07


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
