"""
Tests for AbstractAudioInputStream base class.
"""

import time

import numpy as np
import pytest

from conversa.audio.input_stream.base import AbstractAudioInputStream


class ConcreteAudioInputStream(AbstractAudioInputStream):
    """Concrete implementation for testing AbstractAudioInputStream."""

    def __init__(
        self, sample_rate: int = 16000, channels: int = 1, test_duration: float = 0.5
    ):
        """
        Initialize concrete test stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            test_duration: Duration of test audio to generate in seconds
        """
        super().__init__(sample_rate, channels)
        self.test_duration = test_duration
        self.processing_started = False
        self.processing_completed = False

    def _audio_processing_loop(self) -> None:
        """Test implementation that generates synthetic audio."""
        self.processing_started = True
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        total_samples = int(self.sample_rate * self.test_duration)
        generated_samples = 0

        while self._is_running and generated_samples < total_samples:
            chunk = np.random.randn(chunk_size).astype(np.float32)
            self._buffer.add_audio(chunk)
            generated_samples += chunk_size
            time.sleep(0.05)  # Simulate some processing time

        self.processing_completed = True


class TestAbstractAudioInputStream:
    """Test suite for AbstractAudioInputStream base class."""

    def test_initialization(self):
        """Test base class initialization."""
        stream = ConcreteAudioInputStream()

        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert stream._is_running is False
        assert stream._thread is None
        assert stream.get_buffer_duration() == 0.0

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        stream = ConcreteAudioInputStream(sample_rate=8000, channels=2)

        assert stream.sample_rate == 8000
        assert stream.channels == 2

    def test_start(self):
        """Test starting the audio stream."""
        stream = ConcreteAudioInputStream(test_duration=0.3)

        stream.start()

        assert stream._is_running is True
        assert stream._thread is not None
        assert stream._thread.is_alive()

        # Wait a bit for processing to start
        time.sleep(0.1)
        assert stream.processing_started is True

        stream.stop()

    def test_stop(self):
        """Test stopping the audio stream."""
        stream = ConcreteAudioInputStream(test_duration=1.0)

        stream.start()
        time.sleep(0.2)  # Let it run for a bit

        stream.stop()

        assert stream._is_running is False
        # Thread should have been joined
        if stream._thread:
            assert not stream._thread.is_alive()

    def test_double_start_ignored(self):
        """Test that calling start twice doesn't create multiple threads."""
        stream = ConcreteAudioInputStream(test_duration=0.5)

        stream.start()
        first_thread = stream._thread

        stream.start()  # Second start should be ignored

        assert stream._thread is first_thread

        stream.stop()

    def test_get_unprocessed_chunk(self):
        """Test retrieving audio chunks from the stream."""
        stream = ConcreteAudioInputStream(test_duration=0.3)

        stream.start()
        time.sleep(0.2)  # Wait for some audio to be buffered

        chunk = stream.get_unprocessed_chunk()

        assert chunk is not None
        assert isinstance(chunk, np.ndarray)
        assert len(chunk) > 0

        stream.stop()

    def test_get_unprocessed_chunk_when_empty(self):
        """Test get_unprocessed_chunk returns None when buffer is empty."""
        stream = ConcreteAudioInputStream(test_duration=0.1)

        # Don't start the stream, so buffer remains empty
        chunk = stream.get_unprocessed_chunk()

        assert chunk is None

    def test_get_buffer_duration(self):
        """Test getting buffer duration."""
        stream = ConcreteAudioInputStream(test_duration=0.5)

        stream.start()
        time.sleep(0.2)  # Wait for some buffering

        duration = stream.get_buffer_duration()

        assert duration > 0.0
        assert isinstance(duration, float)

        stream.stop()

    @pytest.mark.slow
    def test_processing_loop_runs_until_stopped(self):
        """Test that processing loop respects the _is_running flag."""
        stream = ConcreteAudioInputStream(test_duration=10.0)  # Long duration

        stream.start()
        time.sleep(0.2)

        assert stream._is_running is True
        assert stream.processing_started is True
        assert stream.processing_completed is False  # Should still be running

        stream.stop()
        time.sleep(0.1)  # Give thread time to complete

        assert stream._is_running is False

    def test_daemon_thread(self):
        """Test that the processing thread is a daemon thread."""
        stream = ConcreteAudioInputStream(test_duration=0.5)

        stream.start()

        assert stream._thread.daemon is True

        stream.stop()

    @pytest.mark.slow
    def test_complete_workflow(self):
        """Test complete workflow: start, process, retrieve chunks, stop."""
        stream = ConcreteAudioInputStream(test_duration=0.5)

        # Start stream
        stream.start()
        assert stream._is_running is True

        # Collect chunks
        chunks = []
        for _ in range(3):
            time.sleep(0.15)
            chunk = stream.get_unprocessed_chunk()
            if chunk is not None:
                chunks.append(chunk)

        # Stop stream
        stream.stop()
        assert stream._is_running is False

        # Verify we collected some audio
        assert len(chunks) > 0
        total_samples = sum(len(chunk) for chunk in chunks)
        assert total_samples > 0

    def test_stop_before_start(self):
        """Test that stopping before starting doesn't cause errors."""
        stream = ConcreteAudioInputStream()

        # Should not raise an exception
        stream.stop()

        assert stream._is_running is False

    def test_abstract_method_enforcement(self):
        """Test that AbstractAudioInputStream cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractAudioInputStream()
