"""
Tests for AudioBuffer class.
"""

import time

import numpy as np
import pytest

from conversa.audio.input_stream.buffer import AudioBuffer


class TestAudioBuffer:
    """Test suite for AudioBuffer class."""

    def test_initialization(self):
        """Test AudioBuffer initialization with default parameters."""
        buffer = AudioBuffer()
        assert buffer.sample_rate == 16000
        assert buffer.max_duration_seconds == 60.0
        assert buffer.max_samples == 16000 * 60
        assert buffer.get_duration() == 0.0

    def test_initialization_custom_parameters(self):
        """Test AudioBuffer initialization with custom parameters."""
        buffer = AudioBuffer(max_duration_seconds=10.0, sample_rate=8000)
        assert buffer.sample_rate == 8000
        assert buffer.max_duration_seconds == 10.0
        assert buffer.max_samples == 8000 * 10
        assert buffer.get_duration() == 0.0

    def test_add_audio_mono(self):
        """Test adding mono audio data to buffer."""
        buffer = AudioBuffer(sample_rate=16000)
        audio_data = np.random.randn(1600).astype(np.float32)  # 0.1 seconds

        buffer.add_audio(audio_data)

        duration = buffer.get_duration()
        assert pytest.approx(duration, 0.01) == 0.1

    def test_add_audio_stereo(self):
        """Test adding stereo audio data to buffer (should be flattened)."""
        buffer = AudioBuffer(sample_rate=16000)
        audio_data = np.random.randn(1600, 2).astype(np.float32)  # 0.1 seconds stereo

        buffer.add_audio(audio_data)

        # Should be flattened to 3200 samples
        duration = buffer.get_duration()
        assert pytest.approx(duration, 0.01) == 0.2

    def test_get_and_clear(self):
        """Test retrieving and clearing buffer."""
        buffer = AudioBuffer(sample_rate=16000)
        audio_data = np.random.randn(1600).astype(np.float32)

        buffer.add_audio(audio_data)
        assert buffer.get_duration() > 0

        retrieved = buffer.get_and_clear()
        assert len(retrieved) == 1600
        assert buffer.get_duration() == 0.0

        # Second call should return empty array
        retrieved = buffer.get_and_clear()
        assert len(retrieved) == 0

    def test_get_and_clear_empty_buffer(self):
        """Test get_and_clear on empty buffer."""
        buffer = AudioBuffer()

        result = buffer.get_and_clear()

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_buffer_trimming(self):
        """Test automatic buffer trimming when exceeding max size."""
        buffer = AudioBuffer(max_duration_seconds=1.0, sample_rate=16000)

        # Add more audio than max_samples
        chunk_size = 8000  # 0.5 seconds
        for _ in range(5):  # Total 2.5 seconds, exceeds 1.0 second limit
            buffer.add_audio(np.random.randn(chunk_size).astype(np.float32))

        # Buffer should have been trimmed to max_duration_seconds
        duration = buffer.get_duration()
        assert duration < 2.5  # Should be less than total added
        assert duration <= buffer.max_duration_seconds

    def test_thread_safety(self):
        """Test thread-safe operations on buffer."""
        import threading

        buffer = AudioBuffer(sample_rate=16000)
        errors = []

        def add_audio_thread():
            try:
                for _ in range(100):
                    buffer.add_audio(np.random.randn(160).astype(np.float32))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def read_audio_thread():
            try:
                for _ in range(50):
                    buffer.get_and_clear()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_audio_thread),
            threading.Thread(target=add_audio_thread),
            threading.Thread(target=read_audio_thread),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors occurred: {errors}"

    def test_multiple_adds_before_clear(self):
        """Test adding multiple chunks before clearing."""
        buffer = AudioBuffer(sample_rate=16000)

        chunk1 = np.ones(1600, dtype=np.float32)
        chunk2 = np.ones(1600, dtype=np.float32) * 2

        buffer.add_audio(chunk1)
        buffer.add_audio(chunk2)

        duration = buffer.get_duration()
        assert pytest.approx(duration, 0.01) == 0.2

        retrieved = buffer.get_and_clear()
        assert len(retrieved) == 3200

    def test_get_duration_accuracy(self):
        """Test that get_duration returns accurate values."""
        buffer = AudioBuffer(sample_rate=16000)

        # Add exactly 1 second of audio
        one_second = np.random.randn(16000).astype(np.float32)
        buffer.add_audio(one_second)

        duration = buffer.get_duration()
        assert pytest.approx(duration, 0.001) == 1.0
