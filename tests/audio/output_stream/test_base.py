"""
Tests for AbstractAudioOutputStream base class.
"""

import numpy as np
import pytest

from conversa.generated.output_stream.base import AbstractAudioOutputStream


class ConcreteAudioOutputStream(AbstractAudioOutputStream):
    """Concrete implementation for testing AbstractAudioOutputStream."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize concrete test stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        super().__init__(sample_rate, channels)
        self.played_chunks = []
        self.is_stopped = False
        self.wait_called = False

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Test implementation that stores played chunks."""
        self.played_chunks.append(audio_data.copy())

    def stop(self) -> None:
        """Test implementation that sets stop flag."""
        self.is_stopped = True

    def wait(self) -> None:
        """Test implementation that sets wait flag."""
        self.wait_called = True

    def is_playing(self) -> bool:
        """Test implementation of is_playing."""
        return not self.is_stopped


class TestAbstractAudioOutputStream:
    """Test suite for AbstractAudioOutputStream base class."""

    def test_initialization(self):
        """Test base class initialization."""
        stream = ConcreteAudioOutputStream()

        assert stream.sample_rate == 16000
        assert stream.channels == 1

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        stream = ConcreteAudioOutputStream(sample_rate=8000, channels=2)

        assert stream.sample_rate == 8000
        assert stream.channels == 2

    def test_play_chunk(self):
        """Test playing audio chunks."""
        stream = ConcreteAudioOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)

        assert len(stream.played_chunks) == 1
        assert np.array_equal(stream.played_chunks[0], audio_data)

    def test_multiple_play_chunks(self):
        """Test playing multiple audio chunks."""
        stream = ConcreteAudioOutputStream()

        for i in range(3):
            audio_data = np.ones(1600) * i
            stream.play_chunk(audio_data)

        assert len(stream.played_chunks) == 3

    def test_stop(self):
        """Test stopping the stream."""
        stream = ConcreteAudioOutputStream()

        stream.stop()

        assert stream.is_stopped is True

    def test_wait(self):
        """Test waiting for playback to finish."""
        stream = ConcreteAudioOutputStream()

        stream.wait()

        assert stream.wait_called is True

    def test_complete_workflow(self):
        """Test complete workflow: play, wait, stop."""
        stream = ConcreteAudioOutputStream()

        # Play some chunks
        for i in range(3):
            audio_data = np.random.randn(800).astype(np.float32)
            stream.play_chunk(audio_data)

        # Wait for completion
        stream.wait()
        assert stream.wait_called is True

        # Stop stream
        stream.stop()
        assert stream.is_stopped is True

        # Verify chunks were played
        assert len(stream.played_chunks) == 3

    def test_abstract_method_enforcement(self):
        """Test that AbstractAudioOutputStream cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractAudioOutputStream()

    def test_play_chunk_with_different_dtypes(self):
        """Test playing chunks with different data types."""
        stream = ConcreteAudioOutputStream()

        # Test with float32
        audio_float32 = np.random.randn(1600).astype(np.float32)
        stream.play_chunk(audio_float32)

        # Test with float64
        audio_float64 = np.random.randn(1600).astype(np.float64)
        stream.play_chunk(audio_float64)

        # Test with int16
        audio_int16 = (np.random.randn(1600) * 32767).astype(np.int16)
        stream.play_chunk(audio_int16)

        assert len(stream.played_chunks) == 3

    def test_play_chunk_with_different_shapes(self):
        """Test playing chunks with different array shapes."""
        stream = ConcreteAudioOutputStream()

        # Mono (1D)
        audio_mono = np.random.randn(1600).astype(np.float32)
        stream.play_chunk(audio_mono)

        # Mono (2D with shape (n, 1))
        audio_mono_2d = np.random.randn(1600, 1).astype(np.float32)
        stream.play_chunk(audio_mono_2d)

        assert len(stream.played_chunks) == 2
