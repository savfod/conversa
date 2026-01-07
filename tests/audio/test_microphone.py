"""
Tests for MicrophoneInputStream class.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conversa.audio.input_stream.microphone import MicrophoneInputStream


class TestMicrophoneInputStream:
    """Test suite for MicrophoneInputStream class."""

    def test_initialization(self):
        """Test MicrophoneInputStream initialization with default parameters."""
        stream = MicrophoneInputStream()

        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert stream.block_size == 1024
        assert stream.device is None
        assert stream._stream is None

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        stream = MicrophoneInputStream(
            sample_rate=8000, channels=2, block_size=512, device=1
        )

        assert stream.sample_rate == 8000
        assert stream.channels == 2
        assert stream.block_size == 512
        assert stream.device == 1

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_start_and_stop(self, mock_input_stream_class):
        """Test starting and stopping the microphone stream."""
        # Setup mock
        mock_stream_instance = MagicMock()
        mock_input_stream_class.return_value.__enter__.return_value = (
            mock_stream_instance
        )

        stream = MicrophoneInputStream()

        # Start the stream
        stream.start()
        time.sleep(0.1)  # Give thread time to start

        assert stream._is_running is True
        assert stream._thread is not None

        # Stop the stream
        stream.stop()

        assert stream._is_running is False

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_audio_callback(self, mock_input_stream_class):
        """Test that audio callback adds data to buffer."""
        stream = MicrophoneInputStream()

        # Simulate audio callback
        test_audio = np.random.randn(1024, 1).astype(np.float32)
        stream._audio_callback(test_audio, 1024, None, None)

        # Check that audio was added to buffer
        duration = stream.get_buffer_duration()
        expected_duration = 1024 / 16000
        assert pytest.approx(duration, 0.01) == expected_duration

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_audio_callback_with_status(self, mock_input_stream_class, capsys):
        """Test audio callback handling status messages."""
        stream = MicrophoneInputStream()

        test_audio = np.random.randn(1024, 1).astype(np.float32)
        status = MagicMock()
        status.__str__ = lambda self: "Input overflow"

        stream._audio_callback(test_audio, 1024, None, status)

        captured = capsys.readouterr()
        assert "Audio callback status:" in captured.out

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_processing_loop_calls_sounddevice(self, mock_input_stream_class):
        """Test that processing loop properly uses sounddevice.InputStream."""
        mock_stream_instance = MagicMock()
        mock_input_stream_class.return_value.__enter__.return_value = (
            mock_stream_instance
        )

        stream = MicrophoneInputStream(
            sample_rate=8000, channels=2, block_size=512, device=3
        )

        stream.start()
        time.sleep(0.2)
        stream.stop()

        # Verify InputStream was called with correct parameters
        mock_input_stream_class.assert_called_once()
        call_kwargs = mock_input_stream_class.call_args[1]
        assert call_kwargs["samplerate"] == 8000
        assert call_kwargs["channels"] == 2
        assert call_kwargs["blocksize"] == 512
        assert call_kwargs["device"] == 3
        assert call_kwargs["dtype"] == np.float32

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_exception_handling_in_processing_loop(
        self, mock_input_stream_class, capsys
    ):
        """Test that exceptions in processing loop are handled gracefully."""
        # Make the InputStream raise an exception
        mock_input_stream_class.return_value.__enter__.side_effect = Exception(
            "Test error"
        )

        stream = MicrophoneInputStream()
        stream.start()
        time.sleep(0.2)

        # Stream should have stopped due to exception
        assert stream._is_running is False

        captured = capsys.readouterr()
        assert "Error in microphone processing:" in captured.out

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_get_unprocessed_chunk(self, mock_input_stream_class):
        """Test retrieving audio chunks from microphone stream."""
        stream = MicrophoneInputStream()

        # Add some test audio directly to buffer
        test_audio = np.random.randn(1600).astype(np.float32)
        stream._buffer.add_audio(test_audio)

        chunk = stream.get_unprocessed_chunk()

        assert chunk is not None
        assert len(chunk) == 1600

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_multiple_callbacks(self, mock_input_stream_class):
        """Test multiple audio callbacks accumulate in buffer."""
        stream = MicrophoneInputStream()

        # Simulate multiple callbacks
        for _ in range(3):
            test_audio = np.random.randn(512, 1).astype(np.float32)
            stream._audio_callback(test_audio, 512, None, None)

        # Total duration should be 3 * 512 / 16000
        duration = stream.get_buffer_duration()
        expected_duration = (3 * 512) / 16000
        assert pytest.approx(duration, 0.01) == expected_duration

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_callback_stereo_audio(self, mock_input_stream_class):
        """Test callback with stereo audio (should be flattened)."""
        stream = MicrophoneInputStream(channels=2)

        # Stereo audio: shape (frames, 2)
        test_audio = np.random.randn(1024, 2).astype(np.float32)
        stream._audio_callback(test_audio, 1024, None, None)

        # Buffer should contain flattened audio
        chunk = stream.get_unprocessed_chunk()
        assert len(chunk) == 2048  # 1024 frames * 2 channels
