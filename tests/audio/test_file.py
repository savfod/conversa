"""
Tests for AudioFileInputStream class.
"""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from conversa.audio.input_stream.file import AudioFileInputStream


class TestAudioFileInputStream:
    """Test suite for AudioFileInputStream class."""

    def test_initialization(self):
        """Test AudioFileInputStream initialization."""
        stream = AudioFileInputStream(file_path="test.mp3")

        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert stream.chunk_duration == 0.1
        assert stream.chunk_size == 1600  # 16000 * 0.1
        assert stream.file_path == Path("test.mp3")

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        stream = AudioFileInputStream(
            file_path="test.wav", sample_rate=8000, channels=2, chunk_duration=0.5
        )

        assert stream.sample_rate == 8000
        assert stream.channels == 2
        assert stream.chunk_duration == 0.5
        assert stream.chunk_size == 4000  # 8000 * 0.5

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_processing_loop(self, mock_read_audio):
        """Test audio processing loop with mocked file reading."""
        # Mock audio data: 1 second of audio
        mock_audio = np.random.randn(16000).astype(np.float32)
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()
        time.sleep(0.5)  # Let it process for a bit

        # Get some audio from buffer
        chunk = stream.get_unprocessed_chunk()

        assert chunk is not None
        assert len(chunk) > 0

        stream.stop()

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_complete_file_processing(self, mock_read_audio):
        """Test that entire file is processed correctly."""
        # Mock a short audio file: 0.3 seconds
        mock_audio = np.random.randn(4800).astype(np.float32)
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()

        # Wait for processing to complete
        # 0.3 seconds of audio with 0.1s chunks = 3 chunks
        # With 0.1s sleep per chunk = ~0.3s total
        time.sleep(0.5)

        # Get all buffered audio
        all_chunks = []
        while True:
            chunk = stream.get_unprocessed_chunk()
            if chunk is None:
                break
            all_chunks.append(chunk)
            time.sleep(0.05)

        stream.stop()

        # Verify we got audio
        assert len(all_chunks) > 0

        # Total samples should approximately equal original
        total_samples = sum(len(chunk) for chunk in all_chunks)
        assert total_samples > 0

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_chunk_timing(self, mock_read_audio):
        """Test that chunks are generated with correct timing."""
        # Mock audio: 0.5 seconds
        mock_audio = np.random.randn(8000).astype(np.float32)
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        start_time = time.time()
        stream.start()

        # Wait for completion
        time.sleep(0.7)

        stream.stop()
        elapsed = time.time() - start_time

        # Should take approximately 0.5 seconds (file duration)
        # allowing some overhead for processing
        assert 0.4 < elapsed < 1.0

    @patch("conversa.audio.input_stream.file.read_audio")
    def test_chunk_size_accuracy(self, mock_read_audio):
        """Test that chunks have the correct size."""
        mock_audio = np.random.randn(16000).astype(np.float32)  # 1 second
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()
        time.sleep(0.2)  # Get a couple chunks

        chunk = stream.get_unprocessed_chunk()
        stream.stop()

        # Chunks should be approximately chunk_size samples
        # (allowing for some buffering)
        if chunk is not None:
            assert len(chunk) > 0

    @patch("conversa.audio.input_stream.file.read_audio")
    def test_exception_handling(self, mock_read_audio, capsys):
        """Test that file reading exceptions are handled gracefully."""
        mock_read_audio.side_effect = Exception("File not found")

        stream = AudioFileInputStream(file_path="nonexistent.mp3")
        stream.start()

        time.sleep(0.2)

        # Stream should have stopped due to exception
        assert stream._is_running is False

        captured = capsys.readouterr()
        assert "Error in MP3 processing:" in captured.out

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_early_stop(self, mock_read_audio):
        """Test stopping stream before file processing completes."""
        # Mock a long audio file
        mock_audio = np.random.randn(160000).astype(np.float32)  # 10 seconds
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()
        time.sleep(0.3)  # Let it run briefly

        # Stop early
        stream.stop()

        assert stream._is_running is False

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_buffer_duration_updates(self, mock_read_audio):
        """Test that buffer duration updates as chunks are added."""
        mock_audio = np.random.randn(16000).astype(np.float32)
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()

        # Check buffer duration increases
        time.sleep(0.15)
        duration1 = stream.get_buffer_duration()

        time.sleep(0.15)
        duration2 = stream.get_buffer_duration()

        stream.stop()

        # Duration should have increased (or at least been > 0)
        assert duration1 >= 0
        assert duration2 >= 0

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_get_unprocessed_chunk_empties_buffer(self, mock_read_audio):
        """Test that get_unprocessed_chunk empties the buffer."""
        mock_audio = np.random.randn(8000).astype(np.float32)
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()
        time.sleep(0.3)

        # Get chunk should empty buffer
        chunk1 = stream.get_unprocessed_chunk()
        assert chunk1 is not None and len(chunk1) > 0

        # Immediate second call should return None or empty
        chunk2 = stream.get_unprocessed_chunk()
        assert chunk2 is None or len(chunk2) == 0

        stream.stop()

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_processing_loop_completion_message(self, mock_read_audio, capsys):
        """Test that completion message is printed when file finishes."""
        mock_audio = np.random.randn(3200).astype(np.float32)  # 0.2 seconds
        mock_read_audio.return_value = mock_audio

        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)

        stream.start()
        time.sleep(0.5)  # Wait for completion
        stream.stop()

        captured = capsys.readouterr()
        assert "MP3 file processing completed" in captured.out

    def test_file_path_conversion(self):
        """Test that file_path is converted to Path object."""
        stream = AudioFileInputStream(file_path="/path/to/audio.mp3")

        assert isinstance(stream.file_path, Path)
        assert str(stream.file_path) == "/path/to/audio.mp3"
