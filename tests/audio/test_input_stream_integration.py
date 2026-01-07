"""
Integration tests for input_stream package.

These tests verify that all components work together correctly
and that the public API maintains backward compatibility.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from conversa.audio.input_stream import (
    AbstractAudioInputStream,
    AudioBuffer,
    AudioFileInputStream,
    MicrophoneInputStream,
)


class TestPackageImports:
    """Test that all classes are properly importable from the package."""

    def test_all_classes_importable(self):
        """Test that __all__ exports are accessible."""
        assert AudioBuffer is not None
        assert AbstractAudioInputStream is not None
        assert MicrophoneInputStream is not None
        assert AudioFileInputStream is not None

    def test_audio_buffer_instantiation(self):
        """Test that AudioBuffer can be instantiated from package import."""
        buffer = AudioBuffer()
        assert buffer is not None
        assert buffer.sample_rate == 16000

    def test_microphone_stream_instantiation(self):
        """Test that MicrophoneInputStream can be instantiated."""
        stream = MicrophoneInputStream()
        assert stream is not None
        assert isinstance(stream, AbstractAudioInputStream)

    def test_file_stream_instantiation(self):
        """Test that AudioFileInputStream can be instantiated."""
        stream = AudioFileInputStream(file_path="test.mp3")
        assert stream is not None
        assert isinstance(stream, AbstractAudioInputStream)


class TestBackwardCompatibility:
    """Test that the new structure maintains backward compatibility."""

    def test_import_from_conversa_audio_input_stream(self):
        """Test legacy import path still works."""
        from conversa.audio.input_stream import (
            AudioFileInputStream,
            MicrophoneInputStream,
        )

        assert AudioFileInputStream is not None
        assert MicrophoneInputStream is not None

    def test_old_api_still_works(self):
        """Test that the old API interface still functions."""
        stream = AudioFileInputStream(
            file_path="test.mp3", sample_rate=16000, channels=1, chunk_duration=0.1
        )

        # Old API methods should work
        assert hasattr(stream, "start")
        assert hasattr(stream, "stop")
        assert hasattr(stream, "get_unprocessed_chunk")
        assert hasattr(stream, "get_buffer_duration")


class TestIntegration:
    """Integration tests for the complete input_stream system."""

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_file_stream_to_buffer_flow(self, mock_read_audio):
        """Test complete flow from file to buffer retrieval."""
        # Setup mock audio
        mock_audio = np.random.randn(16000).astype(np.float32)  # 1 second
        mock_read_audio.return_value = mock_audio

        # Create and start stream
        stream = AudioFileInputStream(file_path="test.mp3", chunk_duration=0.1)
        stream.start()

        # Wait and collect chunks
        time.sleep(0.3)

        chunks_collected = []
        for _ in range(3):
            chunk = stream.get_unprocessed_chunk()
            if chunk is not None:
                chunks_collected.append(chunk)
            time.sleep(0.1)

        stream.stop()

        # Verify we collected audio
        assert len(chunks_collected) > 0
        total_samples = sum(len(c) for c in chunks_collected)
        assert total_samples > 0

    @patch("conversa.audio.input_stream.microphone.sd.InputStream")
    def test_microphone_stream_buffer_flow(self, mock_input_stream):
        """Test complete flow from microphone to buffer retrieval."""
        from unittest.mock import MagicMock

        mock_stream_instance = MagicMock()
        mock_input_stream.return_value.__enter__.return_value = mock_stream_instance

        stream = MicrophoneInputStream(sample_rate=16000)

        # Manually add some test audio to simulate microphone input
        test_audio = np.random.randn(1600).astype(np.float32)
        stream._buffer.add_audio(test_audio)

        # Retrieve chunk
        chunk = stream.get_unprocessed_chunk()

        assert chunk is not None
        assert len(chunk) == 1600

    def test_buffer_shared_across_base_class(self):
        """Test that buffer is properly managed by base class."""
        stream = AudioFileInputStream(file_path="test.mp3")

        # Access buffer through base class interface
        assert stream._buffer is not None
        assert isinstance(stream._buffer, AudioBuffer)

        # Add audio directly to buffer
        test_audio = np.random.randn(1600).astype(np.float32)
        stream._buffer.add_audio(test_audio)

        # Should be retrievable through stream interface
        chunk = stream.get_unprocessed_chunk()
        assert chunk is not None
        assert len(chunk) == 1600

    @pytest.mark.slow
    @patch("conversa.audio.input_stream.file.read_audio")
    def test_multiple_streams_independent(self, mock_read_audio):
        """Test that multiple streams operate independently."""
        mock_audio = np.random.randn(8000).astype(np.float32)
        mock_read_audio.return_value = mock_audio

        stream1 = AudioFileInputStream(file_path="test1.mp3", chunk_duration=0.1)
        stream2 = AudioFileInputStream(file_path="test2.mp3", chunk_duration=0.2)

        stream1.start()
        stream2.start()

        time.sleep(0.3)

        # Each stream should have independent buffers
        chunk1 = stream1.get_unprocessed_chunk()
        chunk2 = stream2.get_unprocessed_chunk()

        stream1.stop()
        stream2.stop()

        # Both should have collected audio independently
        assert chunk1 is not None or chunk2 is not None

    def test_polymorphism(self):
        """Test that streams can be used polymorphically."""
        streams = [
            AudioFileInputStream(file_path="test.mp3"),
            MicrophoneInputStream(),
        ]

        for stream in streams:
            assert isinstance(stream, AbstractAudioInputStream)
            assert hasattr(stream, "start")
            assert hasattr(stream, "stop")
            assert hasattr(stream, "get_unprocessed_chunk")
            assert hasattr(stream, "get_buffer_duration")
