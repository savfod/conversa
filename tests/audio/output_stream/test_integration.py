"""
Integration tests for output_stream package.

These tests verify that all components work together correctly
and that the public API is properly exposed.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from conversa.generated.output_stream import (
    AbstractAudioOutputStream,
    FileOutputStream,
    SpeakerOutputStream,
)


class TestPackageImports:
    """Test that all classes are properly importable from the package."""

    def test_all_classes_importable(self):
        """Test that __all__ exports are accessible."""
        assert AbstractAudioOutputStream is not None
        assert SpeakerOutputStream is not None
        assert FileOutputStream is not None

    def test_speaker_stream_instantiation(self):
        """Test that SpeakerOutputStream can be instantiated."""
        stream = SpeakerOutputStream()
        assert stream is not None
        assert isinstance(stream, AbstractAudioOutputStream)

    def test_file_stream_instantiation(self):
        """Test that FileOutputStream can be instantiated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)
            assert stream is not None
            assert isinstance(stream, AbstractAudioOutputStream)


class TestIntegration:
    """Integration tests for the complete output_stream system."""

    def test_file_stream_complete_workflow(self):
        """Test complete workflow for file output stream."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "integration_test.wav"
            stream = FileOutputStream(output_path=output_path, sample_rate=16000)

            # Generate test audio
            chunks = []
            for i in range(3):
                chunk = np.sin(
                    2 * np.pi * 440 * (i + 1) * np.arange(8000) / 16000
                ).astype(np.float32)
                chunks.append(chunk)
                stream.play_chunk(chunk)

            # Verify buffering
            assert stream.get_chunk_count() == 3
            assert stream.get_total_duration() == pytest.approx(1.5, 0.01)

            # Stop and save
            stream.wait()

            # Verify file exists and has correct content
            assert output_path.exists()
            loaded_audio, loaded_sr = sf.read(output_path)
            assert loaded_sr == 16000
            assert len(loaded_audio) == 24000  # 3 chunks of 8000 samples

    @pytest.mark.slow
    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_speaker_stream_complete_workflow_slow(self, mock_output_stream):
        """Test complete workflow for speaker output stream."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        # Capture the callback
        callback_ref = {}

        # We need to capture the callback passed to the CONSTRUCTOR of the stream
        # However, mock_output_stream IS the class/constructor mock.
        # So when SpeakerOutputStream calls sd.OutputStream(...), it calls this mock.

        def side_effect(*args, **kwargs):
            if "callback" in kwargs:
                callback_ref["cb"] = kwargs["callback"]
                callback_ref["blocksize"] = kwargs.get("blocksize", 1600)
                callback_ref["status"] = kwargs.get("status", None)
            return mock_stream_instance

        mock_output_stream.side_effect = side_effect

        stream = SpeakerOutputStream(sample_rate=16000, channels=1)

        # Start a thread to simulate the callback
        stop_event = threading.Event()

        def simulate_hardware():
            while not stop_event.is_set():
                if "cb" in callback_ref:
                    # Create dummy buffer
                    blocksize = callback_ref["blocksize"]
                    outdata = np.zeros((blocksize, 1), dtype=np.float32)
                    # Call the callback
                    callback_ref["cb"](outdata, blocksize, None, None)
                time.sleep(0.01)  # fast enough simulation

        sim_thread = threading.Thread(target=simulate_hardware, daemon=True)
        sim_thread.start()

        try:
            # Play some chunks
            for _ in range(3):
                audio_data = np.random.randn(1600).astype(np.float32)
                stream.play_chunk(audio_data)

            # Wait for playback
            # This should now succeed because sim_thread is draining the queue
            stream.wait()

            # Verify stream was started
            assert mock_stream_instance.start.called

        finally:
            stop_event.set()
            sim_thread.join(timeout=1.0)
            stream.stop()

        assert mock_stream_instance.stop.called

    def test_polymorphism(self):
        """Test that streams can be used polymorphically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            streams = [
                FileOutputStream(output_path=output_path),
                SpeakerOutputStream(),
            ]

            for stream in streams:
                assert isinstance(stream, AbstractAudioOutputStream)
                assert hasattr(stream, "play_chunk")
                assert hasattr(stream, "stop")
                assert hasattr(stream, "wait")

    def test_multiple_streams_independent(self):
        """Test that multiple streams operate independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path1 = Path(tmpdir) / "test1.wav"
            output_path2 = Path(tmpdir) / "test2.wav"

            stream1 = FileOutputStream(output_path=output_path1)
            stream2 = FileOutputStream(output_path=output_path2)

            # Play different audio to each stream (use values in proper float32 range)
            audio1 = np.ones(1600, dtype=np.float32) * 0.1
            audio2 = np.ones(1600, dtype=np.float32) * 0.2

            stream1.play_chunk(audio1)
            stream2.play_chunk(audio2)

            stream1.stop()
            stream2.stop()

            # Verify files are different
            loaded1, _ = sf.read(output_path1)
            loaded2, _ = sf.read(output_path2)

            assert np.allclose(loaded1, 0.1, rtol=1e-3, atol=1e-3)
            assert np.allclose(loaded2, 0.2, rtol=1e-3, atol=1e-3)

    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_mixed_stream_types_independent(self, mock_output_stream):
        """Test that different stream types work independently."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            # Create both types of streams
            file_stream = FileOutputStream(output_path=output_path)
            sound_stream = SpeakerOutputStream()

            # Play audio to both
            audio_data = np.random.randn(1600).astype(np.float32)

            file_stream.play_chunk(audio_data)
            sound_stream.play_chunk(audio_data)

            # Clean up
            file_stream.stop()
            time.sleep(0.1)
            sound_stream.stop()

            # Verify file was created
            assert output_path.exists()

            # Verify speaker played audio (started)
            assert mock_stream_instance.start.called

    def test_api_consistency_across_implementations(self):
        """Test that all implementations have consistent API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            streams = [
                FileOutputStream(output_path=output_path),
                SpeakerOutputStream(),
            ]

            required_methods = ["play_chunk", "stop", "wait"]
            required_attributes = ["sample_rate", "channels"]

            for stream in streams:
                for method in required_methods:
                    assert hasattr(stream, method)
                    assert callable(getattr(stream, method))

                for attr in required_attributes:
                    assert hasattr(stream, attr)

    def test_file_stream_handles_different_sample_rates(self):
        """Test file stream with various sample rates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_rates = [8000, 16000, 22050, 44100, 48000]

            for sr in sample_rates:
                output_path = Path(tmpdir) / f"test_{sr}.wav"
                stream = FileOutputStream(output_path=output_path, sample_rate=sr)

                # Generate 1 second of audio
                audio_data = np.random.randn(sr).astype(np.float32)
                stream.play_chunk(audio_data)
                stream.stop()

                # Verify file
                loaded_audio, loaded_sr = sf.read(output_path)
                assert loaded_sr == sr
                assert len(loaded_audio) == sr

    @pytest.mark.slow
    @patch("conversa.generated.output_stream.speaker.sd.OutputStream")
    def test_speaker_stream_handles_different_sample_rates(self, mock_output_stream):
        """Test speaker stream with various sample rates."""
        sample_rates = [8000, 16000, 44100]

        for sr in sample_rates:
            mock_stream_instance = MagicMock()
            mock_output_stream.return_value = mock_stream_instance

            stream = SpeakerOutputStream(sample_rate=sr)
            audio_data = np.random.randn(sr).astype(np.float32)

            stream.play_chunk(audio_data)
            time.sleep(0.1)
            stream.stop()

            # Verify OutputStream was created with correct sample rate
            mock_output_stream.assert_called()
            call_args = mock_output_stream.call_args
            assert call_args[1]["samplerate"] == sr

            # Reset mock for next iteration
            mock_output_stream.reset_mock()
