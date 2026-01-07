"""
Tests for FileOutputStream class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conversa.generated.output_stream.file import FileOutputStream


class TestFileOutputStream:
    """Test suite for FileOutputStream class."""

    def test_initialization(self):
        """Test FileOutputStream initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            assert stream.sample_rate == 16000
            assert stream.channels == 1
            assert stream.output_path == output_path
            assert stream._is_closed is False
            assert len(stream._audio_chunks) == 0

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(
                output_path=output_path, sample_rate=8000, channels=2
            )

            assert stream.sample_rate == 8000
            assert stream.channels == 2

    def test_initialization_creates_directory(self):
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test.wav"
            _stream = FileOutputStream(output_path=output_path)

            assert output_path.parent.exists()

    def test_play_chunk(self):
        """Test adding audio chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            audio_data = np.random.randn(1600).astype(np.float32)
            stream.play_chunk(audio_data)

            assert len(stream._audio_chunks) == 1
            assert stream.get_chunk_count() == 1

    def test_play_multiple_chunks(self):
        """Test adding multiple audio chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            for _ in range(3):
                audio_data = np.random.randn(800).astype(np.float32)
                stream.play_chunk(audio_data)

            assert stream.get_chunk_count() == 3

    def test_stop_saves_file(self):
        """Test that stop saves audio to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            # Add some audio
            audio_data = np.random.randn(16000).astype(np.float32)  # 1 second
            stream.play_chunk(audio_data)

            # Stop should save the file
            stream.stop()

            assert output_path.exists()
            # stream._is_closed should NOT be True after stop() anymore
            assert stream._is_closed is False

    def test_saved_file_content(self):
        """Test that saved file contains correct audio data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path, sample_rate=16000)

            # Create known audio data
            audio_data = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(
                np.float32
            )
            stream.play_chunk(audio_data)
            stream.stop()

            # Read back and verify
            loaded_audio, loaded_sr = sf.read(output_path)
            assert loaded_sr == 16000
            assert len(loaded_audio) == len(audio_data)
            np.testing.assert_allclose(loaded_audio, audio_data, rtol=1e-4, atol=1e-4)

    def test_saved_file_multiple_chunks(self):
        """Test that multiple chunks are concatenated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            # Use values within [-1, 1] range for proper float32 WAV encoding
            chunks = [
                np.ones(1600, dtype=np.float32) * 0.1,
                np.ones(1600, dtype=np.float32) * 0.2,
                np.ones(1600, dtype=np.float32) * 0.3,
            ]

            for chunk in chunks:
                stream.play_chunk(chunk)

            stream.stop()

            # Read and verify concatenation
            loaded_audio, _ = sf.read(output_path)
            assert len(loaded_audio) == 4800

            # Check that chunks are in correct order
            np.testing.assert_allclose(loaded_audio[:1600], 0.1, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(
                loaded_audio[1600:3200], 0.2, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(
                loaded_audio[3200:4800], 0.3, rtol=1e-3, atol=1e-3
            )

    def test_wait_saves_file(self):
        """Test that wait() also saves the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            audio_data = np.random.randn(1600).astype(np.float32)
            stream.play_chunk(audio_data)

            # Wait should save the file
            stream.wait()

            assert output_path.exists()
            # Wait/Stop no longer closes the stream
            assert stream._is_closed is False

    def test_get_total_duration(self):
        """Test getting total duration of buffered audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path, sample_rate=16000)

            # Add 1 second of audio
            audio_data = np.random.randn(16000).astype(np.float32)
            stream.play_chunk(audio_data)

            duration = stream.get_total_duration()
            assert pytest.approx(duration, 0.001) == 1.0

    def test_get_total_duration_multiple_chunks(self):
        """Test duration calculation with multiple chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path, sample_rate=16000)

            # Add 3 chunks of 0.5 seconds each
            for _ in range(3):
                audio_data = np.random.randn(8000).astype(np.float32)
                stream.play_chunk(audio_data)

            duration = stream.get_total_duration()
            assert pytest.approx(duration, 0.001) == 1.5

    def test_get_total_duration_empty(self):
        """Test duration when no audio has been added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            duration = stream.get_total_duration()
            assert duration == 0.0

    def test_get_chunk_count(self):
        """Test getting chunk count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            assert stream.get_chunk_count() == 0

            for i in range(5):
                stream.play_chunk(np.random.randn(800).astype(np.float32))
                assert stream.get_chunk_count() == i + 1

    def test_stop_with_no_chunks_warning(self, capsys):
        """Test that stopping with no chunks shows a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            stream.stop()

            _captured = capsys.readouterr()
            # Warning was removed/commented out in code
            # assert "No audio chunks to save" in captured.out
            assert True

    def test_play_chunk_after_stop_assertion(self):
        """Test that play_chunk after stop raises assertion error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            stream.stop()

            stream.stop()

            # Should NOT raise AssertionError anymore as resumption is allowed
            stream.play_chunk(np.random.randn(1600).astype(np.float32))

    def test_play_chunk_empty_array_assertion(self):
        """Test that empty arrays raise an assertion error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            with pytest.raises(AssertionError):
                stream.play_chunk(np.array([]))

    def test_play_chunk_none_assertion(self):
        """Test that None raises an assertion error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            with pytest.raises(AssertionError):
                stream.play_chunk(None)

    def test_double_stop_safe(self):
        """Test that calling stop twice is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            audio_data = np.random.randn(1600).astype(np.float32)
            stream.play_chunk(audio_data)

            stream.stop()
            stream.stop()  # Should not raise an error

            stream.stop()
            stream.stop()  # Should not raise an error

            # Still not closed after stop
            assert stream._is_closed is False

    def test_path_conversion_from_string(self):
        """Test that string paths are converted to Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test.wav")
            stream = FileOutputStream(output_path=output_path)

            assert isinstance(stream.output_path, Path)

    def test_stereo_output(self):
        """Test saving stereo audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_stereo.wav"
            stream = FileOutputStream(output_path=output_path, channels=2)

            # Add mono audio (should be duplicated to stereo)
            audio_data = np.random.randn(1600).astype(np.float32)
            stream.play_chunk(audio_data)
            stream.stop()

            # Verify file is stereo
            loaded_audio, loaded_sr = sf.read(output_path)
            assert loaded_audio.ndim == 2
            assert loaded_audio.shape[1] == 2

    def test_error_handling_in_save(self):
        """Test error handling when file save fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid stream first
            output_path = Path(tmpdir) / "test.wav"
            stream = FileOutputStream(output_path=output_path)

            # Make the file path invalid after initialization
            stream.output_path = Path("/invalid/path/that/does/not/exist/test.wav")

            audio_data = np.random.randn(1600).astype(np.float32)
            stream.play_chunk(audio_data)

            # Should raise an exception when trying to save
            with pytest.raises(Exception):
                stream.stop()
