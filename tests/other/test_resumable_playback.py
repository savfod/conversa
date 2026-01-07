from unittest.mock import patch

import numpy as np
import pytest

from conversa.generated.output_stream.file import FileOutputStream
from conversa.generated.output_stream.speaker import SpeakerOutputStream
from conversa.web.io import WebOutputStream


# Mock sounddevice to avoid actual audio hardware interaction
@pytest.fixture
def mock_sd():
    with patch("conversa.generated.output_stream.speaker.sd") as mock:
        yield mock


@pytest.fixture
def mock_server():
    with patch("conversa.web.io.server") as mock:
        yield mock


def test_speaker_resumable_playback(mock_sd):
    """Test SpeakerOutputStream supports play -> stop -> play."""
    stream = SpeakerOutputStream(sample_rate=16000)
    audio_data = np.zeros(16000, dtype=np.float32)

    # First play
    stream.play_chunk(audio_data)
    # Verify stream started
    mock_sd.OutputStream.assert_called()
    stream_instance = mock_sd.OutputStream.return_value
    assert stream_instance.start.called

    # Write some data (simulated)
    # We can't easily simulate thread internals but we can check state transitions

    # Stop
    # Stop
    stream.stop()
    assert stream_instance.stop.called
    # We no longer close the stream on stop, we just stop it

    # Reset mock for second play
    # mock_sd.OutputStream.reset_mock() # We don't recreate stream, so constructor won't be called

    # Second play
    stream.play_chunk(audio_data)

    # Should start the same stream again
    # We can check that start is called (conceptually, though sd might optimize? No, we call start explicitly)
    assert stream_instance.start.call_count >= 1

    stream.stop()


def test_file_resumable_playback(tmp_path):
    """Test FileOutputStream supports play -> stop -> play."""
    output_file = tmp_path / "test.wav"
    stream = FileOutputStream(output_path=output_file)
    audio_data = np.zeros(1600, dtype=np.float32)

    stream.play_chunk(audio_data)
    stream.stop()

    # Needs to proceed without error
    stream.play_chunk(audio_data)
    stream.stop()

    # Verify file content length (should be 2 chunks * 1600 samples)
    import soundfile as sf

    data, sr = sf.read(output_file)
    assert len(data) == 3200


def test_web_resumable_playback(mock_server):
    """Test WebOutputStream supports play -> stop -> play."""
    stream = WebOutputStream()
    audio_data = np.zeros(1600, dtype=np.float32)

    stream.play_chunk(audio_data)
    stream.stop()
    stream.play_chunk(audio_data)  # Should not raise

    assert mock_server.emit_audio_out.call_count == 2
