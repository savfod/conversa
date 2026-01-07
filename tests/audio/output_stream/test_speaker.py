"""
Tests for SpeakerOutputStream class.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from conversa.generated.output_stream.speaker import SpeakerOutputStream


@patch("conversa.generated.output_stream.speaker.sd.OutputStream")
class TestSpeakerOutputStream:
    """Test suite for SpeakerOutputStream class."""

    def test_initialization(self, mock_output_stream):
        """Test SpeakerOutputStream initialization with default parameters."""
        stream = SpeakerOutputStream()

        assert stream.sample_rate == 16000
        assert stream.channels == 1
        assert stream.device is None
        assert stream._stream is None
        assert not stream.is_playing()

        # Verify stream was NOT created in init
        mock_output_stream.assert_not_called()

    def test_initialization_custom_parameters(self, mock_output_stream):
        """Test initialization with custom parameters."""
        stream = SpeakerOutputStream(sample_rate=8000, channels=2, device=1)

        assert stream.sample_rate == 8000
        assert stream.channels == 2
        assert stream.device == 1

        # Verify stream NOT created yet
        mock_output_stream.assert_not_called()

    def test_play_chunk_starts_stream(self, mock_output_stream):
        """Test that play_chunk starts the stream."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream()
        audio_data = np.random.randn(1600).astype(np.float32)

        stream.play_chunk(audio_data)

        # Verify stream was created and started
        mock_output_stream.assert_called_once()
        assert mock_stream_instance.start.called
        assert stream.is_playing()

    def test_callback_processing(self, mock_output_stream):
        """Test that the callback correctly processes audio from the queue."""
        stream = SpeakerOutputStream()

        # Trigger stream creation to get access to callback
        stream.play_chunk(np.zeros(10))

        # Capture the callback
        _, kwargs = mock_output_stream.call_args
        callback = kwargs["callback"]

        # Clear queue from initial play_chunk call
        while not stream._queue.empty():
            stream._queue.get()
            stream._queue.task_done()

        # Now add real audio to play
        audio_data = np.ones((100, 1), dtype=np.float32)  # 100 samples of 1.0
        stream.play_chunk(audio_data)

        # Prepare outdata buffer
        outdata = np.zeros((100, 1), dtype=np.float32)
        frames = 100
        status = None

        # Call callback
        callback(outdata, frames, None, status)

        # Verify outdata is filled with ones
        assert np.allclose(outdata, 1.0)

        # Verify queue is empty (task_done called)
        assert stream._queue.empty()

    def test_callback_partial_processing(self, mock_output_stream):
        """Test that callback handles data larger than block size (multiple calls)."""
        stream = SpeakerOutputStream()

        # Trigger stream creation
        stream.play_chunk(np.zeros(10))
        _, kwargs = mock_output_stream.call_args
        callback = kwargs["callback"]

        # Clear queue
        while not stream._queue.empty():
            stream._queue.get()
            stream._queue.task_done()

        # Add 200 samples
        audio_data = np.ones((200, 1), dtype=np.float32)
        stream.play_chunk(audio_data)

        # Call 1: Request 100 samples
        outdata1 = np.zeros((100, 1), dtype=np.float32)
        callback(outdata1, 100, None, None)
        assert np.allclose(outdata1, 1.0)

        # Call 2: Request 100 samples
        outdata2 = np.zeros((100, 1), dtype=np.float32)
        callback(outdata2, 100, None, None)
        assert np.allclose(outdata2, 1.0)

        # Queue should be empty now
        assert stream._queue.empty()

    def test_callback_silence_when_empty(self, mock_output_stream):
        """Test that callback fills with zeros when queue is empty."""
        stream = SpeakerOutputStream()

        stream.play_chunk(np.zeros(10))
        _, kwargs = mock_output_stream.call_args
        callback = kwargs["callback"]

        # Clear queue
        while not stream._queue.empty():
            stream._queue.get()
            stream._queue.task_done()

        # No play_chunk called (queue empty)

        outdata = np.ones((100, 1), dtype=np.float32)  # Init with junk
        callback(outdata, 100, None, None)

        assert np.allclose(outdata, 0.0)

    def test_stop(self, mock_output_stream):
        """Test stopping the stream."""
        mock_stream_instance = MagicMock()
        mock_output_stream.return_value = mock_stream_instance

        stream = SpeakerOutputStream()
        stream.play_chunk(np.zeros(100, dtype=np.float32))

        assert stream.is_playing()

        stream.stop()

        # Verify stream was stopped and closed
        assert mock_stream_instance.stop.called
        assert mock_stream_instance.close.called
        assert not stream.is_playing()
        assert stream._stream is None
        assert stream._queue.empty()

    def test_wait(self, mock_output_stream):
        """Test waiting for playback to finish."""
        stream = SpeakerOutputStream()

        # Mock queue.join to verify it's called
        with patch.object(stream._queue, "join") as mock_join:
            stream.play_chunk(np.zeros(100, dtype=np.float32))
            stream.wait()
            mock_join.assert_called_once()

    def test_play_chunk_mono_to_stereo(self, mock_output_stream):
        """Test auto-duplication of mono input to stereo output."""
        stream = SpeakerOutputStream(channels=2)

        audio_mono = np.ones(100, dtype=np.float32)
        stream.play_chunk(audio_mono)

        # Peek at queue
        queued_item = stream._queue.get()
        assert queued_item.shape == (100, 2)
        assert np.allclose(queued_item, 1.0)

    def test_play_chunk_float_conversion(self, mock_output_stream):
        """Test that int audio data is converted to float32."""
        stream = SpeakerOutputStream()

        audio_int = np.ones(100, dtype=np.int16)
        stream.play_chunk(audio_int)

        queued_item = stream._queue.get()
        assert queued_item.dtype == np.float32

    def test_is_playing(self, mock_output_stream):
        """Test is_playing state."""
        stream = SpeakerOutputStream()
        assert not stream.is_playing()

        stream.play_chunk(np.zeros(100, dtype=np.float32))
        assert stream.is_playing()

        # Simulate draining queue
        stream._queue.get()
        stream._queue.task_done()

        # Need to ensure current chunk is cleared too for is_playing to be false
        # But we can't easily access internals here if we just peek
        # Actually is_playing checks: stream is not None AND (queue not empty OR current_chunk not None)

        # If we didn't run callback, current_chunk stays None unless we manually set it
        # Wait, if we pull from queue via get(), stream._current_chunk is not set.
        # So is_playing will be True because stream is not None, but queue empty?
        # NO, code: `if self._stream is None: return False`
        # `return not self._queue.empty() or self._current_chunk is not None`

        # So if we manually empty queue, is_playing should be False (since current_chunk is None)
        assert not stream.is_playing()  # Stream is active but no content

        # queue is passed to get(), so it is empty.

        # Ah, play_chunk sets stream.
        # queue empty. current_chunk None.
        # So is_playing returns False? NO.
        # It returns `not self._queue.empty()` which is False (queue is empty).
        # OR `self._current_chunk is not None` which is False.
        # So is_playing() returns False.

        assert not stream.is_playing()  # stream exists, but no data.

        # BUT this might be flaky if we consider that stream exists means "active".
        # But existing implementation: playing means has data.
