import time
from unittest.mock import patch

import numpy as np
import pytest

from conversa.web.io import WebOutputStream


class TestWebOutputStream:
    @patch("conversa.web.io.server")
    def test_play_chunk_sends_wav(self, mock_server):
        stream = WebOutputStream()
        # Create dummy audio data
        audio_data = np.zeros(16000, dtype=np.float32)

        stream.play_chunk(audio_data)

        # Verify emit_audio_out was called
        assert mock_server.emit_audio_out.called
        # Check args
        args, kwargs = mock_server.emit_audio_out.call_args
        assert isinstance(args[0], bytes)  # WAV bytes
        assert kwargs.get("sid") is None

    @patch("conversa.web.io.server")
    def test_stop_sends_signal(self, mock_server):
        stream = WebOutputStream(sid="test_sid")
        stream.stop()

        assert mock_server.emit_audio_stop.called
        args, kwargs = mock_server.emit_audio_stop.call_args
        assert kwargs.get("sid") == "test_sid"

    @pytest.mark.slow
    @patch("conversa.web.io.server")
    def test_wait_blocks_appropriately(self, mock_server):
        stream = WebOutputStream(sample_rate=16000)

        # 0.5s of audio
        audio_data = np.zeros(8000, dtype=np.float32)

        start_time = time.time()
        stream.play_chunk(audio_data)

        # Immediate wait should block for ~0.5s
        stream.wait()
        end_time = time.time()

        duration = end_time - start_time
        assert duration >= 0.45  # Allow some slush

        start_time = time.time()
        stream.play_chunk(audio_data)
        stream.play_chunk(audio_data)

        # Two waits should block for ~1s
        stream.wait()
        end_time = time.time()

        duration = end_time - start_time
        assert duration >= 0.9  # Allow some slush

    @patch("conversa.web.io.server")
    def test_stop_resets_wait(self, mock_server):
        stream = WebOutputStream(sample_rate=16000)

        # 1.0s of audio
        audio_data = np.zeros(16000, dtype=np.float32)

        stream.play_chunk(audio_data)

        # Stop immediately
        stream.stop()

        start_time = time.time()
        stream.wait()  # Should return instantly
        end_time = time.time()

        duration = end_time - start_time
        assert duration < 0.1
