"""
Web-based input and output stream implementations.
"""

import time
from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.generated.output_stream.base import AbstractAudioOutputStream
from conversa.web import server


class WebInputStream(AbstractAudioInputStream):
    """
    Input stream that receives audio from a web client via SocketIO.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        super().__init__(sample_rate, channels)
        # We might want to filter by specific client_sid in the future
        self.target_sid: Optional[str] = None

    def _audio_processing_loop(self) -> None:
        """
        Polls the server's input queue for audio chunks.
        """
        while self._is_running:
            try:
                # Wait for data with a timeout to allow checking _is_running
                # server.input_queue contains (sid, raw_bytes)
                try:
                    sid, data = server.input_queue.get(timeout=0.1)
                except server.queue.Empty:
                    continue

                # If we were tracking a specific user, we could filter here.
                # For now, we accept all input (Single Client Assumption).

                # Convert raw bytes (assume int16 from browser) to numpy array
                # The browser JS sends Int16Array buffer.
                audio_int16 = np.frombuffer(data, dtype=np.int16)

                # Convert to float32 to match MicrophoneInputStream behavior
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

                # Check channels. If we expect stereo but get mono, duplications might be needed?
                # Browser typically sends mono if config says 1 channel.
                # If we need to reshape for the buffer:
                if self.channels > 1:
                    # This is a naive expansion, assuming input is mono
                    audio_float32 = np.tile(
                        audio_float32.reshape(-1, 1), (1, self.channels)
                    )

                self._buffer.add_audio(audio_float32)

            except Exception as e:
                print(f"Error in WebInputStream loop: {e}")
                # Don't break loop, just continue
                time.sleep(0.1)


class WebOutputStream(AbstractAudioOutputStream):
    """
    Output stream that sends audio to a web client via SocketIO.
    """

    def __init__(
        self, sample_rate: int = 16000, channels: int = 1, sid: Optional[str] = None
    ):
        super().__init__(sample_rate, channels)
        self.sid = sid
        self.last_chunk_finish_time = 0.0

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """
        Convert audio chunk to WAV and send to client.
        """
        if audio_data is None or len(audio_data) == 0:
            return

        try:
            # Update expected finish time
            duration = len(audio_data) / self.sample_rate
            now = time.time()
            if self.last_chunk_finish_time < now:
                self.last_chunk_finish_time = now

            self.last_chunk_finish_time += duration

            # Convert float32 numpy array to WAV bytes
            buffer = BytesIO()
            # soundfile handles float32 (-1.0 to 1.0) automatically

            sf.write(buffer, audio_data, samplerate=self.sample_rate, format="WAV")
            wav_bytes = buffer.getvalue()

            # Send to server
            server.emit_audio_out(wav_bytes, sid=self.sid)

        except Exception as e:
            print(f"Error in WebOutputStream play_chunk: {e}")

    def stop(self) -> None:
        """Stop playback on the client."""
        # Reset timing
        self.last_chunk_finish_time = 0.0
        # Send stop signal
        server.emit_audio_stop(sid=self.sid)

    def wait(self) -> None:
        """Wait for the audio transmission (approximate)."""
        now = time.time()
        wait_time = self.last_chunk_finish_time - now
        if wait_time > 0:
            time.sleep(wait_time)

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return time.time() < self.last_chunk_finish_time
