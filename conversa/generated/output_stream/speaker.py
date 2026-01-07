import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from conversa.generated.output_stream.base import AbstractAudioOutputStream


class SpeakerOutputStream(AbstractAudioOutputStream):
    """Speaker output stream implementation using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
    ):
        """
        Initialize speaker output stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            device: Specific audio device ID (None for default)
        """
        super().__init__(sample_rate, channels)
        self.device = device
        self._queue: queue.Queue = queue.Queue()
        self._current_chunk: Optional[np.ndarray] = None
        self._current_chunk_idx: int = 0
        self._chunk_lock = threading.Lock()

        # Stream is now created on demand
        self._stream: Optional[sd.OutputStream] = None

    def _callback(self, outdata, frames, time, status):
        """Audio callback function."""
        if status:
            print(f"Status: {status}")

        chunk_size = len(outdata)
        out_idx = 0

        # Fill outdata from queue
        while out_idx < chunk_size:
            with self._chunk_lock:
                # Get new chunk if needed
                if self._current_chunk is None:
                    try:
                        data = self._queue.get_nowait()
                        self._current_chunk = data
                        self._current_chunk_idx = 0
                    except queue.Empty:
                        # No more data, fill rest with zeros
                        outdata[out_idx:] = 0
                        return

                # Copy data
                remaining = chunk_size - out_idx
                chunk_remaining = len(self._current_chunk) - self._current_chunk_idx

                to_copy = min(remaining, chunk_remaining)

                outdata[out_idx : out_idx + to_copy] = self._current_chunk[
                    self._current_chunk_idx : self._current_chunk_idx + to_copy
                ]

                out_idx += to_copy
                self._current_chunk_idx += to_copy

                # Check if chunk finished
                if self._current_chunk_idx >= len(self._current_chunk):
                    self._current_chunk = None
                    self._queue.task_done()

    def _finished_callback(self):
        """Called when stream finishes."""
        pass  # We don't auto-stop the stream usually, unless we want to?

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Play audio chunk without blocking.

        Args:
            audio_data: Audio data as numpy array to play
        """
        assert audio_data is not None, "Audio data cannot be None"
        assert len(audio_data) > 0, "Audio data cannot be empty"

        # Ensure correct shape for channels
        if self.channels == 1 and audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        elif self.channels > 1 and audio_data.ndim == 1:
            # Duplicate mono to all channels
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, self.channels))

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        self._queue.put(audio_data)

        # Create/Start stream if not active
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms block size
                device=self.device,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
                finished_callback=self._finished_callback,
            )
            self._stream.start()

    def stop(self) -> None:
        """Stop the audio output stream immediately."""
        if self._stream is None:
            return

        # Stop and close stream immediately
        # We catch potential errors if stream is arguably already closed
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            print(f"Error stopping stream: {e}")

        self._stream = None

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Clear current chunk
        with self._chunk_lock:
            if self._current_chunk is not None:
                self._current_chunk = None
                self._current_chunk_idx = 0
                # Important: Mark the interrupted chunk as done so queue.join() doesn't block
                # HOWEVER: if we pulled it from the queue, we must task_done it.
                # In _callback, we task_done ONLY when finished.
                # So if we interrupt it, we MUST task_done it here.
                try:
                    self._queue.task_done()
                except ValueError:
                    # Could happen if called too many times? Should not if we logic is correct.
                    pass

        print("Speaker stream stopped")

    def wait(self) -> None:
        """Block until all audio playback is finished."""
        if self._stream is None and self._queue.empty():
            return

        self._queue.join()

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if self._stream is None:
            return False
        return not self._queue.empty() or self._current_chunk is not None
