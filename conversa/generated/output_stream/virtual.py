import queue
import threading
import time
from typing import Optional

import numpy as np

from conversa.generated.output_stream.base import AbstractAudioOutputStream


class VirtualSpeaker(AbstractAudioOutputStream):
    """A virtual speaker that simulates audio playback.
    It tracks the expected completion time of the audio sent to it.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_limit: Optional[int] = None,
    ):
        super().__init__(sample_rate, channels)
        self._buffer = queue.Queue()
        self._chunk_limit = chunk_limit
        self._playback_end_time = 0.0
        self._lock = threading.Lock()
        self._playing = False

    def play_chunk(self, audio_data: np.ndarray) -> None:
        if not self._playing:
            self._playing = True

        with self._lock:
            # Add to storage for verification/debugging
            self._buffer.put(audio_data)

            duration = len(audio_data) / self.sample_rate
            current_time = time.time()

            if self._playback_end_time < current_time:
                self._playback_end_time = current_time + duration
            else:
                self._playback_end_time += duration

    def stop(self) -> None:
        self._playing = False
        with self._lock:
            self._playback_end_time = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                wait_time = self._playback_end_time - time.time()

            if wait_time <= 0:
                break

            # Sleep in small checks to be responsive or just sleep the whole time
            # Using loop to be robust against slight timing drifts or new chunks
            time.sleep(min(wait_time, 0.1))

    def is_playing(self) -> bool:
        with self._lock:
            return time.time() < self._playback_end_time

    def get_unprocessed_chunk(self) -> Optional[np.ndarray]:
        """
        Get a chunk of 'played' audio from the internal buffer.
        """
        try:
            chunk = self._buffer.get_nowait()
            if self._chunk_limit and len(chunk) > self._chunk_limit:
                # Logic for splitting could be complex if we want to preserve the rest.
                pass
            return chunk
        except queue.Empty:
            return None
