import queue
import time

import numpy as np

from conversa.audio.input_stream.base import AbstractAudioInputStream


class VirtualMicrophone(AbstractAudioInputStream):
    """
    A virtual microphone that simulates real-time audio input.
    Chunks added via add_chunk are made available in the stream
    at a rate corresponding to their duration.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        super().__init__(sample_rate, channels)
        self._input_queue = queue.Queue()

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add a chunk of audio to be 'captured' by the microphone."""
        self._input_queue.put(chunk)

    def _audio_processing_loop(self) -> None:
        while self._is_running:
            try:
                # Get chunk with a timeout to allow checking _is_running
                chunk = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Calculate duration
            duration = len(chunk) / self.sample_rate

            # Simulate real-time delay
            time.sleep(duration)

            # Make available in the buffer
            self._buffer.add_audio(chunk)
