"""
Abstract base class for audio input streams.
"""

import abc
import threading
from typing import Optional

import numpy as np

from conversa.audio.input_stream.buffer import AudioBuffer


class AbstractAudioInputStream(abc.ABC):
    """Abstract base class for audio input streams."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio input stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._buffer = AudioBuffer(sample_rate=sample_rate)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    @abc.abstractmethod
    def _audio_processing_loop(self) -> None:
        """Internal audio processing loop to be implemented by subclasses.

        Subclasses should implement this method to push audio into
        `self._buffer` while `self._is_running` is True.
        """
        raise NotImplementedError()

    def start(self) -> None:
        """Start the audio input stream in a background thread.

        Returns:
            None
        """
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self._thread.start()
        print(f"Started {self.__class__.__name__}")

    def stop(self) -> None:
        """Stop the audio input stream and join the background thread.

        Returns:
            None
        """
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        print(f"Stopped {self.__class__.__name__}")

    def get_unprocessed_chunk(self) -> Optional[np.ndarray]:
        """Get currently buffered audio data.

        Returns:
            Audio chunk as numpy array, or None if no data available.
        """
        chunk = self._buffer.get_and_clear()
        return chunk if len(chunk) > 0 else None

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds.

        Returns:
            Buffer duration in seconds (float).
        """
        return self._buffer.get_duration()
