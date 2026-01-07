"""
Thread-safe audio buffer for managing audio stream data.
"""

import threading

import numpy as np


class AudioBuffer:
    """Thread-safe audio buffer with automatic size management."""

    def __init__(self, max_duration_seconds: float = 60.0, sample_rate: int = 16000):
        """
        Initialize audio buffer.

        Args:
            max_duration_seconds: Maximum buffer duration before trimming
            sample_rate: Audio sample rate
        """
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self._buffer = []
        self._lock = threading.Lock()

    def add_audio(self, audio_data: np.ndarray) -> None:
        """Add audio data to the buffer with automatic trimming.

        Args:
            audio_data: A numpy array of samples (mono or multi-channel).

        Returns:
            None
        """
        with self._lock:
            self._buffer.extend(audio_data.flatten())

            # Trim buffer if it exceeds maximum size
            if len(self._buffer) > self.max_samples:
                # Trim to max_samples to enforce the limit
                excess = len(self._buffer) - self.max_samples
                self._buffer = self._buffer[excess:]
                print(
                    f"Warning: Audio buffer trimmed by {excess} samples "
                    f"({excess / self.sample_rate:.2f} seconds)"
                )

    def get_and_clear(self) -> np.ndarray:
        """Return all buffered audio as a numpy array and clear the buffer.

        Returns:
            A numpy array of dtype float32 containing the buffered audio
            samples. An empty array is returned if no audio was buffered.
        """
        with self._lock:
            if not self._buffer:
                return np.array([])

            audio_data = np.array(self._buffer, dtype=np.float32)
            self._buffer.clear()
            return audio_data

    def get_duration(self) -> float:
        """Get current buffer duration in seconds.

        Returns:
            Buffer duration in seconds.
        """
        with self._lock:
            return len(self._buffer) / self.sample_rate if self._buffer else 0.0
