"""
Abstract base class for audio output streams.
"""

import abc

import numpy as np


class AbstractAudioOutputStream(abc.ABC):
    """Abstract base class for audio output streams."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio output stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels

    @abc.abstractmethod
    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Play audio chunk without blocking.

        Args:
            audio_data: Audio data as numpy array to play

        Returns:
            None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the audio output stream.

        Returns:
            None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait(self) -> None:
        """Block until all audio playback is finished.

        Returns:
            None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_playing(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if audio is playing, False otherwise
        """
        raise NotImplementedError()
