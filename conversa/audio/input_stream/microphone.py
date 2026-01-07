"""
Microphone input stream implementation using sounddevice.
"""

import time
from typing import Optional

import numpy as np
import sounddevice as sd

from conversa.audio.input_stream.base import AbstractAudioInputStream


class MicrophoneInputStream(AbstractAudioInputStream):
    """Microphone input stream implementation using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 1024,
        device: Optional[int] = None,
    ):
        """
        Initialize microphone input stream.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            block_size: Audio block size for processing
            device: Specific audio device ID (None for default)
        """
        super().__init__(sample_rate, channels)
        self.block_size = block_size
        self.device = device
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Callback function for sounddevice InputStream.

        Args:
            indata: Incoming audio samples as numpy array (frames, channels).
            frames: Number of frames in this callback.
            time_info: Timing information provided by sounddevice.
            status: Callback status object from sounddevice.

        Returns:
            None
        """
        if status:
            print(f"Audio callback status: {status}")

        # Add audio data to buffer
        self._buffer.add_audio(indata)

    def _audio_processing_loop(self) -> None:
        """Audio processing loop for microphone input.

        This opens a sounddevice InputStream and runs until `self._is_running`
        is cleared.
        """
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                device=self.device,
                callback=self._audio_callback,
                dtype=np.float32,
            ) as self._stream:
                print(
                    f"Microphone stream started (device: {self.device}, "
                    f"rate: {self.sample_rate}, channels: {self.channels})"
                )

                while self._is_running:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting

        except Exception as e:
            print(f"Error in microphone processing: {e}")
            self._is_running = False
