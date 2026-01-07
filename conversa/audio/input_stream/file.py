"""
Audio file input stream that simulates real-time processing.
"""

import time
from pathlib import Path

from conversa.audio.audio_io import read_audio
from conversa.audio.input_stream.base import AbstractAudioInputStream


class AudioFileInputStream(AbstractAudioInputStream):
    """Audio file input stream that simulates real-time processing."""

    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.1,
    ):
        """
        Initialize audio file input stream.

        Args:
            file_path: Path to audio file (MP3, WAV, FLAC, etc.)
            sample_rate: Target sample rate
            channels: Target number of channels
            chunk_duration: Duration of each chunk in seconds
        """
        super().__init__(sample_rate, channels)
        self.file_path = Path(file_path)
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # Note: File existence will be checked in _load_and_convert_mp3()
        # This allows for fallback behavior if file doesn't exist

    def _audio_processing_loop(self) -> None:
        """Audio processing loop for MP3/file input.

        Reads the file, splits into chunks of `self.chunk_duration` and
        appends them to the internal buffer to simulate real-time input.
        """
        try:
            # Load the entire audio file
            audio_data = read_audio(self.file_path, sample_rate=self.sample_rate)
            total_samples = len(audio_data)
            current_position = 0

            print(
                f"Starting MP3 playback simulation: {total_samples / self.sample_rate:.2f} seconds"
            )

            while self._is_running and current_position < total_samples:
                # Calculate chunk end position
                chunk_end = min(current_position + self.chunk_size, total_samples)

                # Extract chunk
                chunk = audio_data[current_position:chunk_end]

                # Add chunk to buffer
                if len(chunk) > 0:
                    self._buffer.add_audio(chunk)

                current_position = chunk_end

                # Sleep to simulate real-time playback
                time.sleep(self.chunk_duration)

            print("MP3 file processing completed")

        except Exception as e:
            print(f"Error in MP3 processing: {e}")
            self._is_running = False
