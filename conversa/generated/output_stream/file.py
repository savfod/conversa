"""
File output stream implementation for saving audio to WAV files.
"""

from pathlib import Path

import numpy as np

from conversa.audio.audio_io import save_audio
from conversa.generated.output_stream.base import AbstractAudioOutputStream


class FileOutputStream(AbstractAudioOutputStream):
    """File output stream that saves audio chunks to a WAV file."""

    def __init__(
        self,
        output_path: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """
        Initialize file output stream.

        Args:
            output_path: Path where audio file will be saved
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        super().__init__(sample_rate, channels)
        self.output_path = Path(output_path)
        self._audio_chunks: list[np.ndarray] = []
        self._is_closed = False

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Add audio chunk to buffer (non-blocking).

        Args:
            audio_data: Audio data as numpy array to save

        Returns:
            None
        """
        assert not self._is_closed, "Cannot play_chunk after stream is closed"
        assert audio_data is not None, "Audio data cannot be None"
        assert len(audio_data) > 0, "Audio data cannot be empty"

        # Store the chunk
        self._audio_chunks.append(audio_data.copy())

    def stop(self) -> None:
        """Stop the stream and save all buffered audio to file.

        Returns:
            None
        """
        if self._is_closed:
            return

        self._save_to_file()
        # We don't mark as closed permanently to allow resumption.
        pass

    def close(self) -> None:
        """Permanently close the stream."""
        if self._is_closed:
            return

        self.stop()
        self._is_closed = True

    def wait(self) -> None:
        """Block until file is written (for compatibility with base class).

        For FileOutputStream, this is effectively the same as stop() since
        file writing happens synchronously.

        Returns:
            None
        """
        # Ensure data is saved
        self.stop()

    def _save_to_file(self) -> None:
        """Save all buffered audio chunks to the output file."""
        if not self._audio_chunks:
            # print(f"Warning: No audio chunks to save to {self.output_path}")
            return

        # Concatenate all chunks
        audio_data = np.concatenate(self._audio_chunks)

        # Ensure correct shape for channels
        if self.channels == 1 and audio_data.ndim == 1:
            # Mono audio is fine as 1D array
            pass
        elif self.channels > 1 and audio_data.ndim == 1:
            # Convert mono to multi-channel
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, self.channels))

        # Use shared save_audio utility
        save_audio(audio_data, self.output_path, self.sample_rate)

    def get_total_duration(self) -> float:
        """Get total duration of buffered audio in seconds.

        Returns:
            Total duration in seconds.
        """
        if not self._audio_chunks:
            return 0.0

        total_samples = sum(len(chunk) for chunk in self._audio_chunks)
        return total_samples / self.sample_rate

    def get_chunk_count(self) -> int:
        """Get number of buffered audio chunks.

        Returns:
            Number of chunks.
        """
        return len(self._audio_chunks)

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return False
