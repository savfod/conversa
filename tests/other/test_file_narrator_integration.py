from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.generated.file_narrator import FileNarrator
from conversa.generated.output_stream.base import AbstractAudioOutputStream


class MockInputStream(AbstractAudioInputStream):
    def _audio_processing_loop(self) -> None:
        pass


class MockOutputStream(AbstractAudioOutputStream):
    def play_chunk(self, audio_data: np.ndarray) -> None:
        pass

    def stop(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def is_playing(self) -> bool:
        return False


def test_file_narrator_with_injected_streams(tmp_path: Path):
    """Test FileNarrator using injected mock streams."""
    # Create a dummy text file
    content = "Hello world."
    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    # Mock streams
    mock_input = MagicMock(spec=AbstractAudioInputStream)
    mock_output = MagicMock(spec=AbstractAudioOutputStream)
    mock_output.is_playing.return_value = False

    # Mock ContentProcessor to avoid LLM/TTS calls
    with patch("conversa.generated.file_narrator.ContentProcessor") as MockProcessor:
        mock_proc_instance = MockProcessor.return_value
        # buffer audio
        audio_data = np.zeros(16000, dtype=np.float32)
        mock_proc_instance.prepare_chunk.return_value = (content, audio_data)

        narrator = FileNarrator(
            file_path=str(temp_file),
            chunk_size=100,
            input_stream=mock_input,
            output_stream=mock_output,
            enable_voice_control=True,
        )

        # Mock voice control setup to use our injected streams
        # Note: refactored logic uses injected streams

        # We also need to mock AudioParser inside _setup_voice_control or check if we can bypass
        # The _setup_voice_control creates AudioParser if enabled.
        # We can mock AudioParser class
        with patch("conversa.generated.file_narrator.AudioParser") as _MockParser:
            # Also mock CommandListener to avoid real loop waiting
            with patch(
                "conversa.generated.file_narrator.CommandListener"
            ) as MockListener:
                mock_listener_instance = MockListener.return_value
                # run_voice_control_loop should return something
                mock_listener_instance.run_voice_control_loop.return_value = None

                narrator.read_file()

                # Verify output stream usage
                assert mock_output.play_chunk.called
                # Verify input stream usage (passed to listener)
                # mock_input should be passed to CommandListener
                call_args = MockListener.call_args
                assert call_args[0][0] == mock_input


def test_file_narrator_defaults(tmp_path: Path):
    """Test FileNarrator creates default streams if not provided."""
    content = "Hello world."
    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    with patch("conversa.generated.file_narrator.SpeakerOutputStream") as MockSpeaker:
        with patch("conversa.generated.file_narrator.ContentProcessor"):
            narrator = FileNarrator(
                file_path=str(temp_file), enable_voice_control=False
            )

            # read_file should create SpeakerOutputStream
            # We mock CommandListener to avoid blocking
            with patch(
                "conversa.generated.file_narrator.CommandListener"
            ) as MockListener:
                _mock_listener_instance = MockListener.return_value
                # Mock run_voice_control_loop
                chunk_mock = MagicMock()
                chunk_mock.audio = np.zeros(10)

                # We need to mock ChunkAsyncPreprocessor iteration
                with patch(
                    "conversa.generated.file_narrator.ChunkAsyncPreprocessor"
                ) as MockPreprocessor:
                    mock_prep = MockPreprocessor.return_value
                    mock_prep.__iter__.return_value = [chunk_mock]

                    narrator.read_file()

                    assert MockSpeaker.called
