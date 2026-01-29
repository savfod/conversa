"""Debugging scenario that mirrors input audio to output."""

import argparse
import time

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.audio.output_stream.base import AbstractAudioOutputStream
from conversa.scenarios.base import AbstractScenario
from conversa.util.config import Config


class DebugMirrorScenario(AbstractScenario):
    """Debugging scenario that mirrors input audio to output.

    Useful for testing audio I/O without LLM processing.
    """

    def __init__(
        self,
        input_stream: AbstractAudioInputStream,
        output_stream: AbstractAudioOutputStream,
        config: Config,
        args: argparse.Namespace,
    ):
        """Initialize the debug mirror scenario.

        Args:
            input_stream: Audio input stream.
            output_stream: Audio output stream.
            config: Application configuration (unused, for consistent signature).
            args: Parsed command-line arguments.
        """
        super().__init__(input_stream, output_stream, config, args)

    def start(self) -> None:
        """Start mirroring audio from input to output."""
        print("Mirroring audio. Press Ctrl+C to stop.")

        self.input_stream.start()

        try:
            while not self.should_stop:
                chunk = self.input_stream.get_unprocessed_chunk()
                if chunk is not None and len(chunk) > 0:
                    self.output_stream.play_chunk(chunk)
                else:
                    time.sleep(0.01)
        finally:
            self.input_stream.stop()
            self.output_stream.stop()
            self.output_stream.wait()
            print("Done")

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Add arguments for the debug mirror scenario."""
        pass
