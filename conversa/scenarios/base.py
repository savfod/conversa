"""Abstract base class for scenarios."""

import argparse
from abc import ABC, abstractmethod

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.audio.output_stream.base import AbstractAudioOutputStream
from conversa.util.config import Config


class AbstractScenario(ABC):
    """Base class for all scenarios.

    Provides a consistent interface with start() and stop() methods,
    and a stop mechanism via _stop_requested flag.
    """

    def __init__(
        self,
        input_stream: AbstractAudioInputStream,
        output_stream: AbstractAudioOutputStream,
        config: Config,
        args: argparse.Namespace,
    ):
        """Initialize the scenario.

        Args:
            input_stream: Audio input stream for capturing audio.
            output_stream: Audio output stream for playback.
            config: Application configuration.
            args: Parsed command-line arguments.
        """
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.config = config
        self.args = args
        self._stop_requested: bool = False

    @abstractmethod
    def start(self) -> None:
        """Start the scenario (blocking, runs main loop)."""
        pass

    def stop(self) -> None:
        """Signal the scenario to stop."""
        self._stop_requested = True

    @property
    def should_stop(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Add scenario-specific CLI arguments. Override in subclasses."""
        pass
