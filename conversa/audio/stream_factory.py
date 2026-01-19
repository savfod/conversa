"""Factory for creating audio stream instances."""

from typing import Type

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.audio.input_stream.file import AudioFileInputStream
from conversa.audio.input_stream.microphone import MicrophoneInputStream
from conversa.audio.output_stream.base import AbstractAudioOutputStream
from conversa.audio.output_stream.file import FileOutputStream
from conversa.audio.output_stream.speaker import SpeakerOutputStream

INPUT_STREAMS: dict[str, Type[AbstractAudioInputStream]] = {
    "file": AudioFileInputStream,
    "microphone": MicrophoneInputStream,
}

OUTPUT_STREAMS: dict[str, Type[AbstractAudioOutputStream]] = {
    "file": FileOutputStream,
    "speaker": SpeakerOutputStream,
}


def register_input_stream(
    name: str, stream_class: Type[AbstractAudioInputStream]
) -> None:
    """Register an input stream class for later creation via create_input_stream."""
    if name in INPUT_STREAMS:
        raise ValueError(f"Input stream '{name}' is already registered")
    INPUT_STREAMS[name] = stream_class


def register_output_stream(
    name: str, stream_class: Type[AbstractAudioOutputStream]
) -> None:
    """Register an output stream class for later creation via create_output_stream."""
    if name in OUTPUT_STREAMS:
        raise ValueError(f"Output stream '{name}' is already registered")
    OUTPUT_STREAMS[name] = stream_class


def create_input_stream(stream_type: str, **kwargs) -> AbstractAudioInputStream:
    """Create an input stream instance.

    Args:
        stream_type: Type of input stream (e.g., "file", "microphone", "web").
        **kwargs: Arguments passed to the stream constructor.

    Returns:
        An input stream instance.

    Raises:
        ValueError: If stream_type is unknown.
    """
    if stream_type not in INPUT_STREAMS:
        raise ValueError(
            f"Unknown input stream type: {stream_type}. Available: {list(INPUT_STREAMS.keys())}"
        )
    return INPUT_STREAMS[stream_type](**kwargs)


def create_output_stream(stream_type: str, **kwargs) -> AbstractAudioOutputStream:
    """Create an output stream instance.

    Args:
        stream_type: Type of output stream (e.g., "file", "speaker", "web").
        **kwargs: Arguments passed to the stream constructor.

    Returns:
        An output stream instance.

    Raises:
        ValueError: If stream_type is unknown.
    """
    if stream_type not in OUTPUT_STREAMS:
        raise ValueError(
            f"Unknown output stream type: {stream_type}. Available: {list(OUTPUT_STREAMS.keys())}"
        )
    return OUTPUT_STREAMS[stream_type](**kwargs)
