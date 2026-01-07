"""
Audio output stream implementations for playing and saving audio.

This package provides a modular architecture for audio output streams:
- AbstractAudioOutputStream: Base class for all output streams
- SpeakerOutputStream: Real-time audio playback using sounddevice
- FileOutputStream: Save audio chunks to WAV files
"""

from conversa.generated.output_stream.base import AbstractAudioOutputStream
from conversa.generated.output_stream.file import FileOutputStream
from conversa.generated.output_stream.speaker import SpeakerOutputStream

__all__ = [
    "AbstractAudioOutputStream",
    "SpeakerOutputStream",
    "FileOutputStream",
]
