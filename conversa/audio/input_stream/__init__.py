"""
Audio input stream implementations for handling microphone and file inputs.

This package provides a modular architecture for audio input streams:
- AudioBuffer: Thread-safe buffer for audio data
- AbstractAudioInputStream: Base class for all input streams
- MicrophoneInputStream: Real-time microphone input
- AudioFileInputStream: File-based input with simulated real-time playback
"""

from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.audio.input_stream.buffer import AudioBuffer
from conversa.audio.input_stream.file import AudioFileInputStream
from conversa.audio.input_stream.microphone import MicrophoneInputStream

__all__ = [
    "AudioBuffer",
    "AbstractAudioInputStream",
    "MicrophoneInputStream",
    "AudioFileInputStream",
]
