"""Tests for the simple speech utilities.

These tests check the basic error behavior when the OPENAI_API_KEY is not set.
"""

from __future__ import annotations

import time

import pytest

from conversa.audio.input_stream import AudioFileInputStream
from conversa.audio.speech_api import speech_to_text, text_to_speech

# def test_functions_raise_when_no_api_key(monkeypatch):
#     """If OPENAI_API_KEY is not present, functions should raise RuntimeError."""
#     # Ensure the environment variable is not present
#     monkeypatch.delenv("OPENAI_API_KEY", raising=False)

#     # Import fresh to ensure dotenv/load happens with the variable missing
#     importlib.reload(speech)

#     with pytest.raises(RuntimeError):
#         speech.speech_to_text(b"\x00\x01")

#     with pytest.raises(RuntimeError):
#         speech.text_to_speech("hello")


@pytest.mark.slow
def test_speech_real():
    stream = AudioFileInputStream("data/test_audio/error1.wav", sample_rate=16000)
    stream.start()
    time.sleep(5)
    data = stream.get_unprocessed_chunk()
    text = speech_to_text(data)
    print(text)


# def test_no_input():
#     # todo: fix and move to a separate file
#     with pytest.raises(FileNotFoundError):
#         stream = AudioFileInputStream("no_such_file.wav", sample_rate=16000)
#         stream.start()
#         _chunk = stream.get_unprocessed_chunk()


if __name__ == "__main__":
    # manual test :(

    import sounddevice as sd

    output_stream = sd.RawOutputStream(
        # samplerate=16000,
        # blocksize=2048,
        # channels=1,
        # dtype='float32',
    )
    # Optionally, convert text back to speech
    tts_audio = text_to_speech("Hello world, this is a test!!")
    output_stream.start()
    output_stream.write(tts_audio)

    print(f"Generated TTS audio of length: {len(tts_audio)} bytes")
