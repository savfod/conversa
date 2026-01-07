"""Simple speech utilities using OpenAI for STT and TTS.

This module provides two convenience functions:
- speech_to_text: transcribe audio bytes to text using an OpenAI model
- text_to_speech: synthesize speech bytes from text using an OpenAI model

The OpenAI API key is loaded via dotenv from the environment variable
`OPENAI_API_KEY`.

Type hints and Google-style docstrings are used.
"""

from __future__ import annotations

import os
import wave
from io import BytesIO
from typing import Any, Optional, Union

import librosa
import numpy as np
import openai
import soundfile as sf
from dotenv import load_dotenv

from conversa.util.logs import log_function_duration

# Load environment variables from a .env file if present
load_dotenv()

DEBUG = os.getenv("DEBUG_SPEECH_API", "0") == "1"
if DEBUG:
    print("DEBUG_SPEECH_API is enabled - verbose logging active.")


def _get_api_key() -> Optional[str]:
    """Return the OpenAI API key from the environment, if available.

    Returns:
        The API key string or None if not set.
    """
    return os.getenv("OPENAI_API_KEY")


def _ensure_openai_available() -> None:
    """Ensure the openai package and API key are available.

    Raises:
        RuntimeError: If the openai package is not installed or API key is missing.
    """
    # Verify import succeeded and that an API key exists in the env.
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")


def _get_client() -> Any:
    """Create and return an OpenAI v1 client instance.

    This function expects the installed `openai` package to expose an
    `OpenAI` client class (openai.OpenAI) as introduced in openai>=1.0.0.

    Returns:
        An OpenAI client instance (shape depends on installed SDK).

    Raises:
        RuntimeError: If the OpenAI client class is not present or cannot be
            constructed.
    """
    api_key = _get_api_key()
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as exc:
        # Provide a clear error message guiding the user to install a
        # compatible openai package or pin to the older version.
        raise RuntimeError(
            "Unable to construct OpenAI client. Ensure openai>=1.0.0 is installed or pin to an older SDK."
        ) from exc


@log_function_duration()
def speech_to_text(
    audio_bytes: Union[bytes, bytearray, np.ndarray],
    language: Optional[str] = "en",
    sample_rate: int = 16000,
    *,
    model: str = "gpt-4o-mini-transcribe",
) -> str:
    """Transcribe audio to text using an OpenAI model.

    Args:
        audio_bytes: Either raw audio container bytes (WAV/MP3) or a numeric
            numpy array of samples (float32 or integer). If a numpy array is
            provided it will be converted to 16-bit PCM WAV for the API.
        language: Optional BCP-47 language code hint (e.g., "en", "es").
        sample_rate: Sample rate to assume for numpy-array inputs.
        model: Model name to use for transcription.

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If the OpenAI package or API key is missing, or if
            the API returns an error.
    """
    _ensure_openai_available()

    try:
        # Normalize different input types into a WAV bytes buffer that the
        # OpenAI API understands. Accept either raw audio bytes (already a
        # file like WAV/MP3) or a numeric array (e.g., numpy.ndarray of
        # float32 samples).
        client = _get_client()

        if DEBUG:
            import soundfile as sf

            # save to temp file and play for debugging
            sf.write(
                "debug_speech_api_input_raw.wav", audio_bytes, sample_rate, format="WAV"
            )

        wav_buffer: BytesIO
        if isinstance(audio_bytes, (bytes, bytearray)):
            # Assume already a file-like audio container (mp3/wav) and pass
            # through directly. Set a filename on the buffer so the client
            # sends a sensible Content-Disposition with an extension the
            # server can use to detect format.
            wav_buffer = BytesIO(audio_bytes)
            head = bytes(audio_bytes[:12])
            # simple magic checks
            if head.startswith(b"RIFF"):
                wav_buffer.name = "audio.wav"
            elif head.startswith(b"ID3") or head[:2] == b"\xff\xfb":
                wav_buffer.name = "audio.mp3"
            else:
                # default to wav
                wav_buffer.name = "audio.wav"
        else:
            # Try to convert a numeric array (numpy) to 16-bit PCM WAV.
            try:
                import numpy as _np

                arr = _np.asarray(audio_bytes)
            except Exception:
                raise RuntimeError(
                    "Unsupported audio input type - provide raw bytes or a numpy array of samples"
                )

            # Convert to mono if multi-channel by taking first channel
            if arr.ndim > 1:
                arr = arr[:, 0]

            # If floats, assume in range [-1, 1]. If integers, scale to int16.
            if _np.issubdtype(arr.dtype, _np.floating):
                clipped = _np.clip(arr, -1.0, 1.0)
                int_samples = (clipped * 32767.0).astype(_np.int16)
            else:
                # Cast to int16, but beware of range
                int_samples = arr.astype(_np.int16)

            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                # wave expects bytes in little-endian signed 16-bit
                w.writeframes(int_samples.tobytes())

            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

        # Use the v1 client audio transcription endpoint
        params = {"file": wav_buffer, "model": model}
        if language:
            params["language"] = language

        resp = client.audio.transcriptions.create(
            **params,
            prompt="Please transcribe the following audio precisely (don't fix mistakes).",
        )

        if DEBUG:
            print(resp)
            # save wav buffer for inspection
            with open("debug_speech_api_input.wav", "wb") as f:
                f.write(wav_buffer.getbuffer())
                f.flush()

        # Prefer attribute access which is the usual shape for the v1 SDK
        try:
            return resp.text
        except Exception:
            # Fallback to dict-like access for compatibility
            if isinstance(resp, dict):
                return resp.get("text", "")
            raise
    except Exception as exc:
        raise RuntimeError(f"speech_to_text failed: {exc}") from exc


@log_function_duration()
def text_to_speech(
    text: str,
    *,
    model: str = "gpt-4o-mini-tts",
    voice: Optional[str] = None,
    instructions: Optional[str] = None,
) -> np.ndarray:
    """Synthesize speech from text using an OpenAI model and return audio samples.

    Args:
        text: The input text to synthesize.
        model: The TTS model name to use.
        voice: Optional voice selection string, if supported by the model.
        instructions: Optional additional instructions for voice/style.

    Returns:
        A numpy array of float32 samples (mono, 16 kHz) containing the
        synthesized audio.

    Raises:
        RuntimeError: If the OpenAI package or API key is missing, or if
            the API returns an error.
    """
    _ensure_openai_available()
    if voice is None:
        voice = "coral"  # default voice

    if instructions is None:
        instructions = "Speak in a cheerful and positive tone."

    client = _get_client()
    from datetime import datetime
    from pathlib import Path

    speech_file_path = (
        Path(__file__).parent.parent.parent
        / "conversa_audio"
        / f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )
    # Ensure target directory exists
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
        instructions=instructions,
    ) as response:
        response.stream_to_file(speech_file_path)

    # Read the generated audio file and return as bytes
    data, samplerate = sf.read(str(speech_file_path), dtype="float32")
    # Ensure data is 1-D: (frames, channels)
    if data.ndim == 2:
        data = data[:, 0]

    if samplerate != 16000:
        data = librosa.resample(data.T, orig_sr=samplerate, target_sr=16000).T

    # Clean up the temporary file (keep for debugging during development)
    if not DEBUG:
        try:
            speech_file_path.unlink()
        except FileNotFoundError:
            # If the file was removed or not created, ignore
            pass

    return data

    # try:
    #     client = _get_client()

    #     # Use the v1 client TTS endpoint
    #     params = {"model": model, "input": text}
    #     if voice:
    #         params["voice"] = voice

    #     resp = client.audio.speech.create(**params)

    #     # If the SDK returns raw bytes directly
    #     if isinstance(resp, (bytes, bytearray)):
    #         return bytes(resp)

    #     # Prefer attribute access
    #     try:
    #         return bytes(resp.audio)
    #     except Exception:
    #         # Fallback to dict-like access
    #         if isinstance(resp, dict):
    #             for key in ("audio", "audio_content", "audio_base64"):
    #                 if key in resp:
    #                     val = resp[key]
    #                     if isinstance(val, str):
    #                         import base64

    #                         return base64.b64decode(val)
    #                     return bytes(val)
    #         raise
    # except Exception as exc:
    #     raise RuntimeError(f"text_to_speech failed: {exc}") from exc
