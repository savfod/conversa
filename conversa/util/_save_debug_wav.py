"""Debug CLI to save WAV files from file, microphone, or TTS.

This script is intended as a small debugging utility. It supports three modes:

- file: read an audio file via the project's `AudioFileInputStream` and save one buffered chunk
- mic: record from microphone using `MicrophoneInputStream` for a given duration and save the captured audio
- tts: synthesize speech using `speech_api.text_to_speech` and save the produced audio

Usage examples:

    python -m conversa.util._save_debug_wav file --input audio.mp3 --out out.wav
    python -m conversa.util._save_debug_wav mic --duration 5 --out mic.wav
    python -m conversa.util._save_debug_wav tts --text "Hello world" --out tts.wav

This file intentionally has no tests (debug helper).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from conversa import speech_api
from conversa.audio.input_stream import (
    AudioFileInputStream,
    MicrophoneInputStream,
)


def _save_array_to_wav(data: np.ndarray, sample_rate: int, out_path: Path) -> None:
    """Save a numpy array (float32 or int16) to WAV using soundfile.

    Args:
        data: 1-D array of samples (mono) or 2-D (frames, channels). If 2-D, the first channel
            will be used.
        sample_rate: Sample rate in Hz.
        out_path: Path to write the WAV file.
    """
    # Normalize shape to 1-D (mono) by taking first channel if necessary
    if data.ndim == 2:
        data = data[:, 0]

    # soundfile expects float values in [-1, 1] or integer types
    if np.issubdtype(data.dtype, np.floating):
        write_data = data.astype("float32")
    else:
        # Convert integer types to float32 in range [-1,1]
        info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) else None
        if info is not None:
            write_data = data.astype("float32") / float(info.max)
        else:
            write_data = data.astype("float32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), write_data, sample_rate, format="WAV")
    print(
        f"Saved WAV: {out_path} ({len(write_data) / sample_rate:.2f} s @ {sample_rate} Hz)"
    )


def _handle_file_mode(
    input_file: str,
    out_path: str,
    timeout: float = 30.0,
    duration: Optional[float] = None,
) -> None:
    """Start AudioFileInputStream and save either the first chunk (default) or the
    first `duration` seconds from the start of the file.

    The file stream simulates real-time playback and pushes chunks into its buffer.
    If `duration` is None, the function behaves like before and saves the first
    available chunk. If `duration` is provided, the function collects chunks until
    the requested duration (or timeout) and saves the concatenated audio.
    """
    stream = AudioFileInputStream(input_file)
    stream.start()

    try:
        # Unified logic: decide how many samples to collect. If `duration` is
        # provided, collect until that many samples; otherwise collect until the
        # first non-empty chunk is received.
        trim_to_target = duration is not None
        target_samples = int(duration * stream.sample_rate) if trim_to_target else 1

        collected = []
        total = 0
        start = time.time()
        while time.time() - start < timeout and total < target_samples:
            chunk = stream.get_unprocessed_chunk()
            if chunk is not None and len(chunk) > 0:
                arr = np.array(chunk, dtype=np.float32)
                collected.append(arr)
                total += arr.shape[0]
            else:
                time.sleep(0.05)

        if not collected:
            print("No audio received from file stream")
            return

        data = np.concatenate(collected, axis=0)
        if trim_to_target:
            data = data[:target_samples]

        _save_array_to_wav(data, stream.sample_rate, Path(out_path))

    finally:
        stream.stop()


def _handle_mic_mode(
    duration: float,
    out_path: str,
    device: Optional[int] = None,
    sample_rate: int = 16000,
) -> None:
    """Record from microphone for `duration` seconds and save the captured audio.

    Args:
        duration: seconds to record
        out_path: output WAV path
        device: optional device id passed to sounddevice
        sample_rate: sample rate in Hz
    """
    stream = MicrophoneInputStream(sample_rate=sample_rate, device=device)
    stream.start()

    print(f"Recording from microphone for {duration} s...")
    try:
        elapsed = 0.0
        collected = []
        poll_interval = 0.25
        while elapsed < duration:
            time.sleep(poll_interval)
            elapsed += poll_interval
            chunk = stream.get_unprocessed_chunk()
            if chunk is not None and len(chunk) > 0:
                collected.append(np.array(chunk, dtype=np.float32))
        # final drain
        chunk = stream.get_unprocessed_chunk()
        if chunk is not None and len(chunk) > 0:
            collected.append(np.array(chunk, dtype=np.float32))

        if collected:
            data = np.concatenate(collected, axis=0)
        else:
            data = np.array([], dtype=np.float32)

        if data.size == 0:
            print("No audio captured from microphone.")
        else:
            _save_array_to_wav(data, stream.sample_rate, Path(out_path))
    finally:
        stream.stop()


def _handle_tts_mode(
    text: str, out_path: str, sample_rate: int = 16000, voice: Optional[str] = None
) -> None:
    """Generate speech from `text` using `speech_api.text_to_speech` and save to out_path.

    The project's `speech_api.text_to_speech` may return either raw bytes, or a numpy
    array of float32 samples. Handle both.
    """
    print(f"Synthesizing TTS for text: {text!r}")
    result = speech_api.text_to_speech(text, model="gpt-4o-mini-tts", voice=voice)

    # Normalize out_path to Path for consistent file operations
    out_p = Path(out_path)
    # If the API returns a numpy array, save directly. If bytes, try to interpret.
    if isinstance(result, (bytes, bytearray)):
        # Write bytes to a temporary file path and then read with soundfile to normalize
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(result)
        print(
            f"Wrote raw bytes to {out_p}; attempting to read and re-save at {sample_rate} Hz"
        )
        try:
            data, sr = sf.read(str(out_p), dtype="float32")
            if data.ndim == 2:
                data = data[:, 0]
            if sr != sample_rate:
                import librosa

                data = librosa.resample(data.T, orig_sr=sr, target_sr=sample_rate).T
            _save_array_to_wav(data, sample_rate, out_p)
        except Exception as exc:
            print(f"Failed to normalize raw audio bytes: {exc}")
    else:
        # Assume numpy array-like
        try:
            import numpy as _np

            arr = _np.asarray(result)
        except Exception:
            raise RuntimeError(
                "Unsupported TTS result type from speech_api.text_to_speech"
            )

        if arr.size == 0:
            print("TTS returned empty audio data")
            return

        _save_array_to_wav(arr.astype("float32"), sample_rate, out_p)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug helper: save WAV from file, mic, or TTS"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_file = sub.add_parser(
        "file", help="Create WAV from an audio file using AudioFileInputStream"
    )
    p_file.add_argument(
        "--input", "-i", required=True, help="Input audio file path (mp3/wav/etc.)"
    )
    p_file.add_argument("--out", "-o", required=True, help="Output WAV file path")
    p_file.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=30.0,
        help="Timeout waiting for chunk (seconds)",
    )
    p_file.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Seconds to save from start (default: one chunk)",
    )

    p_mic = sub.add_parser("mic", help="Record microphone and save to WAV")
    p_mic.add_argument(
        "--duration", "-d", type=float, default=5.0, help="Seconds to record"
    )
    p_mic.add_argument("--out", "-o", required=True, help="Output WAV file path")
    p_mic.add_argument(
        "--device", type=int, default=None, help="Optional sounddevice device id"
    )
    p_mic.add_argument("--sr", type=int, default=16000, help="Sample rate to use")

    p_tts = sub.add_parser("tts", help="Synthesize text to speech and save to WAV")
    p_tts.add_argument("--text", "-t", required=True, help="Text to synthesize")
    p_tts.add_argument("--out", "-o", required=True, help="Output WAV file path")
    p_tts.add_argument(
        "--sr", type=int, default=16000, help="Target sample rate to save"
    )
    p_tts.add_argument(
        "--voice", type=str, default=None, help="Optional TTS voice name"
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "file":
        _handle_file_mode(
            args.input, args.out, timeout=args.timeout, duration=args.duration
        )
    elif args.mode == "mic":
        _handle_mic_mode(
            args.duration, args.out, device=args.device, sample_rate=args.sr
        )
    elif args.mode == "tts":
        _handle_tts_mode(args.text, args.out, sample_rate=args.sr, voice=args.voice)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
