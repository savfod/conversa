"""File reader with translation and text-to-speech support.

This module provides functionality to:
1. Load text files (with future support for epub)
2. Process content in parts
3. Translate to simplified language versions
4. Read content using text-to-speech
"""

import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from conversa.audio.audio_parser import AudioParser
from conversa.audio.input_stream import MicrophoneInputStream
from conversa.audio.input_stream.base import AbstractAudioInputStream
from conversa.features.llm_api import call_llm
from conversa.generated.output_stream.base import AbstractAudioOutputStream
from conversa.generated.output_stream.speaker import SpeakerOutputStream
from conversa.generated.speech_api import speech_to_text, text_to_speech
from conversa.util.io import DEFAULT_READING_STATUS, read_json, write_json
from conversa.util.logs import get_logger

logger = get_logger(__name__)


class CommandListener:
    """Listens for voice commands and triggers callbacks.

    This class monitors microphone input for voice commands (specifically 'start')
    and executes callbacks when commands are detected or stop conditions are met.
    Can be used in disabled mode for simple wait-only behavior.
    """

    def __init__(
        self,
        input_stream: AbstractAudioInputStream | None = None,
        output_stream: AbstractAudioOutputStream | None = None,
        audio_parser: AudioParser | None = None,
    ):
        """Initialize the command listener.

        Args:
            input_stream: Audio input stream for capturing audio (None for disabled mode)
            output_stream: Audio output stream for feedback (None for disabled mode)
            audio_parser: Audio parser for detecting voice commands (None for disabled mode)
        """
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.audio_parser = audio_parser
        self.enabled = (
            input_stream is not None
            and output_stream is not None
            and audio_parser is not None
        )

    def _start_waiting_loop(
        self,
        stop_condition_callback: Callable[[], bool],
        on_start_detected_callback: Callable[[], None],
    ) -> bool:
        """Wait for either stop condition or 'start' command detection.

        This method runs a polling loop that:
        1. Checks if stop_condition_callback returns True (e.g., playback finished)
        2. Processes microphone input for voice commands (if enabled)
        3. Calls on_start_detected_callback if 'start' command is detected

        Args:
            stop_condition_callback: Function that returns True when waiting should stop
                                    (e.g., lambda: not sd.get_stream().active)
            on_start_detected_callback: Function to call when 'start' command detected
                                       (e.g., lambda: sd.stop())

        Returns:
            True if 'start' command was detected, False if stopped by stop condition

        """
        assert self.enabled, "Cannot use voice control loop when disabled"
        assert self.input_stream is not None
        assert self.audio_parser is not None
        # Voice control enabled - monitor for commands
        print("Listening for 'start' command to pause... or stop condition met.")
        while not stop_condition_callback():
            time.sleep(0.1)

            # Get and process microphone input
            chunk = self.input_stream.get_unprocessed_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            status, speech, status_changed = self.audio_parser.add_chunk(chunk)

            # If 'start' command detected, trigger callback and wait for 'stop stop'
            if status == "listening":
                print("\n[Voice Control] Paused. Say 'stop stop' to resume...")
                on_start_detected_callback()
                return True

        return False

    def _wait_for_stop_command(self) -> np.ndarray:
        """Wait for 'stop stop' command to resume from pause.

        This is a blocking method that continuously processes microphone input
        until the AudioParser detects transition back to 'waiting' status.
        Only works when enabled.
        """
        assert self.enabled, "Cannot wait for command when voice control is disabled"
        assert self.input_stream is not None
        assert self.audio_parser is not None

        print("Waiting for 'stop stop' command to resume...")
        while True:
            time.sleep(0.1)
            chunk = self.input_stream.get_unprocessed_chunk()
            if chunk is None or len(chunk) == 0:
                continue

            status, speech, status_changed = self.audio_parser.add_chunk(chunk)
            if status == "waiting":
                # 'stop stop' detected - exit pause state
                assert speech is not None
                return speech

    def run_voice_control_loop(
        self,
        context_text: str,
        stop_condition_callback: Callable[[], bool],
        on_start_detected_callback: Callable[[], None],
    ) -> Literal["run_again"] | None:
        """
        Running voice control loop

        Args:
            stop_condition_callback: Function that returns True when waiting should stop
            on_start_detected_callback: Function to call when 'start' command detected

        Returns:
            'run_again' if playback should be repeated, None otherwise
        """
        if not self.enabled:
            # Simple wait mode - just wait for stop condition
            while not stop_condition_callback():
                time.sleep(0.1)
            return None

        start_detected = self._start_waiting_loop(
            stop_condition_callback=stop_condition_callback,
            on_start_detected_callback=on_start_detected_callback,
        )

        if start_detected:
            speech = self._wait_for_stop_command()
            transcription = speech_to_text(speech)
            print("Transcription of the command:", transcription)

            if "resume" in transcription.lower():
                print("[Voice Control] Resuming playback...")
                return None
                # Continue to next chunk

            else:
                # Handle LLM interaction
                about_text = f"\n(about the text {context_text})"
                llm_response = call_llm(
                    query=f"""The user asked the following (text to speech may have errors): "{transcription}" {about_text}.""",
                    sys_prompt="You are a helpful assistant.",
                )
                audio = text_to_speech(
                    llm_response,
                    instructions="Read the text in a clear and friendly tone.",
                )
                assert self.output_stream is not None
                self.output_stream.play_chunk(audio)
                self.output_stream.wait()
                return "run_again"


class ReadingStatus:
    """Class to manage reading status persistence."""

    def __init__(self, file_id: str, status_file: str | Path = DEFAULT_READING_STATUS):
        self.file_id = file_id
        self.status_file = Path(status_file)
        self.status = self._load_status()

    def _load_status(self) -> dict:
        """Load reading status from the JSON file."""
        if self.status_file.exists():
            return read_json(self.status_file)
        return {}

    def get_last_position(self) -> int:
        """Get the last read position for a given file."""
        return self.status.get(self.file_id, 0)

    def update_position(self, position: int) -> None:
        """Update the last read position for a given file."""
        self.status[self.file_id] = position
        write_json(self.status, self.status_file)

    def print_info(self):
        """Print current reading status info."""
        last_pos = self.get_last_position()
        print(
            f"File ID: {self.file_id}, read position: {last_pos}."
            "\nThe setting can be found in the file:"
            f"\nfile://{self.status_file}"
        )


@dataclass
class Chunk:
    i: int
    text: str
    end_position: int
    audio: np.ndarray | None = None


class FileChunkLoader:
    """Handles loading files and splitting text into chunks.

    This class is responsible for:
    - Loading text from files (txt, epub)
    - Splitting text into chunks of appropriate size
    - Breaking at sentence boundaries when possible
    - Tracking current position in the text

    It has no knowledge of simplification, TTS, reading status persistence, or other processing.
    """

    def __init__(
        self,
        file_path: str | Path,
        start_position: int = 0,
        chunk_size: int = 500,
    ):
        """Initialize the chunk loader.

        Args:
            file_path: Path to the file to load
            start_position: Character position to start from (default: 0)
            chunk_size: Target number of characters per chunk (default: 500)
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.text = self._load_file()
        self.chunk_size = chunk_size
        self.current_position = start_position
        self.text_length = len(self.text)
        self.chunk_index = 0

    def _load_txt(self) -> str:
        """Load content from a text file.

        Returns:
            The full text content of the file
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_epub(self) -> str:
        """Load content from an epub file.

        Returns:
            The full text content of the epub
        """
        all_text = []
        book = epub.read_epub(self.file_path)
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                content = item.get_content()
                soup = BeautifulSoup(content, "html.parser")
                all_text.append(soup.get_text())

        full_text = "\n\n".join(all_text)
        return full_text

    def _load_file(self) -> str:
        """Load data from the file based on its extension.

        Returns:
            The full text content

        Raises:
            ValueError: If the file format is not supported
        """
        extension = self.file_path.suffix.lower()

        if extension == ".txt":
            return self._load_txt()
        elif extension == ".epub":
            return self._load_epub()
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def has_more_chunks(self) -> bool:
        """Check if there are more chunks to process.

        Returns:
            True if there are more chunks available
        """
        return self.current_position < self.text_length

    def get_next_chunk(self) -> Chunk:
        """Get the next chunk of text.

        This method advances the internal position and returns a Chunk object.
        It tries to break at sentence boundaries when possible.

        Returns:
            Chunk object or None if no more chunks
        """
        start = self.current_position
        end = start + self.chunk_size

        # If we're not at the end, try to find a sentence boundary
        if end < self.text_length:
            # Look for sentence endings in the next 100 characters
            search_end = min(end + 100, self.text_length)
            chunk = self.text[start:search_end]

            # Find the last sentence ending
            for separator in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                last_sep = chunk.rfind(separator)
                if last_sep != -1:
                    end = start + last_sep + len(separator)
                    break

        chunk_text = self.text[start:end].strip()
        self.current_position = end
        self.chunk_index += 1
        return Chunk(i=self.chunk_index, text=chunk_text, end_position=end)


class ContentProcessor:
    """Manages text processing pipeline: simplification, translation, and TTS.

    This class is responsible for:
    - Simplifying text to target language level
    - Translating text if needed
    - Converting text to speech
    - Preparing chunks with all processing steps combined
    """

    def __init__(
        self,
        simplify: bool = False,
        target_language: str = "English",
        simplification_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "B1",
    ):
        """Initialize the content processor.

        Args:
            simplify: Whether to simplify the text
            target_language: Target language for simplification
            simplification_level: CEFR level for simplification (A1-C2)
        """
        self.simplify = simplify
        self.target_language = target_language
        self.simplification_level = simplification_level

    def _simplify_text(self, text: str) -> str:
        """Translate text if required and simplify to the specified language level.

        Args:
            text: The text to simplify

        Returns:
            Simplified text
        """
        print(
            f"Simplifying text... to language {self.target_language}, level {self.simplification_level}"
        )
        user_prompt = f"""Translate the following text if required and simplify it to the {self.simplification_level} level of language {self.target_language}.
Keep the meaning intact but use simpler vocabulary and sentence structures appropriate for {self.simplification_level} learners. If required, use longer explanations. Very rare words could be described in English also. Include only the final simplified text in your response.

Text to simplify:
{text}"""

        sys_prompt = "You are a language teaching assistant that simplifies texts for language learners."

        print("Original text to simplify:")
        print(text)
        simplified = call_llm(query=user_prompt, sys_prompt=sys_prompt)
        return simplified.strip()

    def _text_to_speech(self, text: str) -> np.ndarray:
        """Convert text chunk to speech audio.

        Args:
            text: The text to convert to speech

        Returns:
            Audio data as numpy array
        """
        return text_to_speech(
            text,
            instructions=f"Read the text in clear {self.target_language} pronunciation, please. Use speed appropriate for language learners on {self.simplification_level} level. Be expressive and cheerful.",
        )

    def prepare_chunk(
        self, text_chunk: str, chunk_index: int
    ) -> tuple[str, np.ndarray]:
        """Prepare a chunk for playback: simplify if needed and convert to audio.

        Args:
            text_chunk: Text to prepare
            chunk_index: Index of the current chunk being processed

        Returns:
            Tuple of (processed_text, audio_data)
        """
        print(f"\n--- Chunk {chunk_index} ---")
        if self.simplify:
            text_chunk = self._simplify_text(text_chunk)
            print("--")
            print("Translated & Simplified text:")

        print(text_chunk)
        print()

        audio = self._text_to_speech(text_chunk)
        return text_chunk, audio


class ChunkAsyncPreprocessor:
    """Manages chunk generation and preparation pipeline with background processing."""

    def __init__(
        self,
        chunk_loader: FileChunkLoader,
        content_processor: ContentProcessor,
        reading_status: ReadingStatus,
    ):
        """Initialize the chunk generator.

        Args:
            chunk_loader: Handles loading and chunking the file
            content_processor: Handles simplification and TTS
            reading_status: Manages reading position persistence
        """
        self._chunk_loader = chunk_loader
        self._content_processor = content_processor
        self._reading_status = reading_status
        self._prepared_chunk: Chunk | None = None
        self._preparation_future: Future[Chunk] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        # state for iterator/commands
        self._run_again: bool = False

    def __iter__(self):
        return self

    def __next__(self) -> Chunk | None:
        """Get the next prepared chunk if available. Not-blocking function.

        Returns:
            Prepared chunk or None if no more chunks
        """
        self._tick()
        if self._finished():
            raise StopIteration

        return self._prepared_chunk

    def _tick(self) -> None:
        """Main loop tick to advance the chunk preparation pipeline.
        _input_stream -> chunk_to_prepare -> prepared_chunk -> _trash
        """
        if self._run_again:
            self._run_again = False
            return

        # prepared_chunk -> _trash
        if self._prepared_chunk is not None:
            self._finalize_current_chunk()

        # chunk_to_prepare -> prepared_chunk
        if self._preparation_future is not None:
            if not self._preparation_future.done():
                logger.warning("Preparation future is expected to be done in tick()")

            try:
                chunk = self._preparation_future.result()
                self._prepared_chunk = chunk
                self._preparation_future = None

            except Exception as e:
                logger.exception(
                    f"Error {e} preparing chunk in background, it'll be omitted"
                )
                self._preparation_future = None

        # chunk_loader -> chunk_to_prepare
        if self._preparation_future is None and self._chunk_loader.has_more_chunks():
            next_chunk = self._chunk_loader.get_next_chunk()
            self._preparation_future = self._executor.submit(
                self._prepare_chunk_sync, next_chunk
            )

    def is_preparation_active(self) -> bool:
        """Check if background preprocessing is currently running.

        Returns:
            True if a chunk is being prepared in the background
        """
        return (
            self._preparation_future is not None and not self._preparation_future.done()
        )

    def _finished(self) -> bool:
        """Return True when no more data will be produced."""
        return (
            self._prepared_chunk is None
            and self._preparation_future is None
            and not self._chunk_loader.has_more_chunks()
        )

    def command(self, cmd: Literal["run_again"] | None = None) -> None:
        """Handle commands to control the chunk generator.

        Args:
            cmd: Command to execute (e.g., 'run_again' to reset state)
        """
        if cmd is None:
            return

        elif cmd == "run_again":
            # Keep the last served chunk available as the prepared chunk
            self._run_again = True

        else:
            raise RuntimeError(f"Unknown command to ChunkAsyncPreprocessor: {cmd}")

    def _prepare_chunk_sync(self, text_chunk: Chunk) -> Chunk:
        """Synchronously prepare a chunk (runs in background thread).

        Args:
            text_chunk: Chunk with text to process

        Returns:
            Chunk with audio prepared
        """
        _text, audio = self._content_processor.prepare_chunk(
            text_chunk.text, text_chunk.i
        )
        return Chunk(
            i=text_chunk.i,
            text=_text,
            audio=audio,
            end_position=text_chunk.end_position,
        )

    def _finalize_current_chunk(self) -> None:
        """Finalize the current chunk and advance to next."""
        assert self._prepared_chunk is not None
        self._reading_status.update_position(self._prepared_chunk.end_position)
        self._prepared_chunk = None

    # def shutdown(self) -> None:
    #     """Shutdown the background thread pool."""
    #     self._executor.shutdown(wait=True)


class FileNarrator:
    """Narrates files with optional translation and TTS."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 500,
        simplify: bool = False,
        target_language: str = "English",
        simplification_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = "B1",
        enable_voice_control: bool = False,
        input_stream: AbstractAudioInputStream | None = None,
        output_stream: AbstractAudioOutputStream | None = None,
    ):
        """Initialize the file narrator.

        Args:
            file_path: Path to the file to narrate
            chunk_size: Number of characters per chunk
            simplify: Whether to simplify the text
            target_language: Target language for simplification
            simplification_level: CEFR level for simplification (A1-C2)
            enable_voice_control: Enable voice commands for pause/resume control
            input_stream: Audio input stream for voice control
            output_stream: Audio output stream for playback
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.reading_status = ReadingStatus(str(self.file_path))
        self.enable_voice_control = enable_voice_control
        self._content_processor = ContentProcessor(
            simplify=simplify,
            target_language=target_language,
            simplification_level=simplification_level,
        )
        self.input_stream = input_stream
        self.output_stream = output_stream

    def _setup_voice_control(self) -> CommandListener:
        """Initialize voice-control helpers and return command_listener.

        Uses self.input_stream and self.output_stream if available.
        """
        if self.enable_voice_control:
            print("Starting microphone for voice control...")
            print("Say 'start' to pause, then 'stop stop' to resume")

            if self.input_stream is None:
                self.input_stream = MicrophoneInputStream(sample_rate=16000)
                self.input_stream.start()

            audio_parser = AudioParser(
                model_path="vosk-model-small-en-us-0.15", sample_rate=16000
            )
            command_listener = CommandListener(
                self.input_stream, self.output_stream, audio_parser
            )
        else:
            command_listener = CommandListener(
                output_stream=self.output_stream
            )  # Disabled mode

        return command_listener

    def read_file(self) -> None:
        """Read the entire file using clean control flow architecture."""
        try:
            # Ensure output stream
            if self.output_stream is None:
                self.output_stream = SpeakerOutputStream(sample_rate=16000)

            command_listener = self._setup_voice_control()

            chunk_loader = FileChunkLoader(
                file_path=self.file_path,
                start_position=self.reading_status.get_last_position(),
                chunk_size=self.chunk_size,
            )
            chunk_preprocessor = ChunkAsyncPreprocessor(
                chunk_loader, self._content_processor, self.reading_status
            )

            for chunk in chunk_preprocessor:
                if chunk is not None:
                    # using output_stream instead of sd.play
                    self.output_stream.play_chunk(chunk.audio)

                def stop_condition() -> bool:
                    assert self.output_stream is not None
                    return (
                        chunk is None or not self.output_stream.is_playing()
                    ) and not chunk_preprocessor.is_preparation_active()

                command = command_listener.run_voice_control_loop(
                    context_text=chunk.text
                    if chunk is not None
                    else "[No text available]",
                    stop_condition_callback=stop_condition,
                    on_start_detected_callback=lambda: self.output_stream.stop(),  # Using stop on stream?
                    # Note: sd.stop() stops playback. Stream.stop() might close stream.
                    # We might want a way to just stop playback or pause?
                    # SpeakerOutputStream.stop closes everything.
                    # sd.stop() just stops the current playback?
                    # If we use SpeakerOutputStream, we might not have a 'pause' or 'cancel current' easily.
                    # But stopping the stream is fine if we restart it or if play_chunk handles it.
                    # Actually SpeakerOutputStream.stop() sets _started=False. play_chunk restarts.
                    # So calling stop() is okay if play_chunk handles restart.
                )
                chunk_preprocessor.command(command)

        finally:
            if self.input_stream:
                self.input_stream.stop()
                print("Microphone stopped.")
            if self.output_stream:
                self.output_stream.stop()


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Read and process text files with optional simplification and TTS"
    )
    parser.add_argument("file_path", type=str, help="Path to the file to read")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of characters per chunk (default: 500)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the text to a target language level",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Target language for simplification (default: English)",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["A1", "A2", "B1", "B2", "C1", "C2"],
        default="B1",
        help="CEFR level for simplification (default: B1)",
    )
    parser.add_argument(
        "--voice-control",
        action="store_true",
        help="Enable voice commands to pause/resume playback (say 'start' to pause, 'stop stop' to resume)",
    )

    args = parser.parse_args()

    output_stream = SpeakerOutputStream(sample_rate=16000)
    input_stream = None
    if args.voice_control:
        input_stream = MicrophoneInputStream(
            sample_rate=16000,
        )
        input_stream.start()

    try:
        narrator = FileNarrator(
            file_path=args.file_path,
            chunk_size=args.chunk_size,
            simplify=args.simplify,
            target_language=args.language,
            simplification_level=args.level,
            enable_voice_control=args.voice_control,
            input_stream=input_stream,
            output_stream=output_stream,
        )

        narrator.read_file()
    finally:
        output_stream.stop()
        if input_stream:
            input_stream.stop()


if __name__ == "__main__":
    main()
