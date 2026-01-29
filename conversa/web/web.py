"""Web (Flask/SocketIO) interface runner for conversa."""

import argparse
import time
from pathlib import Path
from threading import Thread

import numpy as np

import conversa.web.io  # noqa: F401 - registers web streams
from conversa.audio.speech_api import speech_to_text, text_to_speech
from conversa.audio.stream_factory import create_input_stream, create_output_stream
from conversa.features.llm_api import call_llm
from conversa.scenarios.scenario_factory import create_scenario
from conversa.scenarios.talk import run_talk_scenario
from conversa.util.config import Config
from conversa.util.io import DEFAULT_SETTINGS_FILE
from conversa.util.logs import setup_logging
from conversa.web import server

CHUNK_SIZE = 16000 * 5  # e.g. 5 second @ 16kHz


DEFAULT_SAMPLE_RATE = 16000


def run(args: argparse.Namespace, config: Config) -> None:
    """Run scenario with configurable streams.

    Args:
        args: Parsed command-line arguments containing:
              - scenario: scenario name
              - input_type, input_kwargs: input stream config
              - output_type, output_kwargs: output stream config
              - host, port: server config
        config: Application configuration.
    """

    def worker() -> None:
        # Build input stream kwargs with defaults
        input_kwargs = {
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "channels": 1,
            **args.input_kwargs,
        }
        input_stream = create_input_stream(args.input_type, **input_kwargs)

        # Build output stream kwargs with defaults
        output_kwargs = {
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "channels": 1,
            **args.output_kwargs,
        }
        output_stream = create_output_stream(args.output_type, **output_kwargs)

        scenario = create_scenario(
            args.scenario, input_stream, output_stream, config, args
        )
        try:
            scenario.start()
        except KeyboardInterrupt:
            scenario.stop()
        finally:
            input_stream.stop()
            output_stream.stop()

    Thread(target=worker, daemon=True).start()
    server.run_server(host=args.host, port=args.port)


def process_audio(full_audio: np.ndarray, debug: bool = False) -> np.ndarray | None:
    """Process full audio chunk and return processed audio.
    Args:
        full_audio: NumPy array of shape (n,) dtype float32.
        debug: If True, prints debug information.
    Returns:
        Processed audio as NumPy array of shape (n,) dtype float32.
    """
    if debug:
        time.sleep(1)  # Simulate processing delay
        return full_audio

    text = speech_to_text(full_audio, sample_rate=16000, language="en")
    if text != "" and not text.strip().startswith(
        "Please transcribe the following audio"
    ):
        print(f"processing text '{text}'")
        answer = call_llm(text, sys_prompt="You are a helpful assistant.")
        # TODO: Replace with your logic / ML model / filtering
        return text_to_speech(answer)  # For now: identity
    else:
        print(f"skipped text '{text}'")
        return None


def audio_worker(config: Config, debug: bool = False) -> None:
    """
    Continuously collects audio chunks from WebInputStream.
    Accumulates enough samples → process_audio() → send back using WebOutputStream.

    Args:
        config: Application configuration.
        debug: If True, prints debug information.
    """
    # Initialize streams
    input_stream = create_input_stream("web", sample_rate=16000, channels=1)
    output_stream = create_output_stream("web", sample_rate=16000, channels=1)

    if not debug:
        print("Starting production scenario (Web based)...")
        try:
            run_talk_scenario(input_stream, output_stream, config=config)
        except Exception as e:
            print(f"Scenario failed: {e}")
        finally:
            input_stream.stop()
            output_stream.stop()
        return

    input_stream.start()

    print("Streams started (debug mode). Waiting for audio...")

    buffer = np.array([], dtype=np.float32)

    try:
        while True:
            # Get new data
            chunk = input_stream.get_unprocessed_chunk()
            if chunk is not None:
                buffer = np.concatenate((buffer, chunk))

            # Process if enough data
            if len(buffer) >= CHUNK_SIZE:
                to_process = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]  # Keep remainder

                # Run your processing
                processed = process_audio(to_process, debug=debug)

                if processed is not None:
                    # Send back
                    output_stream.play_chunk(processed)

            # Small sleep to avoid busy loop if no data
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        input_stream.stop()
        output_stream.stop()


def arg_parser():
    parser = argparse.ArgumentParser(description="Conversa Web Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_SETTINGS_FILE,
        help=f"Path to config file (default: {DEFAULT_SETTINGS_FILE})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the web server"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port for the web server"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    setup_logging(level=args.log_level)

    config = Config.load(args.config)
    config.print_settings()

    # Start the worker logic in a separate thread
    Thread(target=audio_worker, daemon=True, args=(config, args.debug)).start()

    # Run the server
    # Note: We run this in the main thread as it blocks
    server.run_server(host=args.host, port=args.port, debug=True)
