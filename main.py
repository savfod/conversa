import argparse
from pathlib import Path

from conversa.audio.stream_factory import create_input_stream, create_output_stream
from conversa.scenarios.talk import run_talk_scenario
from conversa.util.config import Config
from conversa.util.io import DEFAULT_SETTINGS_FILE
from conversa.util.logs import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Conversational Teacher")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to an audio file to process instead of using the microphone.",
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)

    config = Config.load(args.config)
    config.print_settings()

    # Create streams based on arguments
    if args.file:
        print("Starting AudioFileInputStream...")
        input_stream = create_input_stream("file", file_path=args.file)
    else:
        print("Starting MicrophoneInputStream...")
        input_stream = create_input_stream("microphone", sample_rate=16000)

    output_stream = create_output_stream("speaker", sample_rate=16000)

    try:
        run_talk_scenario(input_stream, output_stream, config=config)
    finally:
        input_stream.stop()
        output_stream.stop()
