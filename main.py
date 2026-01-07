import argparse

from conversa.audio.input_stream import AudioFileInputStream, MicrophoneInputStream
from conversa.generated.output_stream.speaker import SpeakerOutputStream
from conversa.generated.scenario.talk import run_talk_scenario
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
        "language",
        default="en",
        nargs="?",
        help="Language code for speech recognition and processing (default: en).",
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

    # Create streams based on arguments
    if args.file:
        print("Starting AudioFileInputStream...")
        input_stream = AudioFileInputStream(file_path=args.file)
    else:
        print("Starting MicrophoneInputStream...")
        input_stream = MicrophoneInputStream(sample_rate=16000)

    output_stream = SpeakerOutputStream(sample_rate=16000)

    try:
        run_talk_scenario(input_stream, output_stream, language=args.language)
    finally:
        input_stream.stop()
        output_stream.stop()
