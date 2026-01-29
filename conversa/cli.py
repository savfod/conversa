"""CLI interface runner for conversa."""

import argparse

from conversa.audio.stream_factory import create_input_stream, create_output_stream
from conversa.scenarios.scenario_factory import create_scenario
from conversa.util.config import Config


def run(args: argparse.Namespace, config: Config) -> None:
    """Run scenario with CLI streams (microphone/speaker or file).

    Args:
        args: Parsed command-line arguments containing scenario name
              and optional input_file.
        config: Application configuration.
    """
    if args.input_file:
        input_stream = create_input_stream("file", file_path=args.input_file)
    else:
        input_stream = create_input_stream("microphone", sample_rate=16000)

    output_stream = create_output_stream("speaker", sample_rate=16000)

    scenario = create_scenario(args.scenario, input_stream, output_stream, config, args)
    try:
        scenario.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
        scenario.stop()
    finally:
        input_stream.stop()
        output_stream.stop()
