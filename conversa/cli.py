"""CLI interface runner for conversa."""

import argparse

from conversa.audio.stream_factory import create_input_stream, create_output_stream
from conversa.scenarios.scenario_factory import create_scenario
from conversa.util.config import Config

DEFAULT_SAMPLE_RATE = 16000


def run(args: argparse.Namespace, config: Config) -> None:
    """Run scenario with configurable streams.

    Args:
        args: Parsed command-line arguments containing:
              - scenario: scenario name
              - input_type, input_kwargs: input stream config
              - output_type, output_kwargs: output stream config
        config: Application configuration.
    """
    # Build input stream kwargs with defaults
    input_kwargs = {"sample_rate": DEFAULT_SAMPLE_RATE, **args.input_kwargs}
    input_stream = create_input_stream(args.input_type, **input_kwargs)

    # Build output stream kwargs with defaults
    output_kwargs = {"sample_rate": DEFAULT_SAMPLE_RATE, **args.output_kwargs}
    output_stream = create_output_stream(args.output_type, **output_kwargs)

    scenario = create_scenario(args.scenario, input_stream, output_stream, config, args)
    try:
        scenario.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
        scenario.stop()
    finally:
        input_stream.stop()
        output_stream.stop()
