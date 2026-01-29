"""Unified CLI entry point for conversa.

Usage:
    uv run -m conversa <interface> <scenario> [options]

Examples:
    uv run -m conversa cli talk
    uv run -m conversa cli debug_mirror
    uv run -m conversa web talk --port 8080
    uv run -m conversa gradio talk --share

    # Override input/output for debugging:
    uv run -m conversa cli talk --input file:test.wav --output file:out.wav
    uv run -m conversa web talk --input file:test.wav  # file input, web output
"""

import argparse
from pathlib import Path

from conversa.scenarios.scenario_factory import SCENARIOS
from conversa.util.config import Config
from conversa.util.io import DEFAULT_SETTINGS_FILE
from conversa.util.logs import setup_logging

# Default stream types per interface
INTERFACE_DEFAULTS = {
    "cli": {"input": "microphone", "output": "speaker"},
    "web": {"input": "web", "output": "web"},
    "gradio": {"input": "gradio", "output": "gradio"},
}


def parse_stream_spec(spec: str) -> tuple[str, dict]:
    """Parse stream specification like 'file:/path/to.wav' or 'microphone'.

    Args:
        spec: Stream specification string.

    Returns:
        Tuple of (stream_type, kwargs_dict).
    """
    if ":" in spec:
        stream_type, path = spec.split(":", 1)
        if stream_type == "file":
            return stream_type, {"file_path": path}
        # For other types, ignore the part after ':'
        return stream_type, {}
    return spec, {}


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --input and --output arguments."""
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="TYPE[:PATH]",
        help="Input stream. Examples: microphone, file:/path/to.wav, web, gradio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="TYPE[:PATH]",
        help="Output stream. Examples: speaker, file:/path/to.wav, web, gradio",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Conversa - AI Conversational Language Teacher",
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
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="interface", required=True)

    # CLI interface
    cli_parser = subparsers.add_parser("cli", help="Command-line interface")
    _add_scenario_subparsers(cli_parser)

    # Web interface
    web_parser = subparsers.add_parser("web", help="Web interface (Flask/SocketIO)")
    web_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)"
    )
    web_parser.add_argument(
        "--port", type=int, default=5555, help="Port (default: 5555)"
    )
    _add_scenario_subparsers(web_parser)

    # Gradio interface
    gradio_parser = subparsers.add_parser("gradio", help="Gradio web interface")
    gradio_parser.add_argument(
        "--share", action="store_true", help="Create public link"
    )
    gradio_parser.add_argument(
        "--port", type=int, default=7860, help="Port (default: 7860)"
    )
    _add_scenario_subparsers(gradio_parser)

    return parser


def _add_scenario_subparsers(parent_parser: argparse.ArgumentParser) -> None:
    """Add scenario subparsers to the given parent parser."""
    scenario_subparsers = parent_parser.add_subparsers(dest="scenario", required=True)

    for name, scenario_def in SCENARIOS.items():
        sc_parser = scenario_subparsers.add_parser(name, help=f"Run {name} scenario")
        _add_io_arguments(sc_parser)
        scenario_def.scenario_class.add_arguments(sc_parser)


def _resolve_stream_args(args: argparse.Namespace) -> None:
    """Resolve input/output stream arguments with interface defaults.

    Modifies args in place, adding input_type, input_kwargs, output_type, output_kwargs.
    """
    defaults = INTERFACE_DEFAULTS[args.interface]

    # Parse input spec
    input_spec = args.input or defaults["input"]
    args.input_type, args.input_kwargs = parse_stream_spec(input_spec)

    # Parse output spec
    output_spec = args.output or defaults["output"]
    args.output_type, args.output_kwargs = parse_stream_spec(output_spec)


def main() -> None:
    """Main entry point."""
    args = build_parser().parse_args()

    setup_logging(level=args.log_level)
    config = Config.load(args.config)
    config.print_settings()

    _resolve_stream_args(args)

    if args.interface == "cli":
        from conversa import cli

        cli.run(args, config)
    elif args.interface == "web":
        from conversa.web import web

        web.run(args, config)
    elif args.interface == "gradio":
        from conversa.web import gradio_version

        gradio_version.run(args, config)


if __name__ == "__main__":
    main()
