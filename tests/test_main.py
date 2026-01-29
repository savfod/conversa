"""Tests for CLI argument parsing in conversa.__main__."""

import pytest

from conversa.__main__ import (
    INTERFACE_DEFAULTS,
    _resolve_stream_args,
    build_parser,
    parse_stream_spec,
)


class TestParseStreamSpec:
    """Tests for parse_stream_spec function."""

    def test_simple_type(self):
        stream_type, kwargs = parse_stream_spec("microphone")
        assert stream_type == "microphone"
        assert kwargs == {}

    def test_simple_type_speaker(self):
        stream_type, kwargs = parse_stream_spec("speaker")
        assert stream_type == "speaker"
        assert kwargs == {}

    def test_simple_type_web(self):
        stream_type, kwargs = parse_stream_spec("web")
        assert stream_type == "web"
        assert kwargs == {}

    def test_simple_type_gradio(self):
        stream_type, kwargs = parse_stream_spec("gradio")
        assert stream_type == "gradio"
        assert kwargs == {}

    def test_file_with_path(self):
        stream_type, kwargs = parse_stream_spec("file:/path/to/audio.wav")
        assert stream_type == "file"
        assert kwargs == {"file_path": "/path/to/audio.wav"}

    def test_file_with_relative_path(self):
        stream_type, kwargs = parse_stream_spec("file:./tmp.wav")
        assert stream_type == "file"
        assert kwargs == {"file_path": "./tmp.wav"}

    def test_file_with_path_containing_colons(self):
        # Windows-style path or path with colons
        stream_type, kwargs = parse_stream_spec("file:C:/Users/test/audio.wav")
        assert stream_type == "file"
        assert kwargs == {"file_path": "C:/Users/test/audio.wav"}

    def test_non_file_type_with_colon_ignores_path(self):
        stream_type, kwargs = parse_stream_spec("web:ignored")
        assert stream_type == "web"
        assert kwargs == {}


class TestBuildParser:
    """Tests for argument parser structure."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    @pytest.mark.parametrize("interface", ["cli", "web", "gradio"])
    @pytest.mark.parametrize("scenario", ["talk", "debug_mirror", "file_narrate"])
    def test_parse_interface_scenario(self, parser, interface, scenario):
        cmd = [interface, scenario]
        if scenario == "file_narrate":
            cmd.append("test.txt")  # required positional arg
        args = parser.parse_args(cmd)
        assert args.interface == interface
        assert args.scenario == scenario

    @pytest.mark.parametrize("interface", ["cli", "web", "gradio"])
    def test_input_output_args_available(self, parser, interface):
        args = parser.parse_args(
            [interface, "talk", "--input", "file:test.wav", "--output", "file:out.wav"]
        )
        assert args.input == "file:test.wav"
        assert args.output == "file:out.wav"

    def test_cli_defaults(self, parser):
        args = parser.parse_args(["cli", "talk"])
        assert args.input is None
        assert args.output is None

    def test_web_has_host_port(self, parser):
        # --host and --port are on interface parser, before scenario
        args = parser.parse_args(["web", "--host", "0.0.0.0", "--port", "8080", "talk"])
        assert args.host == "0.0.0.0"
        assert args.port == 8080

    def test_gradio_has_share_port(self, parser):
        # --share and --port are on interface parser, before scenario
        args = parser.parse_args(["gradio", "--share", "--port", "7777", "talk"])
        assert args.share is True
        assert args.port == 7777


class TestResolveStreamArgs:
    """Tests for _resolve_stream_args function."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    def test_cli_defaults(self, parser):
        args = parser.parse_args(["cli", "talk"])
        _resolve_stream_args(args)
        assert args.input_type == "microphone"
        assert args.input_kwargs == {}
        assert args.output_type == "speaker"
        assert args.output_kwargs == {}

    def test_web_defaults(self, parser):
        args = parser.parse_args(["web", "talk"])
        _resolve_stream_args(args)
        assert args.input_type == "web"
        assert args.input_kwargs == {}
        assert args.output_type == "web"
        assert args.output_kwargs == {}

    def test_gradio_defaults(self, parser):
        args = parser.parse_args(["gradio", "talk"])
        _resolve_stream_args(args)
        assert args.input_type == "gradio"
        assert args.input_kwargs == {}
        assert args.output_type == "gradio"
        assert args.output_kwargs == {}

    def test_override_input_with_file(self, parser):
        args = parser.parse_args(["cli", "talk", "--input", "file:/path/to/test.wav"])
        _resolve_stream_args(args)
        assert args.input_type == "file"
        assert args.input_kwargs == {"file_path": "/path/to/test.wav"}
        assert args.output_type == "speaker"  # default

    def test_override_output_with_file(self, parser):
        args = parser.parse_args(["cli", "talk", "--output", "file:./out.wav"])
        _resolve_stream_args(args)
        assert args.input_type == "microphone"  # default
        assert args.output_type == "file"
        assert args.output_kwargs == {"file_path": "./out.wav"}

    def test_override_both(self, parser):
        args = parser.parse_args(
            ["web", "talk", "--input", "file:in.wav", "--output", "speaker"]
        )
        _resolve_stream_args(args)
        assert args.input_type == "file"
        assert args.input_kwargs == {"file_path": "in.wav"}
        assert args.output_type == "speaker"
        assert args.output_kwargs == {}

    def test_cross_interface_streams(self, parser):
        # Use web streams in CLI mode (for debugging)
        args = parser.parse_args(["cli", "talk", "--input", "web", "--output", "web"])
        _resolve_stream_args(args)
        assert args.input_type == "web"
        assert args.output_type == "web"


class TestInterfaceDefaults:
    """Tests for INTERFACE_DEFAULTS constant."""

    def test_all_interfaces_have_defaults(self):
        assert "cli" in INTERFACE_DEFAULTS
        assert "web" in INTERFACE_DEFAULTS
        assert "gradio" in INTERFACE_DEFAULTS

    def test_defaults_have_input_output(self):
        for interface, defaults in INTERFACE_DEFAULTS.items():
            assert "input" in defaults, f"{interface} missing input default"
            assert "output" in defaults, f"{interface} missing output default"
