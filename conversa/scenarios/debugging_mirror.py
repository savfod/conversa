"""Debugging scenario that mirrors input audio to output."""

import argparse
import time

from conversa.audio.stream_factory import create_input_stream, create_output_stream


def run_mirror(
    input_type: str,
    output_type: str,
    sample_rate: int = 16000,
    input_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Mirror audio from input to output.

    Args:
        input_type: Type of input stream ("microphone", "file", "web", "gradio").
        output_type: Type of output stream ("speaker", "file", "web", "gradio").
        sample_rate: Audio sample rate.
        input_file: Path to input file (required if input_type is "file").
        output_file: Path to output file (required if output_type is "file").
    """
    # Build kwargs for streams
    input_kwargs: dict = {"sample_rate": sample_rate}
    output_kwargs: dict = {"sample_rate": sample_rate}

    if input_type == "file":
        assert input_file, "--input-file required when input is 'file'"
        input_kwargs["file_path"] = input_file

    if output_type == "file":
        assert output_file, "--output-file required when output is 'file'"
        output_kwargs["output_path"] = output_file

    input_stream = create_input_stream(input_type, **input_kwargs)
    output_stream = create_output_stream(output_type, **output_kwargs)

    print(f"Mirroring: {input_type} -> {output_type} @ {sample_rate}Hz")
    print("Press Ctrl+C to stop")

    input_stream.start()

    try:
        while True:
            chunk = input_stream.get_unprocessed_chunk()
            if chunk is not None and len(chunk) > 0:
                # # Ensure float32
                # if chunk.dtype != np.float32:
                #     chunk = chunk.astype(np.float32)
                output_stream.play_chunk(chunk)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        input_stream.stop()
        output_stream.wait()
        output_stream.stop()
        print("Done")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mirror audio from input to output")
    parser.add_argument(
        "--input",
        type=str,
        default="microphone",
        choices=["microphone", "file"],
        help="Input stream type (default: microphone)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="speaker",
        choices=["speaker", "file"],
        help="Output stream type (default: speaker)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file path (required if --input is 'file')",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (required if --output is 'file')",
    )

    args = parser.parse_args()

    run_mirror(
        input_type=args.input,
        output_type=args.output,
        sample_rate=args.sample_rate,
        input_file=args.input_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
