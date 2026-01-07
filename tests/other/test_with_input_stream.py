"""Test the AudioParser with input stream using the start_stop.mp3 file.

This test is intended to be run from the repository root. Use:

    uv run tests/test_with_input_stream.py

It uses the existing `aiteacher/audio/start_stop.mp3` file and the
`AudioFileInputStream` implementation to feed chunks to `AudioParser`.
"""

import time
from pathlib import Path

from conversa.audio.audio_parser import AudioParser
from conversa.audio.input_stream import AudioFileInputStream


def test_audio_parser_with_input_stream():
    """Test AudioParser using AudioFileInputStream with start_stop.mp3."""
    print("Testing AudioParser with AudioFileInputStream")
    print("=" * 50)

    # Path to the test audio file (inside the repo)
    audio_file_path = (
        Path(__file__).parent.parent / "aiteacher" / "audio" / "start_stop.mp3"
    )

    if not audio_file_path.exists():
        print(f"Error: Audio file not found at {audio_file_path}")
        return

    print(f"Using audio file: {audio_file_path}")

    # Initialize the input stream
    input_stream = AudioFileInputStream(
        file_path=str(audio_file_path),
        sample_rate=16000,
        chunk_duration=0.1,  # 100ms chunks
    )

    # Initialize the audio parser
    try:
        parser = AudioParser(
            model_path="vosk-model-small-en-us-0.15", sample_rate=16000
        )
        print("AudioParser initialized successfully")
    except FileNotFoundError as e:
        print(f"Error initializing AudioParser: {e}")
        print("Make sure the Vosk model is downloaded and in the correct location")
        return

    # Start the input stream
    input_stream.start()
    print("Input stream started")

    speech_intervals = []
    chunk_count = 0

    try:
        print("\nProcessing audio chunks...")
        print("Looking for 'start' and 'stop' commands in the audio...")
        print("-" * 50)

        # We'll guard the loop so the test doesn't run forever in CI.
        max_empty_rounds = 40
        empty_rounds = 0

        while True:
            time.sleep(1)
            # Get audio chunk from input stream
            audio_chunk = input_stream.get_unprocessed_chunk()

            if audio_chunk is not None and len(audio_chunk) > 0:
                chunk_count += 1
                chunk_duration = len(audio_chunk) / input_stream.sample_rate

                print(
                    f"\nChunk {chunk_count}: {chunk_duration:.3f}s, "
                    f"buffer: {input_stream.get_buffer_duration():.2f}s"
                )

                # Process chunk through audio parser
                status, speech_audio = parser.add_chunk(audio_chunk)

                print(f"Parser status: {status}")

                if speech_audio is not None:
                    speech_duration = len(speech_audio) / parser.sample_rate
                    speech_intervals.append(speech_audio)
                    print(
                        f"*** SPEECH INTERVAL CAPTURED: {speech_duration:.2f} seconds ***"
                    )
                    print(f"Total intervals captured: {len(speech_intervals)}")

                # Show parser buffer status
                if parser.buffered_duration > 0:
                    print(f"Parser buffered: {parser.buffered_duration:.2f}s")

                empty_rounds = 0
            else:
                # No audio available right now
                empty_rounds += 1

                # If the stream finished reading file, break
                if not input_stream._is_running and (
                    audio_chunk is None or len(audio_chunk) == 0
                ):
                    print("\nInput stream finished")
                    break

                # If we've seen many empty polls, stop to avoid infinite loop
                if empty_rounds > max_empty_rounds:
                    print("\nNo new audio for a while, stopping test loop")
                    break

    except KeyboardInterrupt:
        print("\nStopped by user")

    except Exception as e:
        print(f"\nError during processing: {e}")

    finally:
        # Stop the input stream
        input_stream.stop()
        print("Input stream stopped")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total chunks processed: {chunk_count}")
    print(f"Speech intervals captured: {len(speech_intervals)}")

    if speech_intervals:
        for i, interval in enumerate(speech_intervals):
            duration = len(interval) / parser.sample_rate
            print(f"  Interval {i + 1}: {duration:.2f} seconds")
    else:
        print("No speech intervals were captured.")
        print("This could mean:")
        print("- No 'start' and 'stop' commands were detected in the audio")
        print("- The Vosk model didn't recognize the speech")
        print("- The audio file format is not compatible")

    print("\nTo improve recognition, you can:")
    print("- Ensure the audio contains clear 'start' and 'stop' words")
    print("- Check that the audio quality is good")
    print("- Verify the Vosk model is working correctly")


if __name__ == "__main__":
    test_audio_parser_with_input_stream()
