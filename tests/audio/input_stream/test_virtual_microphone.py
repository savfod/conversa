import time

import numpy as np

from conversa.audio.input_stream.virtual import VirtualMicrophone
from conversa.generated.output_stream.virtual import VirtualSpeaker


def test_virtual_microphone_basic():
    """Test basic functionality of VirtualMicrophone."""
    mic = VirtualMicrophone()
    mic.start()

    chunk = np.zeros(1600, dtype=np.int16)
    mic.add_chunk(chunk)

    # Wait for processing
    time.sleep(0.2)

    read_chunk = mic.get_unprocessed_chunk()
    assert read_chunk is not None
    assert len(read_chunk) == 1600

    mic.stop()


def test_debug_echo_scenario():
    """
    Debug scenario: Echo input in 2 seconds with small chunks.
    This tests the interaction between VirtualMicrophone and VirtualSpeaker.
    """
    rate = 16000
    mic = VirtualMicrophone(sample_rate=rate)
    speaker = VirtualSpeaker(sample_rate=rate)

    mic.start()

    # 1. Feed audio into microphone
    print("Starting Echo Scenario...")
    start_time = time.time()

    t = np.linspace(0, 1.0, rate, endpoint=False)
    audio_source = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    chunk_size = 1600  # 100ms
    for i in range(0, len(audio_source), chunk_size):
        mic.add_chunk(audio_source[i : i + chunk_size])

    # Echo loop
    loop_duration = 2.0
    loop_start = time.time()

    chunks_processed = 0

    while time.time() - loop_start < loop_duration:
        chunk = mic.get_unprocessed_chunk()

        if chunk is not None:
            speaker.play_chunk(chunk)
            chunks_processed += 1

        time.sleep(0.01)

    mic.stop()
    speaker.wait()
    speaker.stop()

    # Verify
    received_data_len = 0
    while True:
        c = speaker.get_unprocessed_chunk()
        if c is None:
            break
        received_data_len += len(c)

    elapsed = time.time() - start_time

    assert received_data_len > 0
    assert elapsed >= 1.0
