import threading
import time

import numpy as np
import pytest

from conversa.generated.output_stream.speaker import SpeakerOutputStream


@pytest.mark.slow
def test_hang():
    print("Initializing SpeakerOutputStream...")
    # Use dummy audio data
    sr = 16000
    duration = 1.0  # 1 second
    audio_data = np.zeros(int(sr * duration), dtype=np.float32)

    stream = SpeakerOutputStream(sample_rate=sr)

    print("1. Playing first chunk...")
    stream.play_chunk(audio_data)

    # Simulate user saying "start" -> stop() called
    time.sleep(0.5)  # Let it play a bit
    print("2. Stopping stream...")
    stream.stop()

    print("Stream stopped. Queue size:", stream._queue.qsize())

    # Simulate response
    print("3. Playing second chunk (response)...")
    audio_data2 = np.zeros(int(sr * 0.5), dtype=np.float32)
    stream.play_chunk(audio_data2)

    print("4. Waiting for playback to finish...")
    # This is where it reportedly hangs

    # Run wait in a thread with timeout to avoid blocking this script forever
    def wait_wrapper():
        stream.wait()
        print("Wait finished!")

    t = threading.Thread(target=wait_wrapper)
    t.start()
    t.join(timeout=5.0)
    error = t.is_alive()
    stream.stop()  # should be called in any case to clean up SpeakerOutputStream

    if error:
        print("FAIL: wait() hung for more than 5 seconds!")
        raise Exception("wait() hung")
    else:
        print("SUCCESS: wait() returned.")


if __name__ == "__main__":
    try:
        test_hang()
    except Exception as e:
        print(f"An error occurred: {e}")
