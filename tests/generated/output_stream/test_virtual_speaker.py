import time

import numpy as np

from conversa.generated.output_stream.virtual import VirtualSpeaker


def test_virtual_speaker_basic():
    """Test basic functionality of VirtualSpeaker."""
    speaker = VirtualSpeaker()

    chunk = np.zeros(1600, dtype=np.int16)  # 0.1s
    start = time.time()
    speaker.play_chunk(chunk)

    speaker.wait()
    elapsed = time.time() - start

    # It should wait at least 0.1s
    assert elapsed >= 0.1

    retrieved = speaker.get_unprocessed_chunk()
    assert retrieved is not None
    assert len(retrieved) == 1600
