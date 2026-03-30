
"""
This ensures the acoustic model sees consistent energy levels regardless of whether the singer is
belting, whispering, or trailing off.

Do not use simple peak normalization (scaling by max amplitude). A whispered phrase and a belted chorus can have
the same peak but very different perceived loudness. LUFS is perceptually weighted.
"""

import pyloudnorm as pyln
import numpy as np

TARGET_LUFS = -14.0
SAMPLE_RATE = 16000

def lufs_normalize(segment_data: list[dict]) -> list[dict]:
    """
    returns same format as input
    """
    meter = pyln.Meter(SAMPLE_RATE)  # creates BS.1770 meter

    # pyloudnorm block size is 0.4s; at 16kHz that's 6400 samples
    min_samples = int(SAMPLE_RATE * meter.block_size)

    for segment in segment_data:
        audio = segment["audio"]

        # Skip LUFS normalization for segments shorter than the block size
        if len(audio) < min_samples:
            continue

        loudness = meter.integrated_loudness(audio)
        # print(f"Loudness before: {loudness}")
        normalized = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)

        # Clip guard: prevent clipping after normalization
        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)

        segment["audio"] = normalized

    return segment_data