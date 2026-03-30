import parselmouth
import numpy as np
import pyrubberband as rubberband
from helpers.config import SAMPLE_RATE

TARGET_F0 = 170.0       # gender-agnostic speech midpoint
MAX_SHIFT_SEMI = 4.0    # hard cap: ±4 semitones
SAFE_F0_LOW = 90.0      # below this → shift up
SAFE_F0_HIGH = 400.0    # above this → shift down (soprano top ~350-400 Hz)

def pitch_normalize(segment_data: list[dict]) -> list[dict]:
    """
    returns same format as input
    """

    for segment in segment_data:
        audio = segment["audio"]
        shifted, shift = _pitch_normalizer_helper(audio)
        segment["audio"] = shifted

    return segment_data

def _pitch_normalizer_helper(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Auto pitch normalization. Only shifts extreme F0 (<90 Hz or >400 Hz)."""

    if audio.size == 0:
        return audio, 0.0

    # Praat needs ≥3 periods of the pitch floor for analysis.
    # Very short segments will fail with a PraatError. Skip them.
    MIN_SAMPLES = int(SAMPLE_RATE * 0.2)  # 0.2 seconds
    if audio.size < MIN_SAMPLES:
        return audio, 0.0

    # Ensure float64 for parselmouth and pyrubberband compatibility
    audio = audio.astype(np.float64)

    snd = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE)
    pitch = snd.to_pitch(time_step=0.01)
    f0_arr = pitch.selected_array['frequency']
    f0_voiced = f0_arr[f0_arr > 0]

    if len(f0_voiced) < 10:
        return audio, 0.0  # not enough voiced frames to measure

    f0_mean = float(np.mean(f0_voiced))

    # Safe range — no shift needed (covers 95%+ of songs)
    if SAFE_F0_LOW <= f0_mean <= SAFE_F0_HIGH:
        return audio, 0.0

    shift = 12 * np.log2(TARGET_F0 / f0_mean)
    shift = float(np.clip(shift, -MAX_SHIFT_SEMI, MAX_SHIFT_SEMI))

    if abs(shift) > 0.5:
        shifted = rubberband.pitch_shift(audio, SAMPLE_RATE, n_steps=shift)
    else:
        shifted = audio

    return shifted, shift