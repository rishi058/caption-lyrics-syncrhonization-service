import librosa
import numpy as np
import parselmouth
from helpers.config import SAMPLE_RATE

def classify_style(segment_data: list[dict]) -> list[dict]:
    """
    returns:
    [
        {
            "speaker": "SPEAKER_00", "start": 0.00, "end": 14.40, 
            "chunks": [
                {"start": 0.00, "end": 10.00, "audio": np.ndarray, "style": "", "syllable_rate": 2},
                {"start": 10.00, "end": 14.40, "audio": np.ndarray, "style": "", "syllable_rate": 6},
            ]
        },
        ...
    ]
    """

    for segment in segment_data:
        for chunk in segment["chunks"]:
            info = _style_classification_helper(chunk["audio"])
            chunk["style"] = info["style"]
            chunk["syllable_rate"] = info["syllable_rate"]
            
    return segment_data


def _style_classification_helper(chunk_audio: np.ndarray) -> dict:
    # Guard against empty or near-empty audio
    if chunk_audio.size == 0:
        return {
            "style": "pop",
            "syllable_rate": 0.0,
            "f0_mean": 0.0,
            "f0_std": 0.0,
        }

    duration = len(chunk_audio) / SAMPLE_RATE

    if duration < 0.05:
        return {
            "style": "pop",
            "syllable_rate": 0.0,
            "f0_mean": 0.0,
            "f0_std": 0.0,
        }

    # Ensure float64 for parselmouth compatibility
    chunk_audio = chunk_audio.astype(np.float64)

    # Syllable rate via onset detection
    onsets = librosa.onset.onset_detect(y=chunk_audio, sr=SAMPLE_RATE, units='time')
    syllable_rate = len(onsets) / duration

    # F0 via Praat
    snd = parselmouth.Sound(chunk_audio, sampling_frequency=SAMPLE_RATE)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]

    if len(f0_values) > 0:
        f0_mean = float(np.mean(f0_values))
        f0_std  = float(np.std(f0_values))
    else:
        # No voiced frames detected — cannot reliably classify by pitch,
        # fall back to syllable-rate-only classification
        f0_mean = 0.0
        f0_std  = 0.0

    # Classification rules (syllable-rate driven, pitch as secondary signal)
    if syllable_rate > 6.0:
        style = "rap"
    elif syllable_rate > 4.5 and f0_std < 40:
        style = "fast_pop"   # fast but monotone → rap-like delivery
    elif f0_mean > 350 and f0_std > 80 and syllable_rate < 3.5:
        style = "opera"      # high tessitura + wide vibrato (soprano/tenor range)
    elif syllable_rate < 2.5 and f0_std < 30:
        style = "ballad"
    else:
        style = "pop"

    return {
        "style": style,
        "syllable_rate": round(syllable_rate, 2),
        "f0_mean": round(f0_mean, 1),
        "f0_std":  round(f0_std, 1),
    }