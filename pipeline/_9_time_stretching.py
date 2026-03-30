import os
import numpy as np
from helpers.config import SAMPLE_RATE

import pyrubberband.pyrb as pyrb
# Monkey-patch the rubberband path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
pyrb.__RUBBERBAND_UTIL = os.path.join(root_dir, "rubberband", "rubberband.exe")
import pyrubberband as rubberband

def time_stretching(segment_data: list[dict]) -> list[dict]:
    """
    returns:
    [
        {
            "speaker": "SPEAKER_00", "start": 0.00, "end": 14.40, 
            "chunks": [
                {"start": 0.00, "end": 10.00, "audio": np.ndarray, "style": "rap", "stretch_ratio": 0.7},
                {"start": 10.00, "end": 14.40, "audio": np.ndarray, "style": "fast_pop", "stretch_ratio": 0.85},
            ]
        },
        ...
    ]  
    """

    for segment in segment_data:  
        for chunk in segment["chunks"]:
            stretch_ratio = _get_stretch_ratio(chunk["style"], chunk["syllable_rate"])

            chunk["audio"] = _time_stretch_chunk(chunk["audio"], stretch_ratio) 
            chunk["stretch_ratio"] = stretch_ratio

            del chunk["syllable_rate"]  
            
    return segment_data


#!-----------PRIVATE HELPERS--------------

def _time_stretch_chunk(audio: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if abs(stretch_ratio - 1.0) < 0.01:
        return audio

    # rubberband handles ratio as speed factor:
    # ratio < 1.0 = slower = more time for each phoneme
    stretched = rubberband.time_stretch(audio, SAMPLE_RATE, rate=stretch_ratio)
    return stretched



def _get_stretch_ratio(style: str, syllable_rate: float) -> float:
    if style == "rap":
        if syllable_rate > 8.0: return 0.70   # speed rap
        if syllable_rate > 6.0: return 0.75   # normal rap
    elif style == "fast_pop":
        return 0.85
    elif style == "pop":
        if syllable_rate > 3.5: return 0.90
    elif style == "opera":
        return 1.15

    return 1.0  # don't stretch — speed is not the issue  