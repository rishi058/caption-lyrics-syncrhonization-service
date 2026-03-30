"""
If too many words are low-confidence, the chunk is re-processed with more aggressive time-stretching.
i.e REPEAT STEP 9,10,11
"""
import copy
import numpy as np
import pyrubberband as rubberband
from typing import Literal
from helpers.config import SAMPLE_RATE
from pipeline._10_transcription import transcribe_chunk
from pipeline._11_alignment import align_chunk
import time 

MAX_RETRIES = 2
CONFIDENCE_THRESHOLD = 0.6
RATIO_STEP = 0.10
MIN_RATIO = 0.50

def process_chunk_with_retry(segment_data: list[dict], language: Literal["en", "hi"], lyrics: str) -> list[dict]:
    """
    returns:
    [
        {
            "speaker": "SPEAKER_00", "start": 0.00, "end": 14.40, 
            "chunks": [
                {   "start": 0.00, "end": 10.00, "audio": np.ndarray, "style": "rap", "stretch_ratio": 0.7,
                    "words":[ {"word": ".......", "start": 0.00, "end": 5.00, "score": 0.99},
                              {"word": ".......", "start": 5.00, "end": 10.00, "score": 0.99}, ...
                            ]
                },
                ...
            ]
        },
        ...
    ]  
    """

    avg_confidence = _get_avg_confidence(segment_data)
    segment_data[0]["avg_confidence"] = avg_confidence

    if lyrics: 
        print(f"[LYRICS PRESENT]: Skipping retry mechanism. (Average confidence is {avg_confidence:.2f})")
        return segment_data


    if avg_confidence >= CONFIDENCE_THRESHOLD:
        print(f"[RETRY-MECHANISM] Average confidence is {avg_confidence:.2f} which is above threshold {CONFIDENCE_THRESHOLD:.2f}")
        return segment_data

    
    print(f"[Average confidence is {avg_confidence:.2f} which is below threshold {CONFIDENCE_THRESHOLD:.2f}]")

    # Store each attempt with its data and confidence
    conf_mapp = [{"segment_data": copy.deepcopy(segment_data), "avg_confidence": avg_confidence}]

    for retry in range(MAX_RETRIES + 1): 
        print(f"[RETRY {retry+1}] Initiated...")
        new_segment_data = copy.deepcopy(segment_data)

        for seg in new_segment_data:
            for chunk in seg["chunks"]:
                if "words" in chunk:
                    del chunk["words"]
        
        for seg in new_segment_data:
            for chunk in seg["chunks"]:
                stretch_ratio = chunk["stretch_ratio"] 
                style = chunk.get("style", "")

                if style == "opera":
                    new_stretch_ratio = stretch_ratio + (RATIO_STEP * (retry + 1))
                else:
                    new_stretch_ratio = stretch_ratio - (RATIO_STEP * (retry + 1))
                    new_stretch_ratio = max(new_stretch_ratio, MIN_RATIO)
                
                chunk["stretch_ratio"] = new_stretch_ratio
                chunk["audio"] = _time_stretch_chunk(chunk["audio"], new_stretch_ratio)


        new_segment_data = transcribe_chunk(new_segment_data, language, "")
        new_segment_data = align_chunk(new_segment_data, language) 
                
        conf_mapp.append({"segment_data": new_segment_data, "avg_confidence": _get_avg_confidence(new_segment_data)})     

    # sort by confidence
    conf_mapp.sort(key=lambda x: x["avg_confidence"], reverse=True)
    print(f"[RETRY COMPLETE]: Best confidence: {conf_mapp[0]['avg_confidence']:.2f}")
    
    return conf_mapp[0]["segment_data"]
            

#!----------------PRIVATE HELPERS----------------

def _get_avg_confidence(segment_data: list[dict]) -> float:
    all_scores = []
    for seg in segment_data:
        for chunk in seg.get("chunks", []):
            for w in chunk.get("words", []):
                all_scores.append(w.get("score", 0.0))

    return float(np.mean(all_scores)) if all_scores else 0.0

def _time_stretch_chunk(audio: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if abs(stretch_ratio - 1.0) < 0.01:
        return audio

    # rubberband handles ratio as speed factor:
    # ratio < 1.0 = slower = more time for each phoneme
    stretched = rubberband.time_stretch(audio, SAMPLE_RATE, rate=stretch_ratio)
    return stretched