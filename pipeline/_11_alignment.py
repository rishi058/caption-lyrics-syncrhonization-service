import whisperx
import numpy as np
from helpers.config import DEVICE, COMPUTE_TYPE, WHISPERX_MODEL_NAME, SAMPLE_RATE, ALIGN_MODEL_EN, ALIGN_MODEL_HI
from typing import Literal

# ── Cached model ────────────────────────────────────────────────────────────
_cached_model = None
_cached_model_language = None

def align_chunk(segment_data: list[dict], language: Literal["en", "hi"]) -> list[dict]:
    """
    returns:
    [
        {
            "speaker": "SPEAKER_00", "start": 0.00, "end": 14.40, 
            "chunks": [
                {   "start": 0.00, "end": 10.00, "audio": np.ndarray, "style": "rap", "stretch_ratio": 0.7, "text": ".......",
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

    for segment in segment_data:
        for chunk in segment["chunks"]:
            aligned = _align_chunk_helper(chunk["text"], chunk["audio"], language)
            chunk["words"] = aligned

    return segment_data

#!-------------PRIVATE HELPERS-----------------

def _get_model(language: Literal["en", "hi"]):
    global _cached_model, _cached_model_language
    if _cached_model is None or _cached_model_language != language:
        _cached_model = whisperx.load_model(
            WHISPERX_MODEL_NAME, DEVICE,
            compute_type=COMPUTE_TYPE,
            asr_options={"beam_size": 5},
            language=language
        )
        _cached_model_language = language
    return _cached_model

def _align_chunk_helper(transcription: str, audio: np.ndarray, language: Literal["en", "hi"]) -> list[dict]: 
    try: 
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=DEVICE,
            model_name=ALIGN_MODEL_EN if language == "en" else ALIGN_MODEL_HI
        )

        # Build the segments structure whisperx.align expects
        duration = len(audio) / SAMPLE_RATE
        segments = [{"text": transcription, "start": 0, "end": duration}]

        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio.astype(np.float32),
            DEVICE,
            return_char_alignments=False  # word-level is enough
        )

        # Extract word-level results from aligned output
        words = []
        for seg in aligned.get("segments", []):
            words.extend(seg.get("words", []))

        return words
    except Exception as e:
        raise RuntimeError(f"Alignment failed: {e}") from e