import whisperx
from typing import Literal
import numpy as np
from helpers.config import DEVICE, COMPUTE_TYPE, WHISPERX_MODEL_NAME
from helpers.en.process_en import process_en_language
from helpers.hi.process_hi import process_hi_language

# ── Cached model ────────────────────────────────────────────────────────────
_cached_model = None
_cached_model_language = None

# chunk_size splits audio, batch_size groups chunks for parallel processing, beam_size controls decoding quality per chunk.

def transcribe_chunk(segment_data: list[dict], language: Literal["en", "hi"], lyrics: str) -> list[dict]:
    """
    returns:
    [
        {
            "speaker": "SPEAKER_00", "start": 0.00, "end": 14.40, 
            "chunks": [
                {"start": 0.00, "end": 10.00, "audio": np.ndarray, "style": "rap", "stretch_ratio": 0.7, "text": "......."},
                ...
            ]
        },
        ...
    ]
    """
    
    for segment in segment_data:
        for chunk in segment["chunks"]:
            text = _transcription_helper(chunk["audio"], language)
            chunk["text"] = text

    if language == "hi":
        segment_data = process_hi_language(segment_data, lyrics)
    elif language == "en":
        segment_data = process_en_language(segment_data, lyrics) 

    return segment_data

#!-------------PRIVATE HELPERS-----------------

def _get_model(language: Literal["en", "hi"]):
    global _cached_model, _cached_model_language
    if _cached_model is None or _cached_model_language != language:
        _cached_model = whisperx.load_model(
            WHISPERX_MODEL_NAME, DEVICE,
            compute_type=COMPUTE_TYPE,
            asr_options={"beam_size": 6},
            language=language
        )
        _cached_model_language = language
    return _cached_model

def _transcription_helper(audio: np.ndarray, language: Literal["en", "hi"]) -> str:
    try:
        model = _get_model(language)

        audio = audio.astype(np.float32)
        result = model.transcribe(
            audio,
            batch_size=6,         # increase for GPU
            chunk_size=30,
            language=language,
            task="transcribe",     # not translate
        )
        # result is { "text": str, "segments": list, "language": str }
        # When no speech is detected, WhisperX omits the top-level "text" key.
        text = result.get("text") or " ".join(
            seg.get("text", "") for seg in result.get("segments", [])
        )
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
