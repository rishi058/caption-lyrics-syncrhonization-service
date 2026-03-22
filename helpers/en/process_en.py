import whisperx
import time
from helpers.config import COMPUTE_TYPE, DEVICE, MODEL_NAME, ALIGN_MODEL_EN, SAMPLE_RATE
from helpers.utils import clean_for_alignment
from helpers.silero_vad import _detect_vocal_bounds


def process_english_language(lyrics: str, audio) -> list[dict]:
    """Returns sync data (list of aligned word dicts) for English language."""

    audio_duration = len(audio) / SAMPLE_RATE
    vocal_start, vocal_end = _detect_vocal_bounds(audio, audio_duration)

    if not lyrics:
        # Transcribe using WhisperX model
        print(f"[{time.strftime('%X')}] Transcribing English words...")
        try:
            model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
            result = model.transcribe(audio, batch_size=16, chunk_size=10, language="en")
            
            full_text = " ".join(seg["text"].strip() for seg in result.get("segments", []) if seg.get("text"))
            segments = [{"text": full_text, "start": vocal_start, "end": vocal_end}]
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
    else:
        # Use provided lyrics as segments
        lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
        segments = [{"text": " ".join(lines), "start": vocal_start, "end": vocal_end}]

    # Align the transcript with the audio
    try:
        print(f"[{time.strftime('%X')}] Aligning English words...")
        model_a, metadata = whisperx.load_align_model(
            language_code="en", device=DEVICE, model_name=ALIGN_MODEL_EN
        )
        result_aligned = whisperx.align(segments, model_a, metadata, audio, DEVICE)
        aligned_words = result_aligned["word_segments"]
    except Exception as e:
        raise RuntimeError(f"English alignment failed: {e}") from e

    if not aligned_words:
        raise RuntimeError("English alignment produced no words.")

    return aligned_words
