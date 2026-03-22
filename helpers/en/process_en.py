import whisperx
from helpers.config import COMPUTE_TYPE, DEVICE, MODEL_NAME, ALIGN_MODEL_EN, SAMPLE_RATE
from helpers.utils import clean_for_alignment


def process_english_language(lyrics: str, audio) -> list[dict]:
    """Returns sync data (list of aligned word dicts) for English language."""

    if not lyrics:
        # Transcribe using WhisperX model
        try:
            model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
            result = model.transcribe(audio, batch_size=16, chunk_size=10, language="en")
            segments = result["segments"]
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
    else:
        # Use provided lyrics as segments
        lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
        audio_duration = len(audio) / SAMPLE_RATE
        segments = [{"text": " ".join(lines), "start": 0.0, "end": audio_duration}]

    # Align the transcript with the audio
    try:
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
