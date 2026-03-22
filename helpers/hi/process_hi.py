"""
process_hi.py
─────────────
Hindi/Hinglish lyrics processing pipeline.

Flow:
  1. Lyrics → transliterate + LLM refine → mixed_words [{lat, dev, lang}, ...]
  2. Filter Hindi words → align with Hindi model → hindi_aligned_words
  3. Use english_gap_filler to place English words in gaps between Hindi words
  4. Merge both lists sorted by timestamp
"""

import whisperx
from helpers.config import COMPUTE_TYPE, DEVICE, MODEL_NAME, ALIGN_MODEL_HI, SAMPLE_RATE
from helpers.hi.transliteration import is_devanagari
from helpers.utils import clean_for_alignment
from helpers.hi.process_helper import process_devanagari_script, process_latin_script
from helpers.hi.english_gap_filler import fill_english_gaps


def process_hindi_language(lyrics: str, devanagari_output: bool, audio) -> list[dict]:
    """Returns sync data for Hindi/Hinglish language."""

    audio_duration = len(audio) / SAMPLE_RATE

    if lyrics:
        has_devanagari = is_devanagari(lyrics)

        if has_devanagari:
            lines = [clean_for_alignment(line, "devanagari") for line in lyrics.splitlines() if line.strip()]
            mixed_words, word_mapp = process_devanagari_script(lines)
        else:  # Latin script (Hinglish)
            lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
            mixed_words, word_mapp = process_latin_script(lines)
    else:
        segments = _transcribe_with_whisperx(audio)
        if not segments:
            raise RuntimeError("Hindi transcription failed: no segments returned.")
        # Combine all segment texts into one string
        full_text = " ".join(seg["text"] for seg in segments)
        lines = [clean_for_alignment(line, "devanagari") for line in full_text.splitlines() if line.strip()]
        mixed_words, word_mapp = process_devanagari_script(lines)

    # mixed_words = [{"lat":__, "dev":__, "lang":__}, ...]

    # ── Step 1: Build Hindi-only text for alignment ──────────────────────────
    hindi_text = _build_hindi_text(mixed_words)

    if not hindi_text.strip():
        raise RuntimeError("No Hindi words found in lyrics after processing.")

    hindi_segments = [{"text": hindi_text, "start": 0, "end": audio_duration}]

    # ── Step 2: Align Hindi words ────────────────────────────────────────────
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code="hi", device=DEVICE, model_name=ALIGN_MODEL_HI
        )
        result_aligned = whisperx.align(hindi_segments, model_a, metadata, audio, DEVICE)
        hindi_aligned_words = result_aligned["word_segments"]
    except Exception as e:
        raise RuntimeError(f"Hindi alignment failed: {e}") from e

    if not hindi_aligned_words:
        raise RuntimeError("Hindi alignment produced no word-level timestamps.")

    # ── Step 3: Fill English words into gaps ──────────────────────────────────
    en_aligned_words = fill_english_gaps(
        hindi_aligned_words=hindi_aligned_words,
        mixed_words=mixed_words,
        audio=audio,
        audio_duration=audio_duration,
    )

    # ── Step 4: Merge both and sort by timestamp ─────────────────────────────
    merged = _merge_and_tag(hindi_aligned_words, en_aligned_words, mixed_words, word_mapp)

    # ── Step 5: Output format ────────────────────────────────────────────────
    if devanagari_output:
        return merged
    else:
        # Convert Hindi words back to Latin/Hinglish
        for item in merged:
            if item.get("lang") == "hi" and item["word"] in word_mapp:
                item["word"] = word_mapp[item["word"]]
        return merged


# ──────────────────────────────────────────────────────────────────────────────
# Private Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_hindi_text(mixed_words: list[dict]) -> str:
    """Extracts only Hindi (Devanagari) words joined by spaces."""
    parts = [w["dev"] for w in mixed_words if w["lang"] == "hi"]
    return " ".join(parts)


def _merge_and_tag(
    hindi_words: list[dict],
    english_words: list[dict],
    mixed_words: list[dict],
    word_mapp: dict,
) -> list[dict]:
    """
    Merge Hindi and English aligned words into a single sorted list.
    Each entry has: word, start, end, score, lang.
    
    - 'word' uses the Devanagari form for Hindi words (for devanagari_output=True).
    - Output uses 'text' key for downstream compatibility with format_sync_data.
    """
    result = []

    for w in hindi_words:
        result.append({
            "text":  w.get("word", ""),
            "word":  w.get("word", ""),
            "start": w.get("start", 0.0),
            "end":   w.get("end", 0.0),
            "score": w.get("score", 0.0),
            "lang":  "hi",
        })

    for w in english_words:
        result.append({
            "text":  w.get("word", ""),
            "word":  w.get("word", ""),
            "start": w.get("start", 0.0),
            "end":   w.get("end", 0.0),
            "score": w.get("score", 0.0),
            "lang":  "en",
        })

    # Sort by start time; stable sort preserves order for equal timestamps
    result.sort(key=lambda x: x["start"])
    return result


def _transcribe_with_whisperx(audio) -> list[dict]:
    """Transcribes audio using WhisperX in Hindi mode."""
    try:
        model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
        result = model.transcribe(audio, batch_size=16, chunk_size=10, language="hi")
        return result.get("segments", [])
    except Exception as e:
        raise RuntimeError(f"Hindi transcription failed: {e}") from e