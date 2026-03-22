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
import time
from helpers.config import COMPUTE_TYPE, DEVICE, MODEL_NAME, ALIGN_MODEL_HI, SAMPLE_RATE
from helpers.hi.transliteration import is_devanagari
from helpers.utils import clean_for_alignment
from helpers.hi.process_helper import process_devanagari_script, process_latin_script
from helpers.hi.english_gap_filler import fill_english_gaps
from helpers.silero_vad import _detect_vocal_bounds

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
        full_text = " ".join(seg["text"] for seg in segments)
        lines = [clean_for_alignment(line, "devanagari") for line in full_text.splitlines() if line.strip()]
        mixed_words, word_mapp = process_devanagari_script(lines)

    # mixed_words = [{"lat":__, "dev":__, "lang":__}, ...]

    # ── Step 1: Build Hindi-only text for alignment ──────────────────────────
    hindi_text = _build_hindi_text(mixed_words)

    if not hindi_text.strip():
        raise RuntimeError("No Hindi words found in lyrics after processing.")

    # ── Step 2: Detect vocal bounds in a SINGLE VAD pass ────────────────────
    vocal_start, vocal_end = _detect_vocal_bounds(audio, audio_duration)

    hindi_segments = [{"text": hindi_text, "start": vocal_start, "end": vocal_end}]

    # ── Step 3: Align Hindi words ────────────────────────────────────────────
    try:
        print(f"[{time.strftime('%X')}] Aligning Hindi words...")
        model_a, metadata = whisperx.load_align_model(
            language_code="hi", device=DEVICE, model_name=ALIGN_MODEL_HI
        )
        result_aligned = whisperx.align(hindi_segments, model_a, metadata, audio, DEVICE)
        hindi_aligned_words = result_aligned["word_segments"]
    except Exception as e:
        raise RuntimeError(f"Hindi alignment failed: {e}") from e

    if not hindi_aligned_words:
        raise RuntimeError("Hindi alignment produced no word-level timestamps.")

    # ── Step 4: Fill English words into gaps ─────────────────────────────────
    en_aligned_words = fill_english_gaps(
        hindi_aligned_words=hindi_aligned_words,
        mixed_words=mixed_words,
        audio=audio,
        audio_duration=audio_duration,
        vocal_start=vocal_start,
        vocal_end=vocal_end,
    )

    # ── Step 5: Merge both and preserve original word order ───────────────────
    merged = _merge_and_tag(hindi_aligned_words, en_aligned_words, mixed_words)

    # ── Step 6: Output format ─────────────────────────────────────────────────
    if devanagari_output:
        return merged
    else:
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
) -> list[dict]:
    """
    Merge Hindi and English aligned words into a single list that preserves
    the original word order from *mixed_words*.

    Instead of sorting by timestamp (which breaks when alignment produces
    identical or zero times), we walk *mixed_words* sequentially and consume
    the next aligned word from the appropriate queue (Hindi or English).

    Each output entry has keys: text, word, start, end, score, lang.
    """
    result: list[dict] = []

    hi_iter = iter(hindi_words)
    en_iter = iter(english_words)

    for m in mixed_words:
        if m["lang"] == "hi":
            w = next(hi_iter, None)
            if w is None:
                result.append({
                    "text":  m["dev"],
                    "word":  m["dev"],
                    "start": 0.0,
                    "end":   0.0,
                    "score": 0.0,
                    "lang":  "hi",
                })
                continue
            result.append({
                "text":  w.get("word", m["dev"]),
                "word":  w.get("word", m["dev"]),
                "start": w.get("start", 0.0),
                "end":   w.get("end", 0.0),
                "score": w.get("score", 0.0),
                "lang":  "hi",
            })
        else:  # English
            w = next(en_iter, None)
            if w is None:
                result.append({
                    "text":  m["lat"],
                    "word":  m["lat"],
                    "start": 0.0,
                    "end":   0.0,
                    "score": 0.0,
                    "lang":  "en",
                })
                continue
            result.append({
                "text":  w.get("word", m["lat"]),
                "word":  w.get("word", m["lat"]),
                "start": w.get("start", 0.0),
                "end":   w.get("end", 0.0),
                "score": w.get("score", 0.0),
                "lang":  "en",
            })

    return result

def _transcribe_with_whisperx(audio) -> list[dict]:
    """Transcribes audio using WhisperX in Hindi mode."""
    try:
        model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
        result = model.transcribe(audio, batch_size=16, chunk_size=10, language="hi")
        return result.get("segments", [])
    except Exception as e:
        raise RuntimeError(f"Hindi transcription failed: {e}") from e