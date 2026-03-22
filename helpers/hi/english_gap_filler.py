"""
english_gap_filler.py
─────────────────────
Assigns timestamps to English words that are skipped by the Hindi alignment
model (theainerd/Wav2Vec2-large-xlsr-hindi has no English phonemes in its
vocabulary).

HOW IT WORKS:
  1. Build a timeline of "known" timestamps from Hindi-aligned words.
  2. Identify contiguous gaps between Hindi words.
  3. Distribute English words across gaps proportionally by gap duration.
  4. For each gap, create a mini-segment and run the English alignment model.
     If English alignment also fails (rare), fall back to proportional split.

Example:
  Hindi aligned:  तेरा(0.2–0.6s)  ...gap(0.6–1.2s)...  है(1.2–1.5s)
  English word:   "lonely" → segment {"text": "lonely", "start": 0.6, "end": 1.2}
  English aligned: lonely(0.7–1.1s)  ← precise timestamp from gap
"""

import gc
import time

import torch
import whisperx

from helpers.config import DEVICE, ALIGN_MODEL_EN, SAMPLE_RATE


def fill_english_gaps(
    hindi_aligned_words: list[dict],
    mixed_words: list[dict],
    audio,
    audio_duration: float,
    device: str = DEVICE,
) -> list[dict]:
    """
    Assigns timestamps to English words using chronological gaps between aligned Hindi words.
    Returns a list of aligned English word dicts with 'word', 'start', 'end', 'score'.
    """
    en_words_count = sum(1 for m in mixed_words if m["lang"] == "en")
    if not en_words_count:
        return []

    print(f"[{time.strftime('%X')}] Gap-filling {en_words_count} English word(s) in their exact positions...")

    words_per_gap = _map_chronological_gaps(hindi_aligned_words, mixed_words, audio_duration)

    if not words_per_gap:
        print(f"[{time.strftime('%X')}] ⚠️  No gaps found for English words.")
        return []

    en_aligned_words = _align_words_in_gaps(words_per_gap, audio, device)

    print(f"[{time.strftime('%X')}] English gap-fill complete: {len(en_aligned_words)} word(s) timestamped.")
    return en_aligned_words

# ── Private Helpers ──────────────────────────────────────────────────────────

def _map_chronological_gaps(
    hindi_aligned_words: list[dict],
    mixed_words: list[dict],
    audio_duration: float,
) -> list[tuple[float, float, list[str]]]:
    """
    Scans the mixed sequence of words and creates [gap_start, gap_end, words] 
    tuples bounded exactly by the successfully aligned Hindi words.
    """
    hi_times = [None] * len(mixed_words)
    align_idx = 0
    
    # 1. Map successfully aligned Hindi words to their slots
    for i, m in enumerate(mixed_words):
        if m["lang"] == "hi":
            dev_word = m["dev"]
            for j in range(align_idx, len(hindi_aligned_words)):
                aligned_word = hindi_aligned_words[j].get("word", "")
                if aligned_word == dev_word:
                    if "start" in hindi_aligned_words[j] and "end" in hindi_aligned_words[j]:
                        hi_times[i] = (hindi_aligned_words[j]["start"], hindi_aligned_words[j]["end"])
                    align_idx = j + 1
                    break

    gaps_with_words = []
    current_en_words = []
    gap_start = 0.0

    # 2. Assign English word clusters to the physical gaps between the Hindi words
    for i, m in enumerate(mixed_words):
        if m["lang"] == "en":
            current_en_words.append(m["lat"])
        else:
            hi_time = hi_times[i]
            if hi_time:
                gap_end = hi_time[0]
                if current_en_words:
                    # Enforce monotonic boundaries for whisperx
                    gap_end = max(gap_start, gap_end)
                    gaps_with_words.append((gap_start, gap_end, current_en_words))
                    current_en_words = []
                
                gap_start = hi_time[1]

    # Trailing English words
    if current_en_words:
        gaps_with_words.append((gap_start, max(gap_start, audio_duration), current_en_words))

    return gaps_with_words


def _align_words_in_gaps(
    words_per_gap: list[tuple[float, float, list[str]]],
    audio,
    device: str,
) -> list[dict]:
    """
    Runs the English wav2vec2 alignment model on each gap segment.
    Falls back to proportional timestamps when alignment fails.
    """
    en_aligned_words: list[dict] = []

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code="en",
            device=device,
            model_name=ALIGN_MODEL_EN,
        )

        for gap_start, gap_end, words in words_per_gap:
            if not words:
                continue
            
            # Skip gaps that are too small to align
            if gap_end - gap_start < 0.05:
                en_aligned_words.extend(_proportional_fallback(words, gap_start, gap_end))
                continue

            segment_text = " ".join(words)
            gap_segment  = [{"text": segment_text, "start": gap_start, "end": gap_end}]

            try:
                gap_result = whisperx.align(gap_segment, model_a, metadata, audio, device)
                aligned_in_gap = gap_result.get("word_segments", [])
                
                if aligned_in_gap:
                    for word_info in aligned_in_gap:
                        if "start" in word_info and "end" in word_info:
                            en_aligned_words.append(word_info)
                        else:
                            en_aligned_words.extend(
                                _proportional_fallback([word_info.get("word", "")], gap_start, gap_end)
                            )
                else:
                    # Alignment returned nothing — fallback
                    en_aligned_words.extend(_proportional_fallback(words, gap_start, gap_end))
                    
            except Exception as e:
                print(f"[{time.strftime('%X')}] ⚠️  English alignment failed for gap "
                      f"({gap_start:.2f}–{gap_end:.2f}s): {e}")
                en_aligned_words.extend(_proportional_fallback(words, gap_start, gap_end))

        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[{time.strftime('%X')}] ⚠️  English alignment model failed to load: {e}")
        for gap_start, gap_end, words in words_per_gap:
            en_aligned_words.extend(_proportional_fallback(words, gap_start, gap_end))

    return en_aligned_words


def _proportional_fallback(words: list[str], start: float, end: float) -> list[dict]:
    """
    Last-resort: divide a time range equally among words.
    Used when alignment completely fails for a gap.
    """
    if not words:
        return []
    duration_per_word = (end - start) / len(words)
    return [
        {
            "word":  word,
            "start": round(start + i * duration_per_word, 4),
            "end":   round(start + (i + 1) * duration_per_word, 4),
            "score": 0.0,
        }
        for i, word in enumerate(words)
    ]
