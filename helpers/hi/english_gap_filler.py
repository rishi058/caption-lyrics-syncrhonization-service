"""
english_gap_filler.py
─────────────────────
Assigns timestamps to English words that are skipped by the Hindi alignment
model (theainerd/Wav2Vec2-large-xlsr-hindi has no English phonemes in its
vocabulary).

HOW IT WORKS:
  1. Build a timeline of "known" timestamps from Hindi-aligned words.
  2. Identify contiguous gaps between Hindi words.
  3. For each gap, create a mini-segment and run the English alignment model.
     If English alignment fails, fall back to proportional time distribution.
  4. Every returned word is guaranteed to have valid 'word', 'start', 'end',
     and 'score' keys — no missing timestamps.

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

from typing import List, Tuple, Optional, Dict, Any

# Minimum gap duration (seconds) worth sending to the alignment model.
_MIN_GAP_FOR_ALIGNMENT = 0.05


def fill_english_gaps(
    hindi_aligned_words: list[dict],
    mixed_words: list[dict],
    audio: Any,
    audio_duration: float,
    vocal_start: float = 0.0,
    vocal_end: float = None,
    device: str = DEVICE,
) -> list[dict]:
    """
    Assigns timestamps to English words using chronological gaps between
    aligned Hindi words.

    Returns
    -------
    list[dict]
        Each dict has keys: word, start, end, score.
        Words are returned in the same order they appear in *mixed_words*.
    """
    en_words_count = sum(1 for m in mixed_words if m["lang"] == "en")
    if not en_words_count:
        return []

    print(
        f"[{time.strftime('%X')}] Gap-filling {en_words_count} English "
        f"word(s) in their exact positions..."
    )

    words_per_gap = _map_chronological_gaps(
        hindi_aligned_words, mixed_words, vocal_end, vocal_start
    )

    if not words_per_gap:
        print(f"[{time.strftime('%X')}] ⚠️  No gaps found for English words.")
        return []

    en_aligned_words = _align_words_in_gaps(words_per_gap, audio, device)

    print(
        f"[{time.strftime('%X')}] English gap-fill complete: "
        f"{len(en_aligned_words)} word(s) timestamped."
    )
    return en_aligned_words


# ── Private Helpers ──────────────────────────────────────────────────────────


def _map_chronological_gaps(
    hindi_aligned_words: list[dict],
    mixed_words: list[dict],
    audio_duration: float,
    vocal_start: float = 0.0,
) -> list[tuple[float, float, list[str]]]:
    """
    Walk *mixed_words* in order and collect contiguous runs of English words,
    each bounded by the timestamps of the surrounding Hindi words (or by
    0 / audio_duration for leading/trailing runs).

    Returns a list of (gap_start, gap_end, [english_word_strings]).
    """
    # ── 1. Map each Hindi slot to its aligned (start, end) ───────────────
    hi_times: list[tuple[float, float] | None] = [None for _ in range(len(mixed_words))]
    align_idx = 0

    for i, m in enumerate(mixed_words):
        if m["lang"] != "hi":
            continue
        dev_word = m["dev"]
        for j in range(align_idx, len(hindi_aligned_words)):
            if hindi_aligned_words[j].get("word", "") == dev_word:
                hw = hindi_aligned_words[j]
                if "start" in hw and "end" in hw:
                    hi_times[i] = (hw["start"], hw["end"])
                align_idx = j + 1
                break

    # ── 2. Collect English clusters bounded by Hindi timestamps ──────────
    gaps: list[tuple[float, float, list[str]]] = []
    current_en_words: list[str] = []
    gap_start = vocal_start  # actual vocal onset, not always 0

    for i, m in enumerate(mixed_words):
        if m["lang"] == "en":
            current_en_words.append(m["lat"])
        else:
            hi_time = hi_times[i]
            if hi_time is not None:
                if current_en_words:
                    gap_end = max(gap_start, hi_time[0])
                    gaps.append((gap_start, gap_end, current_en_words))
                    current_en_words = []
                gap_start = hi_time[1]

    # Trailing English words after the last Hindi word
    if current_en_words:
        gap_end = max(gap_start, audio_duration)
        gaps.append((gap_start, gap_end, current_en_words))

    # ── 3. Fix degenerate leading gap ────────────────────────────────────
    if gaps and gaps[0][2] and gaps[0][1] - gaps[0][0] < _MIN_GAP_FOR_ALIGNMENT:
        _fix_leading_gap(gaps, hi_times, hindi_aligned_words, mixed_words, vocal_start)

    # ── 4. Fix degenerate trailing gap ───────────────────────────────────
    if gaps and gaps[-1][2] and gaps[-1][1] - gaps[-1][0] < _MIN_GAP_FOR_ALIGNMENT:
        _fix_trailing_gap(gaps, hindi_aligned_words, audio_duration)

    return gaps


def _fix_leading_gap(
    gaps: list[tuple[float, float, list[str]]],
    hi_times: list[tuple[float, float] | None],
    hindi_aligned_words: list[dict],
    mixed_words: list[dict],
    vocal_start: float = 0.0,
) -> None:
    """
    When leading English words have a zero-length gap (because the alignment
    model assigned start≈vocal_start to the first Hindi word), steal a proportional
    share of the first Hindi word's time range for the English words.

    Modifies *gaps* and *hindi_aligned_words* in place.
    """
    # Find the first Hindi word with valid timestamps
    first_hi_time = None
    first_hi_mixed_idx = None
    for i, t in enumerate(hi_times):
        if t is not None:
            first_hi_time = t
            first_hi_mixed_idx = i
            break

    if first_hi_time is None:
        return

    hi_duration = first_hi_time[1] - first_hi_time[0]
    if hi_duration <= 0:
        return

    en_count = len(gaps[0][2])
    total_words = en_count + 1  # English words + the first Hindi word

    # Give English words a proportional share of the first Hindi word's range
    new_boundary = first_hi_time[0] + hi_duration * (en_count / total_words)
    gaps[0] = (vocal_start, new_boundary, gaps[0][2])

    # Update the corresponding Hindi aligned word's start time
    dev_word = mixed_words[first_hi_mixed_idx]["dev"]
    for hw in hindi_aligned_words:
        if hw.get("word", "") == dev_word:
            hw["start"] = new_boundary
            break


def _fix_trailing_gap(
    gaps: list[tuple[float, float, list[str]]],
    hindi_aligned_words: list[dict],
    audio_duration: float,
) -> None:
    """
    When trailing English words have a zero-length gap (because the alignment
    model pushed the last Hindi words to the end of the audio), reclaim time
    by detecting compressed Hindi words and pushing them backward.

    The alignment model often stretches the last few Hindi words to fill up
    to *audio_duration*, even though the actual Hindi speech ends earlier and
    the English section occupies the tail of the audio.  This function detects
    that pattern (large timestamp gap followed by very short Hindi words) and
    reclaims that time for the English words.

    Modifies *gaps* and *hindi_aligned_words* in place.
    """
    en_count = len(gaps[-1][2])
    if not en_count:
        return

    # ── 1. Detect compressed trailing Hindi words ────────────────────────
    # Walk backward through hindi_aligned_words to find words that are
    # suspiciously short and clustered at the audio's tail.
    _COMPRESSED_THRESHOLD = 0.1  # words shorter than 100ms are suspicious

    compressed_start_idx = len(hindi_aligned_words)
    for i in range(len(hindi_aligned_words) - 1, -1, -1):
        hw = hindi_aligned_words[i]
        duration = hw.get("end", 0) - hw.get("start", 0)
        if duration < _COMPRESSED_THRESHOLD:
            compressed_start_idx = i
        else:
            break

    # ── 2. Check if there's a large gap before the compressed cluster ────
    if compressed_start_idx < len(hindi_aligned_words) and compressed_start_idx > 0:
        prev_hw = hindi_aligned_words[compressed_start_idx - 1]
        first_compressed = hindi_aligned_words[compressed_start_idx]
        gap_before_compressed = first_compressed.get("start", 0) - prev_hw.get("end", 0)

        if gap_before_compressed > 1.0:
            # There IS a large natural gap — the English words belong there.
            # Re-assign: put the compressed Hindi words right after the last
            # "real" Hindi word, freeing the gap for English words.
            real_end = prev_hw.get("end", 0)
            n_compressed = len(hindi_aligned_words) - compressed_start_idx

            # Give the compressed Hindi words a small slice right after real_end
            c_start = int(compressed_start_idx)
            compressed_duration = sum(
                hw.get("end", 0) - hw.get("start", 0)
                for hw in hindi_aligned_words[c_start:]
            )
            # At minimum give each compressed word 50ms
            compressed_duration = float(max(compressed_duration, float(n_compressed) * 0.05))

            cursor = real_end
            for hw in hindi_aligned_words[c_start:]:
                word_dur = float(max(float(hw.get("end", 0.0)) - float(hw.get("start", 0.0)), 0.05))  
                hw["start"] = float(cursor)
                hw["end"] = float(cursor + word_dur)
                cursor = hw["end"]

            # Now the English gap starts after the repositioned Hindi words
            en_gap_start = float(cursor)
            en_gap_end = float(audio_duration)
            if en_gap_end - en_gap_start >= _MIN_GAP_FOR_ALIGNMENT:
                gaps[-1] = (en_gap_start, en_gap_end, gaps[-1][2])
                return

    # ── 3. Fallback: steal proportional time from the last Hindi word ────
    # Find the last Hindi word with meaningful duration
    for i in range(len(hindi_aligned_words) - 1, -1, -1):
        hw = hindi_aligned_words[i]
        hi_start = float(hw.get("start", 0.0))
        hi_end = float(hw.get("end", 0.0))
        hi_dur = hi_end - hi_start
        if hi_dur > 0.1:
            total_words = en_count + 1
            en_share = hi_dur * (en_count / total_words)
            new_boundary = hi_end - en_share
            hw["end"] = new_boundary
            gaps[-1] = (new_boundary, float(audio_duration), gaps[-1][2])
            return


def _align_words_in_gaps(
    words_per_gap: list[tuple[float, float, list[str]]],
    audio,
    device: str,
) -> list[dict]:
    """
    For each gap, attempt precise English alignment via wav2vec2.
    Falls back gracefully to proportional distribution when alignment
    fails or returns incomplete data.

    Every word in the returned list is guaranteed to have:
        word  (str)
        start (float)
        end   (float)
        score (float)
    """
    all_aligned: list[dict] = []

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code="en",
            device=device,
            model_name=ALIGN_MODEL_EN,
        )
    except Exception as e:
        print(
            f"[{time.strftime('%X')}] ⚠️  English alignment model failed "
            f"to load: {e} — using proportional fallback for all gaps."
        )
        for gap_start, gap_end, words in words_per_gap:
            all_aligned.extend(_proportional_fallback(words, gap_start, gap_end))
        return all_aligned

    try:
        for gap_start, gap_end, words in words_per_gap:
            if not words:
                continue

            aligned_for_gap = _try_align_gap(
                words, gap_start, gap_end, model_a, metadata, audio, device,
            )
            all_aligned.extend(aligned_for_gap)
    finally:
        # Always release the model, even on unexpected errors
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return all_aligned


def _try_align_gap(
    words: list[str],
    gap_start: float,
    gap_end: float,
    model_a,
    metadata,
    audio,
    device: str,
) -> list[dict]:
    """
    Attempt alignment for a single gap.  Returns a list of word dicts
    with guaranteed start/end/score keys.
    """
    # Gap too narrow for meaningful alignment → proportional immediately
    if gap_end - gap_start < _MIN_GAP_FOR_ALIGNMENT:
        return _proportional_fallback(words, gap_start, gap_end)

    segment_text = " ".join(words)
    gap_segment = [{"text": segment_text, "start": gap_start, "end": gap_end}]

    try:
        gap_result = whisperx.align(
            gap_segment, model_a, metadata, audio, device,
        )
        aligned_in_gap = gap_result.get("word_segments", [])
    except Exception as e:
        print(
            f"[{time.strftime('%X')}] ⚠️  English alignment failed for gap "
            f"({gap_start:.2f}–{gap_end:.2f}s): {e}"
        )
        return _proportional_fallback(words, gap_start, gap_end)

    if not aligned_in_gap:
        return _proportional_fallback(words, gap_start, gap_end)

    # Validate every word that came back from the aligner.
    # Words missing start/end get proportional timestamps within this gap.
    result: list[dict] = []
    missing_words: list[str] = []  # accumulate words without timestamps

    for word_info in aligned_in_gap:
        has_times = "start" in word_info and "end" in word_info
        if has_times:
            # Flush any accumulated missing-timestamp words first
            if missing_words:
                fallback_end = word_info["start"]
                fallback_start = result[-1]["end"] if result else gap_start
                result.extend(
                    _proportional_fallback(
                        missing_words,
                        fallback_start,
                        max(fallback_start, fallback_end),
                    )
                )
                missing_words = []
            result.append(_normalise_word(word_info, gap_start, gap_end))
        else:
            missing_words.append(word_info.get("word", ""))

    # Any remaining missing words go after the last good timestamp
    if missing_words:
        fallback_start = result[-1]["end"] if result else gap_start
        result.extend(
            _proportional_fallback(missing_words, fallback_start, gap_end)
        )

    return result


def _normalise_word(
    word_info: dict,
    gap_start: float,
    gap_end: float,
) -> dict:
    """
    Return a clean word dict with exactly the four required keys.
    Clamps start/end within the gap boundaries to prevent timestamp leaks.
    """
    start = max(gap_start, float(word_info["start"]))
    end = min(gap_end, float(word_info["end"]))
    # Ensure end >= start even after clamping
    end = max(start, end)
    
    # Use standard rounding compatible with pyre type checks
    return {
        "word":  word_info.get("word", ""),
        "start": float(f"{start:.4f}"),
        "end":   float(f"{end:.4f}"),
        "score": float(f"{word_info.get('score', 0.0):.6f}"),
    }


def _proportional_fallback(
    words: list[str],
    start: float,
    end: float,
) -> list[dict]:
    """
    Distribute a time range equally among *words*.
    Used when the alignment model cannot produce precise timestamps.
    Every word receives a non-overlapping slice of [start, end].
    """
    if not words:
        return []

    # Guard against degenerate range (end <= start)
    if end <= start:
        st = float(f"{start:.4f}")
        return [
            {"word": w, "start": st,
             "end": st, "score": 0.0}
            for w in words
        ]

    step = (end - start) / len(words)
    return [
        {
            "word":  w,
            "start": float(f"{start + i * step:.4f}"),
            "end":   float(f"{start + (i + 1) * step:.4f}"),
            "score": 0.0,
        }
        for i, w in enumerate(words)
    ]

