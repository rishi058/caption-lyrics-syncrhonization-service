"""
refine_lyrics_segment.py
────────────────────────
Corrects WhisperX-generated segment text against user-provided lyrics.
Fills missing words, replaces misheard words, preserves segment boundaries.
"""

import time
from pydantic import BaseModel, Field
from llm.base import BaseLLM
from helpers.logger import CustomLogger

# ── Prompt ───────────────────────────────────────────────────────────────────
REFINE_LYRICS_SEGMENTS_PROMPT = """\
You are a lyrical-segment corrector for English and Hindi songs.

INPUT:
• segmented_lyrics — list of text segments (WhisperX output, may have missing/wrong words).
• lyrics — the correct reference lyrics for this portion.

TASK:
For each segment, compare against the reference lyrics and:
1. Fill in words the transcriber missed.
2. Replace misheard words (similar pronunciation / singer accent).
3. Do NOT force-fit leftover lyrics into the last segment.

RULES:
- Preserve the EXACT number & order of segments.
- Do NOT merge, split, add, or remove segments.
- If a segment already looks correct, keep it unchanged.\
"""


# ── Structured output ───────────────────────────────────────────────────────
class RefineLyricsSegmentResponse(BaseModel):
    refined_lyrics: list[str] = Field(description="Corrected segment texts (same count & order)")


# ── Mixin ────────────────────────────────────────────────────────────────────
class RefineLyricsSegment(BaseLLM):
    def __init__(self):
        super().__init__()

    def refine_lyrics_segment(
        self,
        segmented_lyrics: list[str],
        lyrics: str,
        language: str,
    ) -> list[str]:
        CustomLogger.log(f"--- [REFINE-SEGMENT] LLM INPUT ---\n{segmented_lyrics}")

        seg_chunks = _chunk_segments(segmented_lyrics)
        lyr_chunks = _align_lyrics_to_chunks(seg_chunks, lyrics)
        all_results: list[str] = []

        ts = time.strftime('%X')
        print(f"[{ts}] [REFINE-SEGMENT] Total Chunks: {len(seg_chunks)}")

        for idx, chunk in enumerate(seg_chunks, 1):
            ts = time.strftime('%X') 

            user_input = {"segmented_lyrics": chunk, "lyrics": lyr_chunks[idx - 1]}

            try:
                result = self.invoke(
                    REFINE_LYRICS_SEGMENTS_PROMPT,
                    RefineLyricsSegmentResponse,
                    user_input,
                )
                all_results.extend(result.refined_lyrics)
                print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] chunk {idx} ✔, input-seg: {len(chunk)}, output-seg: {len(result.refined_lyrics)}")
            except Exception as e:
                print(f"[{time.strftime('%X')}] ⚠️ [REFINE-SEGMENT] chunk {idx} failed: {e}")
                print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] falling back to originals")
                all_results.extend(chunk)

        print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] Completed — final segment count: {len(all_results)}")
        CustomLogger.log(f"--- [REFINE-SEGMENT] LLM OUTPUT ---\n{all_results}")
        return all_results


# ── Helper: smart chunking ──────────────────────────────────────────────────

def _chunk_segments(
    sentences: list[str],
    target_words: int = 120,
    tolerance: int = 30,
) -> list[list[str]]:
    """Group sentences into chunks of roughly *target_words* (±tolerance)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    word_count = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and word_count >= target_words and word_count + n > target_words + tolerance:
            chunks.append(current)
            current = []
            word_count = 0
        current.append(sentence)
        word_count += n

    if current:
        chunks.append(current)
    return chunks


def _align_lyrics_to_chunks(
    seg_chunks: list[list[str]],
    lyrics: str,
    buffer: int = 7,
) -> list[str]:
    """Slice *lyrics* proportionally to match each segment chunk (+ small buffer)."""
    words = lyrics.split()
    chunk_sizes = [sum(len(s.split()) for s in c) for c in seg_chunks]
    result: list[str] = []
    pos = 0

    for size in chunk_sizes:
        end = min(pos + size + buffer, len(words))
        result.append(" ".join(words[pos:end]))
        pos = end

    # Attach leftover words to the last chunk
    if pos < len(words) and result:
        result[-1] += " " + " ".join(words[pos:])

    return result