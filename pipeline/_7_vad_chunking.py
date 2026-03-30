import torch
import numpy as np
import logging
from helpers.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

MAX_CHUNK_SECONDS = 28
MIN_CHUNK_SECONDS = 1        # discard trivially short chunks

# ── Lazy-loaded Silero VAD ──────────────────────────────────────────────────
_vad_model = None
_get_speech_timestamps = None

def _load_vad():
    global _vad_model, _get_speech_timestamps
    if _vad_model is None:
        logger.info("Loading Silero VAD model …")
        _vad_model, utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', trust_repo=True
        )
        _get_speech_timestamps = utils[0]
    return _vad_model, _get_speech_timestamps


# ── Public API ──────────────────────────────────────────────────────────────
def vad_chunking(segment_data: list[dict]) -> list[dict]:
    """
    returns:
    [
        {
            "speaker": "SPEAKER_00", "start": 0.00, "end": 14.40,
            "chunks": [
                {"start": 0.00, "end": 10.00, "audio": np.ndarray},
                {"start": 10.00, "end": 14.40, "audio": np.ndarray},
            ]
        },
        ...
    ]
    """
    for segment in segment_data:
        audio = segment["audio"]
        offset = segment.get("start", 0.0)
        chunks = _vad_chunk_helper(audio, offset)
        segment["chunks"] = chunks
        del segment["audio"]

    return segment_data


# ── Core chunker ────────────────────────────────────────────────────────────
def _vad_chunk_helper(audio: np.ndarray, offset: float = 0.0) -> list[dict]:
    """
    Run Silero VAD on *audio* and return a list of chunk dicts.
    All timestamps are absolute (offset is added).
    """
    model, get_ts = _load_vad()
    audio_tensor = torch.from_numpy(audio).float()

    speech_timestamps = get_ts(
        audio_tensor, model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=300,   # merge pauses < 300 ms
        min_speech_duration_ms=250,    # ignore clicks < 250 ms
        threshold=0.35
    )

    if not speech_timestamps:
        logger.warning("VAD found no speech in segment at offset %.2f s", offset)
        return []

    chunks: list[dict] = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / SAMPLE_RATE + offset
        end_sec   = ts['end']   / SAMPLE_RATE + offset
        duration  = end_sec - start_sec

        if duration < MIN_CHUNK_SECONDS:
            logger.debug("Skipping tiny VAD segment (%.3f s) at %.2f", duration, start_sec)
            continue

        if duration > MAX_CHUNK_SECONDS:
            chunk_audio = audio[ts['start']:ts['end']]
            chunks.extend(_hard_split(chunk_audio, SAMPLE_RATE, start_sec, end_sec, MAX_CHUNK_SECONDS))
        else:
            chunk_audio = audio[ts['start']:ts['end']]
            chunks.append({
                "start": start_sec,
                "end":   end_sec,
                "audio": chunk_audio,
            })

    return chunks


# ── Hard split at zero-crossings ────────────────────────────────────────────
def _hard_split(segment: np.ndarray, sr: int, abs_start: float,
                abs_end: float, max_len: float) -> list[dict]:
    """
    Split a 1-D audio segment at zero-crossings closest to *max_len*
    boundaries.  Falls back to equal-length splits when no crossings exist.

    Parameters
    ----------
    segment   : 1-D numpy array (already sliced from the parent audio)
    sr        : sample rate
    abs_start : absolute start time (seconds) of this segment
    abs_end   : absolute end time (seconds) of this segment
    max_len   : maximum chunk length in seconds
    """
    total_samples = segment.shape[0]

    # Find zero-crossings
    sign_changes = np.diff(np.sign(segment))
    crossings = np.where(sign_changes != 0)[0]

    if len(crossings) == 0:
        return _split_equal(segment, sr, max_len, abs_start)

    chunks: list[dict] = []
    current_pos = 0

    while current_pos < total_samples:
        target = current_pos + int(max_len * sr)

        # If remaining audio fits in one chunk, stop
        if target >= total_samples:
            break

        # Only consider crossings strictly after current_pos
        future = crossings[crossings > current_pos]
        if len(future) == 0:
            break

        # Pick the crossing closest to the ideal split point
        split_idx = int(future[np.argmin(np.abs(future - target))])

        # Safety: ensure we always advance
        if split_idx <= current_pos:
            split_idx = min(current_pos + int(max_len * sr), total_samples)

        chunk_audio = segment[current_pos:split_idx]
        chunks.append({
            "start": abs_start + current_pos / sr,
            "end":   abs_start + split_idx / sr,
            "audio": chunk_audio,
        })
        current_pos = split_idx

    # Remaining tail
    if current_pos < total_samples:
        chunks.append({
            "start": abs_start + current_pos / sr,
            "end":   abs_end,
            "audio": segment[current_pos:],
        })

    return chunks


# ── Equal-length fallback ───────────────────────────────────────────────────
def _split_equal(segment: np.ndarray, sr: int, max_len: float,
                 offset_sec: float = 0.0) -> list[dict]:
    """
    Divide *segment* into roughly equal parts, each ≤ *max_len* seconds.
    """
    total_samples = segment.shape[0]
    samples_per_chunk = int(max_len * sr)
    num_chunks = int(np.ceil(total_samples / samples_per_chunk))

    chunks: list[dict] = []
    for i in range(num_chunks):
        s = i * samples_per_chunk
        e = min(s + samples_per_chunk, total_samples)
        chunks.append({
            "start": offset_sec + s / sr,
            "end":   offset_sec + e / sr,
            "audio": segment[s:e],
        })

    return chunks