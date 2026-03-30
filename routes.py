import os
import time
from fastapi import APIRouter, HTTPException

from helpers.config import DEVICE, COMPUTE_TYPE,SUPPORTED_AUDIO_EXTS
from helpers.models import SyncLyricsRequest
from helpers.hi.transliteration import is_devanagari
from pre_processing import pre_process_audio
from pipeline._10_transcription import transcribe_chunk
from pipeline._11_alignment import align_chunk
from pipeline._12_retry_mechanism import process_chunk_with_retry
from pipeline._13_timestamp_remapping import remap_timestamps
from pipeline._14_format_and_save import format_and_save

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/sync-lyrics")
def sync_lyrics(request: SyncLyricsRequest):
    print(f"[{time.strftime('%X')}] Using device: {DEVICE.upper()} (Compute: {COMPUTE_TYPE})")

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.isfile(request.media_path):
        raise HTTPException(status_code=404, detail=f"Media file not found: {request.media_path}")

    _, ext = os.path.splitext(request.media_path)
    if ext.lower() not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use {', '.join(SUPPORTED_AUDIO_EXTS)}")

    has_devanagari_lyrics = is_devanagari(request.lyrics)
    if request.language == "en" and request.lyrics and has_devanagari_lyrics:
        raise HTTPException(status_code=400, detail="Language is set to English but lyrics has Devanagari script.")

    if request.language == "en" and request.devanagari_output:  
        raise HTTPException(status_code=400, detail="Language is set to English but devanagari_output is True.")

    # ── PHASE 1: PRE-PROCESSING ────────────────────────
    segment_data, duration = pre_process_audio(request.media_path)

    # ── PHASE 2: TRANSCRIPTION ────────────────────────
    segment_data = transcribe_chunk(segment_data, request.language, request.lyrics)  # adds "text" to each chunk
    print(f"[{time.strftime('%X')}] Transcription complete")

 
    # ── PHASE 3: ALIGNMENT ────────────────────────
    segment_data = align_chunk(segment_data, request.language)
    print(f"[{time.strftime('%X')}] Alignment complete")
    

    # ── PHASE 4: RETRY FOR IMPROVEMENT ────────────────────────
    segment_data = process_chunk_with_retry(segment_data, request.language, request.lyrics) 
    print(f"[{time.strftime('%X')}] Retry complete")


    # ── PHASE 5: TIME REMAPPING ────────────────────────
    segment_data = remap_timestamps(segment_data)
    print(f"[{time.strftime('%X')}] Time remapping complete")


    # ── PHASE 6: FINAL OUTPUT ────────────────────────
    format_and_save(segment_data, request.media_path, request.output_path, duration, request.language, request.devanagari_output)
    print(f"[{time.strftime('%X')}] Format and save complete")

    return {"message": "Synchronization complete"}