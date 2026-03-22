"""
routes.py
─────────
FastAPI router containing all HTTP endpoints.
Keeps routing logic decoupled from business logic.
"""

import os
import time

import whisperx
from fastapi import APIRouter

from helpers.config import DEVICE, COMPUTE_TYPE, SAMPLE_RATE, SUPPORTED_AUDIO_EXTS
from helpers.models import SyncLyricsRequest
from helpers.utils import get_isolated_vocals, format_sync_data, save_sync_data
from helpers.en.process_en import process_english_language
from helpers.hi.process_hi import process_hindi_language
from helpers.hi.transliteration import is_devanagari

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/sync-lyrics")
async def sync_lyrics(request: SyncLyricsRequest):
    print(f"[{time.strftime('%X')}] Using device: {DEVICE.upper()} (Compute: {COMPUTE_TYPE})")

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.isfile(request.media_path):
        return {"error": f"Media file not found: {request.media_path}"}

    _, ext = os.path.splitext(request.media_path)
    if ext.lower() not in SUPPORTED_AUDIO_EXTS:
        return {"error": f"Unsupported format. Use {', '.join(SUPPORTED_AUDIO_EXTS)}"}

    if request.force_alignment and not request.lyrics.strip():
        return {"error": "force_alignment is True but no lyrics were provided."}

    if request.language == "en" and request.lyrics and is_devanagari(request.lyrics):
        return {"error": "Language is set to English but lyrics has Devanagari script."}

    if request.language == "en" and request.devanagari_output:  
        return {"error": "Language is set to English but devanagari_output is True."}

    start_time = time.time()

    # ── Step 1: Load audio (optionally isolate vocals) ────────────────────────
    audio_source = (
        get_isolated_vocals(request.media_path)
        if request.isolate_vocals
        else request.media_path
    )
    audio          = whisperx.load_audio(audio_source)
    audio_duration = len(audio) / SAMPLE_RATE

    # ── Step 2: Process by language ───────────────────────────────────────────
    try:
        if request.language == "en":
            sync_data = process_english_language(request.lyrics, audio)
        else:
            sync_data = process_hindi_language(request.lyrics, request.devanagari_output, audio)
    except RuntimeError as e:
        return {"error": str(e)}

    # ── Step 3: Format and save output JSON ───────────────────────────────────
    sync_data = format_sync_data(sync_data, audio_duration)
    output_path = save_sync_data(sync_data, request.media_path, request.output_path)

    print(f"[{time.strftime('%X')}] Complete! Total time: {time.time() - start_time:.2f}s")
    print(f"[{time.strftime('%X')}] Output saved to: {output_path}")
    return {"message": "Synchronization complete", "output_path": output_path}
