"""
models.py
─────────
Pydantic request/response models for the FastAPI endpoints.
"""

from typing import Literal
from pydantic import BaseModel


class SyncLyricsRequest(BaseModel):
    media_path: str
    output_path: str
    language: Literal["en", "hi"]
    lyrics: str = ""
    force_alignment: bool = False
    devanagari_output: bool = True   # Hindi only. False → ITRANS/Hinglish output.
    isolate_vocals: bool = True
