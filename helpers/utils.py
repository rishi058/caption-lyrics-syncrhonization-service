import os
import re
import subprocess
import time
import sys 
import json
from typing import Literal

# ── Vocal Isolation ──────────────────────────────────────────────────────────
def get_isolated_vocals(media_path: str) -> str:
    """
    Runs Demucs to separate vocals from the mix.

    Returns the path to the isolated vocals WAV file.
    Falls back to the original media path if Demucs fails or is unavailable.
    Caches result — re-runs are skipped if the output file already exists.
    """
    print(f"[{time.strftime('%X')}] Isolating vocals with Demucs...")
    
    output_dir = os.path.abspath(os.getcwd())                             # output goes to ./htdemucs/ relative to current working directory
    stem       = os.path.splitext(os.path.basename(media_path))[0]        # extracts the filename without extension
    expected   = os.path.join(output_dir, "htdemucs", stem, "vocals.wav") # constructs the expected output path: ./htdemucs/{stem}/vocals.wav

    if os.path.isfile(expected):
        print(f"[{time.strftime('%X')}] Reusing cached vocals: {expected}")
        return expected

    result = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", output_dir, media_path],
        capture_output=True, text=True
    )

    if result.returncode != 0 or not os.path.isfile(expected):
        print(f"[{time.strftime('%X')}] ⚠️  Demucs failed — using original audio.")
        if result.stderr:
            print(f"    stderr: {result.stderr.strip()}")
        return media_path

    print(f"[{time.strftime('%X')}] Vocals isolated: {expected}")
    return expected

# ── Text Utilities ───────────────────────────────────────────────────────────
def clean_for_alignment(text: str, script: Literal["latin", "devanagari"]) -> str:
    """
    Strips characters that alignment models cannot handle:
    1. keeps only ASCII letters and whitespace
    2. keeps only Devanagari codepoints and whitespace
    Collapses multiple spaces and trims the result.
    """
    if script == "latin":
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    else:
        cleaned = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

# ── Other Utilities ───────────────────────────────────────────────────────────
def format_sync_data(sync_data: list[dict], audio_duration: float) -> list[dict]:
    """
    Add prefix space, convert start/end times to ms, and add confidence.
    Add empty entries for start/end.
    """
    if not sync_data:
        # Return minimal data if there's nothing to format
        return [
            {"text": " ", "startMs": 0, "endMs": int(audio_duration * 1000),
             "timestampMs": 0, "confidence": 1}
        ]

    fresh_data = []

    for item in sync_data:
        # Create a fresh dictionary to ensure only desired fields are included
        temp = {}
        temp["text"] = " " + item.get("text", item.get("word", ""))
        temp["startMs"] = int(item.get("start", 0) * 1000)  
        temp["endMs"]   = int(item.get("end", 0) * 1000) 
        temp["timestampMs"] = temp["startMs"] 
        temp["confidence"] = round(item.get("score", 0.0), 6)

        fresh_data.append(temp) 

    
    fresh_data.insert(0, {
        "text": " ", "startMs": 0,
        "endMs": max(0, fresh_data[0]["startMs"] - 1),
        "timestampMs": 0, "confidence": 1,
    })

    fresh_data.append({
        "text": " ", "startMs": fresh_data[-1]["endMs"] + 1,
        "endMs": int(audio_duration * 1000),
        "timestampMs": fresh_data[-1]["endMs"] + 1, "confidence": 1,
    })

    return fresh_data 

def save_sync_data(sync_data: list[dict], media_path: str, output_dir: str) -> str:
    """
    Writes the sync data JSON to disk.
    The output file is named after the media file stem, e.g.: output_dir/MySong.json
    Returns the output file path.
    """
    media_name  = os.path.splitext(os.path.basename(media_path))[0]
    output_path = os.path.join(output_dir, media_name + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sync_data, f, indent=4, ensure_ascii=False)

    return output_path
