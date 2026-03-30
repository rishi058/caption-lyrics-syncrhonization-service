import subprocess
import os, sys
import shutil

def separate_vocals(input_path: str) -> str:
    """
    returns output_path
    """
    # 1. Path Setup
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(input_path))[0]
    cache_dir = os.path.join(root_dir, "cache", "seperations")
    os.makedirs(cache_dir, exist_ok=True)
    
    final_path = os.path.join(cache_dir, f"{song_name}_vocals.wav")

    # 2. Cache Check
    if os.path.exists(final_path):
        return final_path

    # 3. Step 1: Run Demucs (Intermediate output)
    # Demucs creates a folder structure: [output_dir]/[model]/[song_name]/vocals.wav
    model_name = "htdemucs_ft"
    subprocess.run([
        sys.executable, "-m", "demucs",
        "--name", model_name,
        "--two-stems", "vocals",
        "-o", cache_dir,
        input_path
    ], check=True)

    vocal_native = os.path.join(cache_dir, model_name, song_name, "vocals.wav")

    # 4. Step 2: Resample to 16kHz Mono
    subprocess.run([
        "ffmpeg",
        "-i", vocal_native,     # input
        "-ac", "1",             # mono
        "-ar", "16000",         # sr = 16k Hz
        "-sample_fmt", "s16",   # 16-bit signed integer
        final_path, "-y"        # output
    ], check=True, capture_output=True)

    # 5. Step 3: Cleanup Intermediate Files
    # Removes the 'htdemucs_ft' folder and its contents
    shutil.rmtree(os.path.join(cache_dir, model_name))
    
    return final_path