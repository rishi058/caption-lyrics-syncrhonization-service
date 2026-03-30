import subprocess
import os
import shutil
import soundfile as sf

def ingest(input_path: str) -> str:
    """
    returns output_path
    """
    _check_ffmpeg()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(input_path))[0]
    
    os.makedirs(os.path.join(root_dir, "cache", "ingestions"), exist_ok=True)
    output_path = os.path.join(root_dir, "cache", "ingestions", f"{song_name}.wav")

    # 2. Caching System
    # If the file exists, read and return immediately
    if os.path.exists(output_path):
        return output_path

    # 3. Processing (FFmpeg)
    # The extension is .wav; FFmpeg automatically selects the PCM encoder for this container
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn",           # Strip video if present
        "-ar", "44100",  # Force 44.1kHz sample rate
        "-ac", "2",      # Force 2 channels (stereo)
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # 4. Post-processing & Validation
    data, sr = sf.read(output_path)
    duration = len(data) / sr

    assert 5 < duration < 600, f"File duration {duration:.1f}s is outside allowed limits."

    return output_path

#!-------------------------PRIVATE HELPERS------------------------------

def _check_ffmpeg():
    """Raise a helpful error if ffmpeg is not found on PATH."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "FFmpeg not found. Install it and ensure it's on your PATH.\n"
        )