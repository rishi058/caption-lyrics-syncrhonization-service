"""
Reverb smears the temporal edges of phonemes — the very start/stop points that WhisperX's CTC aligner depends on.
Removing reverb sharpens those edges.

NOTE: deepfilternet is installed separately, due to numpy version conflict with whisperx.
      This module calls the `deep-filter` CLI via subprocess.
"""
import subprocess
import os
import shutil
import numpy as np
import soundfile as sf

def dereverberate(audio_path: str, diarization_info: list[dict]) -> list[dict]:
    """
    De-reverb each speaker segment using DeepFilterNet CLI.

    returns:
    [
        {"speaker": "SPEAKER_00", "start": 0.00,  "end": 14.40, "audio": np.ndarray},
        {"speaker": "SPEAKER_01", "start": 14.40, "end": 28.80, "audio": np.ndarray},
        {"speaker": "SPEAKER_00", "start": 28.80, "end": 42.00, "audio": np.ndarray},
        ...
    ]
    """
    _check_deep_filter_available()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    raw_dir = os.path.join(os.path.join(root_dir, "cache", "dereverberations", song_name), "raw")
    enhanced_dir = os.path.join(os.path.join(root_dir, "cache", "dereverberations", song_name), "enhanced")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    # Load the full audio
    full_audio, sr = sf.read(audio_path, dtype="float32")
    # Ensure shape is (samples,) or (samples, channels)
    if full_audio.ndim == 1:
        full_audio = full_audio[:, np.newaxis]  # (samples, 1)

    results = []
    
    for i, segment in enumerate(diarization_info):
        out_path = os.path.join(enhanced_dir, f"seg_{i:04d}.wav")
        
        # Check cache
        if os.path.exists(out_path):
            enhanced, _ = sf.read(out_path, dtype="float32")
            if enhanced.ndim > 1:
                enhanced = enhanced[:, 0]  # take first channel

            segment["audio"] = enhanced
            results.append(segment)
            continue

        start_idx = int(segment["start"] * sr)
        end_idx = int(segment["end"] * sr)

        # Extract segment
        segment_audio = full_audio[start_idx:end_idx]

        duration = segment["end"] - segment["start"]

        # DeepFilterNet requires a minimum audio length (its internal block size).
        # Skip very short segments and pass them through unprocessed.
        MIN_DURATION = 0.2  # seconds
        if duration < MIN_DURATION:
            raw_mono = segment_audio[:, 0] if segment_audio.ndim > 1 else segment_audio
            segment["audio"] = raw_mono
            results.append(segment)
            continue

        # Write segment to raw cache directory
        seg_path = os.path.join(raw_dir, f"seg_{i:04d}.wav")
        sf.write(seg_path, segment_audio, sr)

        # Run deep-filter CLI
        out_path = _run_deep_filter(seg_path, enhanced_dir)

        # Read enhanced audio back
        enhanced, _ = sf.read(out_path, dtype="float32")
        if enhanced.ndim > 1:
            enhanced = enhanced[:, 0]  # take first channel

        segment["audio"] = enhanced
        results.append(segment)

    return results

#!-----------------------PRIVATE HELPERS----------------------------------

def _check_deep_filter_available():
    """Verify that the deep-filter CLI is accessible."""
    if shutil.which("deepFilter") is None:
        raise RuntimeError("deep-filter CLI not found. Install it separately:\n")


def _run_deep_filter(input_path: str, output_dir: str) -> str:
    """Run deep-filter CLI on a single audio file and return the output path."""
    result = subprocess.run(
        ["deepFilter", input_path, "--output-dir", output_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"deep-filter failed: {result.stderr}")

    # deep-filter appends _DeepFilterNet3 to the filename
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    df_output_name = f"{name}_DeepFilterNet3{ext}"
    df_output_path = os.path.join(output_dir, df_output_name)
     
    if not os.path.exists(df_output_path):
        raise FileNotFoundError(
            f"Expected deep-filter output at {df_output_path} but it was not found.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        
    final_output_path = os.path.join(output_dir, basename)
    os.replace(df_output_path, final_output_path)
    return final_output_path