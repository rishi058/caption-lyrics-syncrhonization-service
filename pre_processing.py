import soundfile as sf
from pipeline._1_ingestion import ingest
from pipeline._2_seperation import separate_vocals
from pipeline._3_diarization import diarize
from pipeline._4_de_reverberation import dereverberate
from pipeline._5_lufs_normalization import lufs_normalize
from pipeline._6_pitch_range_normalization import pitch_normalize
from pipeline._7_vad_chunking import vad_chunking
from pipeline._8_style_classification import classify_style
from pipeline._9_time_stretching import time_stretching
import time, sys
from helpers.logger import CustomLogger

def pre_process_audio(media_path: str) -> tuple[list[dict], float]:
    formatted_media_path = ingest(media_path)   # removes video, force 44Khz 
    print(f"[{time.strftime('%X')}] Ingestion complete")

    vocal_media_path = separate_vocals(formatted_media_path)
    print(f"[{time.strftime('%X')}] Vocal separation complete")

    diarize_info = diarize(vocal_media_path)  # add [{"speaker", "start", "end"},...]
    print(f"[{time.strftime('%X')}] Diarization complete")

    segment_data = dereverberate(vocal_media_path, diarize_info)  # add [{"audio": np.ndarray},...]
    print(f"[{time.strftime('%X')}] Dereverberation complete")

    segment_data = lufs_normalize(segment_data) 
    print(f"[{time.strftime('%X')}] LUFS normalization complete")

    segment_data = pitch_normalize(segment_data)
    print(f"[{time.strftime('%X')}] Pitch normalization complete")

    segment_data = vad_chunking(segment_data)  # add "chunks": [{"start": 0.00, "end": 10.00, "audio": np.ndarray},...] delete "audio" key
    print(f"[{time.strftime('%X')}] VAD chunking complete")

    segment_data = classify_style(segment_data)  # add {"style": "", "syllable_rate"} for each chunk 
    print(f"[{time.strftime('%X')}] Style classification complete")

    segment_data = time_stretching(segment_data)  # add "stretch_ratio", delete "syllable_rate" for each chunk 
    print(f"[{time.strftime('%X')}] Time stretching complete")

    data, sr = sf.read(vocal_media_path)
    duration = len(data) / sr
 
    return segment_data, duration