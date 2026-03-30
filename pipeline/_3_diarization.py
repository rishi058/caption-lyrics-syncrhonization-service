"""
This is critical because each singer needs different de-reverb and loudness parameters.
"""
from pyannote.audio import Pipeline
import torchaudio
import os 
import json
import torch
from helpers.config import HF_TOKEN
import soundfile as sf

def diarize(audio_path: str) -> list[dict]:
    """
    returns:
    [
        {"speaker": "SPEAKER_00", "start": 0.00,  "end": 14.40},
        {"speaker": "SPEAKER_01", "start": 14.40, "end": 28.80},
        {"speaker": "SPEAKER_00", "start": 28.80, "end": 42.00},
        ...
    ]
    """
    data, sr = sf.read(audio_path)  
    duration = len(data) / sr
    
    return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration}]
    # #-------------- CACHING --------------
    # root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # song_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # os.makedirs(os.path.join(root_dir, "cache", "diarizations"), exist_ok=True)
    # output_path = os.path.join(root_dir, "cache", "diarizations", f"{song_name}.json")

    # if os.path.exists(output_path):
    #     with open(output_path, "r") as f:
    #         return json.load(f) 


    # #-------------- LOGIC --------------
    # pipeline = Pipeline.from_pretrained(
    #     "pyannote/speaker-diarization-3.1",
    #      token=HF_TOKEN
    # )
    # pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # waveform, sample_rate = torchaudio.load(audio_path)
    # diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # annotation = getattr(diarization, "speaker_diarization", diarization)
    # segments = []
    # for turn, _, speaker in annotation.itertracks(yield_label=True):
    #     segments.append({
    #         "speaker": speaker,
    #         "start": turn.start,
    #         "end": turn.end,
    #     })


    # with open(output_path, "w") as f:
    #     json.dump(segments, f)

    # return segments

