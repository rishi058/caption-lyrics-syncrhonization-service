import whisperx
import torch
ALIGN_MODEL_HI = "theainerd/Wav2Vec2-large-xlsr-hindi"


print(f"CUDA: {torch.cuda.is_available()} count: {torch.cuda.device_count()}")

for device in ["cuda", "cuda:0", "cpu"]:
    try:
        print(f"Loading hi model on {device}...")
        model_a, metadata = whisperx.load_align_model(
            language_code="hi", device=device, model_name=ALIGN_MODEL_HI
        )
        print(f"Success on {device}!")
    except Exception as e:
        print(f"Error loading model on {device}: {e}")


# import traceback
# from helpers.hi.process_hi import process_hindi_language
# import soundfile as sf
# import whisperx
# print("Loading audio...")
# audio = whisperx.load_audio(r"C:\Users\Rishi\Downloads\Afusic_-_Not_Enough.mp4")

# with open(r"D:\STUDY 2\MediaEditor\01\test\lyrics1.txt", "r", encoding="utf-8") as f:
#     lyrics = f.read()

# try:
#     print("Running process_hindi_language...")
#     result = process_hindi_language(lyrics, devanagari_output=True, audio=audio)
#     print("Success! Length:", len(result))
# except Exception as e:
#     traceback.print_exc()

