from helpers.utils import clean_for_alignment
from llm.llm_service import LLMService

def process_en_language(segment_data: list[dict], lyrics: str) -> list[dict]: 
    if lyrics:
        transcribed_segmented_lyrics = []
        for segment in segment_data:
            for chunk in segment["chunks"]:
                transcribed_segmented_lyrics.append(chunk["text"])

        cleaned_lyrics = " ".join(clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip())
 
        best_fit_lyrics = LLMService().refine_lyrics_segment(transcribed_segmented_lyrics, cleaned_lyrics, "en")
       
        for segment in segment_data:
            for chunk in segment["chunks"]:
                chunk["text"] = best_fit_lyrics.pop(0) if best_fit_lyrics else chunk["text"] 

    return segment_data 