from helpers.utils import clean_for_alignment
from llm.llm_service import LLMService
from helpers.hi.process_helper import process_devanagari_script, process_latin_script
from helpers.hi.transliteration import is_devanagari 

def process_hi_language(segment_data: list[dict], lyrics: str) -> list[dict]: 
    if lyrics:
        has_devanagari = is_devanagari(lyrics)

        if has_devanagari:
            lines = [clean_for_alignment(line, "devanagari") for line in lyrics.splitlines() if line.strip()]
            words_data = process_devanagari_script(lines)
        else:  # Latin script (Hinglish)
            lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
            words_data = process_latin_script(lines)
    else:
        transcribed_segmented_lyrics = []
        for segment in segment_data:
            for chunk in segment["chunks"]:
                transcribed_segmented_lyrics.append(chunk["text"])

        #! THIS PART OF CODE GET EXECUTED FOR RETRIES : MAY LEADING TO TOKEN LIMIT EXCEEDED
        transcribed_lyrics = " ".join(transcribed_segmented_lyrics)
        words_data = process_devanagari_script(transcribed_lyrics)

 
    #! IF Lyrics were provided by user: USE LLM FOR BEST FIT
    #! ELSE CORRECT WORD BY WORD FROM THE WORD DATA
    if lyrics:
        parts = [w["dev"] for w in words_data]
        devanagari_lyrics = " ".join(parts).strip() 
        best_fit_lyrics = LLMService().refine_lyrics_segment(transcribed_segmented_lyrics, devanagari_lyrics, "hi")
        
        for segment in segment_data:
            for chunk in segment["chunks"]:
                chunk["text"] = best_fit_lyrics.pop(0) if best_fit_lyrics else chunk["text"]
    else:
        idx = 0
        for segment in segment_data:
            for chunk in segment["chunks"]:
                words_in_chunk = chunk["text"].split()
                corrected_words = []
                for _ in words_in_chunk:
                    if idx < len(words_data):
                        corrected_words.append(words_data[idx]["dev"])
                        idx += 1
                chunk["text"] = " ".join(corrected_words)


    return segment_data

