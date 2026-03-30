import re
from typing import Literal

#---------------------------------------------------------------------------------------------------

# {'key': {'lat':___, 'lang':___}, }
global_word_mapp = {}

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

def format_segment_for_hindi(segment_data, devanagari_output) -> list[dict]:
    # currently segment_data has only dev script

    if devanagari_output:
        # only replace dev words which are actually english
        for seg in segment_data:
            for chunk in seg["chunks"]:
                for i in range(len(chunk["words"])): 
                    word = chunk["words"][i]    # {'word': '...', 'start': 0.00, 'end': 5.00, 'score': 0.99}
                    if word["word"] in global_word_mapp and global_word_mapp[word["word"]]["lang"] == "en":
                        word["word"] = global_word_mapp[word["word"]]["lat"] 

    else:
        # replace all with lat words
        for seg in segment_data:
            for chunk in seg["chunks"]:
                for i in range(len(chunk["words"])): 
                    word = chunk["words"][i]    # {'word': '...', 'start': 0.00, 'end': 5.00, 'score': 0.99}
                    if word["word"] in global_word_mapp:
                        word["word"] = global_word_mapp[word["word"]]["lat"] 

    return segment_data 