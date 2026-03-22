from helpers.hi.transliteration import hinglish_to_devanagari, devanagari_to_hinglish
from helpers.hi.llm.llm_service import LLMService


def process_devanagari_script(lines: list[str]) -> tuple[list[dict], dict]:
    """
    Takes Devanagari lyrics lines, converts to Hinglish, refines via LLM.
    Returns (refined_words, word_mapping) where word_mapping maps dev→lat.
    """
    lyrics_text = " ".join(lines)
    formatted_words = devanagari_to_hinglish(lyrics_text)

    refined_words = LLMService().refine_hinglish(formatted_words)

    # key = devanagari, value = latin
    word_mapp = {}
    for word in refined_words:
        word_mapp[word["dev"]] = word["lat"]

    return refined_words, word_mapp


def process_latin_script(lines: list[str]) -> tuple[list[dict], dict]:
    """
    Takes Hinglish/Latin lyrics lines, converts to Devanagari, refines via LLM.
    Returns (refined_words, word_mapping) where word_mapping maps dev→lat.
    """
    lyrics_text = " ".join(lines)
    formatted_words = hinglish_to_devanagari(lyrics_text)

    refined_words = LLMService().refine_hinglish(formatted_words)
    # refined_words = formatted_words  # bypassing LLM for debuging

    # key = devanagari, value = latin
    word_mapp = {}
    for word in refined_words:
        word_mapp[word["dev"]] = word["lat"]

    return refined_words, word_mapp