
def remap_timestamps(segment_data: list[dict]) -> list[dict]:
    """
    returns same format with updated timestamps
    """
    for segment in segment_data:
        for chunk in segment["chunks"]:
            chunk["words"] = _remap_timestamps_helper(chunk["words"], chunk["stretch_ratio"], chunk["start"]) 

    return segment_data

def _remap_timestamps_helper(words_list: list[dict], stretch_ratio: float, original_start_sec: float = 0) -> list[dict]:
    ratio = stretch_ratio
    offset = original_start_sec

    remapped = []
    for word in words_list:
        remapped.append({
                "text":       word.get("word", word.get("text", "")),
                "start":      word["start"] * ratio + offset,
                "end":        word["end"]   * ratio + offset,
                "confidence": round(word.get("score", 0.0), 6)
            })
    
    return remapped