import json
import re
import difflib

with open(r'd:\STUDY 2\MediaEditor\01\test\lyrics_json\Die_For_You_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

json_text = []
for segment in data:
    for chunk in segment.get('chunks', []):
        json_text.append(chunk.get('text', '').strip())

extracted_text = ' \n'.join(json_text)

with open(r'd:\STUDY 2\MediaEditor\01\test\extracted_lyrics.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)

with open(r'd:\STUDY 2\MediaEditor\01\test\lyrics_text\lyrics.txt', 'r', encoding='utf-8') as f:
    original_text = f.read()

# Optional: Output some diff
import sys
print("JSON Characters:", len(extracted_text))
print("Original Characters:", len(original_text))
print("Done")
