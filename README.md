# 🎵 Lyrics Synchronization Engine

A multilingual lyrics-to-audio alignment system supporting **English** and **Hindi (Devanagari / Hinglish)** with intelligent fallback strategies, LLM-assisted normalization via a centralized `RefineHinglish` pipeline (`helpers/hi/llm`), and gap-filling for mixed-language content.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Processing Pipelines](#processing-pipelines)
  - [Case 1: English Songs](#case-1-english-songs)
  - [Case 2: Hindi Songs](#case-2-hindi-songs)
- [Hinglish Support — Deep Dive](#hinglish-support--deep-dive)
- [Handling English Words Inside Hindi Songs](#handling-english-words-inside-hindi-songs)
- [Unified LLM Pipeline (`RefineHinglish`)](#unified-llm-pipeline-refinehinglish)
  - [Pre-Transliteration](#pre-transliteration)
  - [LLM Refinement Layer](#llm-refinement-layer)
- [Key Functions](#key-functions)
  - [fill_english_gaps()](#fill_english_gaps)
  - [format_word_text()](#format_word_text)
- [Models Used](#models-used)
- [Output Modes](#output-modes)
- [Edge Cases & Known Behaviors](#edge-cases--known-behaviors)
- [Data Flow Diagram](#data-flow-diagram)

---

## Overview

This engine synchronizes song lyrics with audio — producing word-level timestamps — for two language families:

| Language | Script Variants Supported | Transcription | Alignment |
|----------|--------------------------|---------------|-----------|
| English | Latin | Whisper `large-v3` | `jonatasgrosman/wav2vec2-large-xlsr-53-english` |
| Hindi | Devanagari, Hinglish (Latin) | Whisper `large-v3` | `theainerd/Wav2Vec2-large-xlsr-hindi` |

The system is **lyrics-aware**: when lyrics are provided, it uses forced alignment instead of transcription, yielding far more accurate word timestamps.

---

## Architecture
```
Audio Input
    │
    ├─── Language = EN ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                                                                                 │
    │        Lyrics Provided?                                                                                                                                         │
    │        ├── YES → wav2vec2 (EN) forced alignment                                                                                                                 │
    │        └── NO  → Whisper large-v3 transcription → wav2vec2 (EN)                                                                                                 │
    │                                                                                                                                                                 ▼
    │                                                                                                                                                        Word-level Timestamps
    │                                                                                                                                                        (Latin output)
    │
    └─── Language = HI ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
             │                                                                                                                                                        │
             │   Lyrics Provided?                                                                                                                                     │
             │   ├── YES → Script Detection                                                                                                                           │ 
             │   │         ├── Hinglish → convert hindi-latin to Devanagari-script                                                                                    │ 
             │   │         │              & leave english words as it is [using LLM only]                                                                             │ 
             │   │         │              (store map: devanagari → hinglish)                                                                                          │ 
             │   │         │              align seperately → merge                                                                                                    │
             │   │         │                                                                                                                                          │
             │   │         └── Devanagari → check for english words(also consider english words in devanagari)[using LLM only]                                        │
             │   │                          (seperate them)                                                                                                           │
             │   │                          align seperately → merge                                                                                                  │
             │   │                                                                                                                                                    │
             │   └── NO → Whisper large-v3 (outputs Devanagari + English)                                                                                             │
             │            (Use LLM to validate, error due to accent)                                                                                                  │ 
             │            (i.e check for english words in devanagari)                                                                                                 │
             │            (seperate them)                                                                                                                             │
             │            align seperately → merge                                                                                                                    │
             │                                                                                                                                                        │
             │   ┌────────────────────────────────────────────────────────┐                                                                                           │
             │   │  Wav2Vec2-large-xlsr-hindi alignment (Devanagari in)   │                                                                                           │
             │   └────────────────────────────────────────────────────────┘                                                                                           │
             │                         │                                                                                                                              │
             │             English words in song?                                                                                                                     │
             │             └── YES → fill_english_gaps()                                                                                                              │
             │                                                                                                                                                        │
             │   devanagari_output(may not be purely devanagari, which is fine)                                                                                       │
             │   |                                                                                                                                                    │
             │   ├── True  → Lyrics Provided Hinglish → No issue(we converted Hinglish to Devanagari-script initially)                                                │
             │   │           Lyrics Provided Devanagari → No issue(Devanagari input -> Devanagaru output)                                                             │
             │   │           No Lyrics → No issue(whisperX transcribtion is in devanagari)                                                                            │
             │   │                                                                                                                                                    │
             │   ├── False → Lyrics Provided Hinglish →  No issue(will use that "map" to convert back to Hinglish)                                                    │
             │               Lyrics Provided Devanagari → (Need to convert to devanagari script to Hinglish) → [using indic_transliteration + refine using LLM]       │
             │               No Lyrics → (Need to convert to devanagari script to Hinglish) → [using indic_transliteration + refine using LLM]                        │
             │                                                                                                                                                        │
             └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ 
                                                                                                                                                                      ▼
                                                                                                                                                        Word-level Timestamps
                                                                                                                                                        (Hinglish or Devanagari[+ English words if present])
```
---

## Processing Pipelines

### Case 1: English Songs

#### Sub-case A — No Lyrics Provided
1. **Transcription**: Audio is passed to `Whisper large-v3`, which produces a raw transcript with approximate word/segment timestamps.
2. **Alignment**: The transcript is then refined using `wav2vec2-large-xlsr-53-english` forced alignment to produce accurate word-level timestamps.
3. **Output**: Word timestamps in Latin script.

#### Sub-case B — Lyrics Provided
1. **Skip transcription** entirely.
2. **Forced Alignment**: Provided lyrics are directly aligned to audio using `wav2vec2-large-xlsr-53-english`.
3. **Output**: Word timestamps in Latin script, tightly matched to the user-supplied lyrics text.

---

### Case 2: Hindi Songs

#### Sub-case A — No Lyrics Provided
1. **Transcription**: Whisper `large-v3` transcribes the audio. Its Hindi output is natively in **Devanagari script**.
2. **LLM Validation**: The transcript is passed to `process_helper.process_devanagari_script()`. It transliterates words to ITRANS and runs the LLM to identify English words (which Whisper often incorrectly writes in Devanagari due to accent).
3. **Alignment**: Hindi (Devanagari) words are aligned using `theainerd/Wav2Vec2-large-xlsr-hindi`.
4. **English gap-filling**: English words identified by the LLM are aligned separately via `fill_english_gaps()`.
5. **Output Control**:
   - `devanagari_output = true` → export as-is in Devanagari.
   - `devanagari_output = false` → convert to Hinglish using the `word_mapp` populated during the LLM processing.

#### Sub-case B — Lyrics Provided (Hinglish or Devanagari input)
1. **Script Detection**: Determine whether the provided lyrics are in Hinglish (Latin alphabet) or Devanagari.
2. **Processing Pipeline**:
   - Depending on the detected script, either `process_devanagari_script()` or `process_latin_script()` is invoked.
   - Initial rule-based transliteration runs first (via `indict_transliteration` mapping to ITRANS or Devanagari).
   - The unified `RefineHinglish` pipeline corrects the `lang` tags (identifying English words) and refines the transliteration to fix spelling errors.
   - This process builds a **reverse-map** that maps the final Devanagari word to its refined Latin spelling.
   - Align the Devanagari lyrics with the Hindi wav2vec2 model.
   - Run `fill_english_gaps()` for any English words that were present in the lyrics.
   - Apply `format_word_text()` logic on output which restores the **refined Hinglish spellings** using the reverse mapping if `devanagari_output` is `false`.

---

## Hinglish Support — Deep Dive

Hinglish is Hindi written in Latin characters. It is informal and has no standardized spelling — `"kyun"`, `"kyu"`, `"kyoon"` are all valid representations of `"क्यों"`. Rule-based transliteration systems like ITRANS fail on this input because:

- They require **standardized Latin encodings**, not casual typed forms.
- They cannot infer intent from ambiguous romanizations.
- They produce incorrect Devanagari for colloquial contractions.

### Why LLM normalization is necessary

The single `RefineHinglish` agent acts as a **semantic translator** — it understands that `"pyar"` is `"प्यार"`, `"teri"` is `"तेरी"`, and `"hoga"` is `"होगा"` — none of which ITRANS reliably handles. The LLM response also correctly assigns each word a `lang` (`"hi"` or `"en"`), which drives the English gap-fill logic downstream.

### Reverse-mapping for output fidelity

When a user provides Hinglish lyrics, they expect Hinglish back in the sync data — not Devanagari, and not rigid ITRANS transliterations. The reverse-map (`word_mapp`) stores:

```
"तेरा" → "tera"
"प्यार" → "pyaar"   # user's original/refined spelling preserved
```

This ensures the final sync output provides a natural and correct Hinglish representation, character for character.

---

## Handling English Words Inside Hindi Songs

Real-world Hindi songs frequently contain English words — lines like *"baby mujhe chhod ke mat ja"* or *"feeling something"*. These words are problematic because:

1. The **Hindi wav2vec2 model is trained on Devanagari phonemes** — it cannot align Latin-script English words.
2. They are simply **absent from `sync_data`** after Hindi alignment, leaving gaps in the timeline.

### Solution: `fill_english_gaps()`

This function runs post-Hindi-alignment whenever English words are detected. It bridges the silence left by unaligned English words using a targeted English alignment model.

**Step-by-step logic:**

```
1. Collect all Hindi word timestamps → sorted timeline
2. Identify gaps between consecutive Hindi words
3. Map each English word (from the LLM's lang="en" tags) to a gap
4. For each gap:
   a. Slice the audio segment for that gap
   b. Run wav2vec2 (English) forced alignment on the mini-segment
   c. If alignment succeeds → insert timestamps into sync_data
   d. If alignment fails → _proportional_fallback():
      split gap duration equally among English words in that gap
5. Merge Hindi + English timestamps → final sorted sync_data
```

This means English words in a Hindi song are **never silently dropped** — they either get precise timestamps from the English aligner, or reasonable proportional estimates.

---

## Unified LLM Pipeline (`RefineHinglish`)

All contextual normalization is centralized within `helpers/hi/process_helper.py` which passes words to `RefineHinglish` (`helpers/hi/llm/refine_lyrics.py`). This performs transliteration refinement and language detection in a single, efficient LLM pass via **Cohere `command-a-03-2025`** (using LangChain structured outputs).

### Pre-Transliteration
Before the LLM is hit, rules-based pre-processing steps run (`helpers/hi/transliteration.py`):
- Converts the string to a list of `{lat, dev, lang}` dictionary tokens.
- Fills in rough ITRANS/Devanagari defaults using `indic_transliteration`.
- Attempts a naive `lang` tag initialization via `wordfreq`.

### LLM Refinement Layer
The populated dictionary tokens are sent in a batch to the `RefineHinglish` LLM prompt:
- **Validates Tags**: Corrects naive mappings (e.g. tagging true English words as `"en"` and catching "Hinglish-looking" Hindi words like `"Aja"` or `"ko"` to assign them `"hi"`).
- **Refines Spelling**: Corrects rough ITRANS phoneme errors (e.g., changes `"Mujhay"` -> `"मुझे"` accurately).
- **Graceful Fallback**: If the LLM API call fails, the pipeline automatically falls back to the original dictionary array using the unrefined `indic_transliteration` results, ensuring processing completes.

---

## Key Functions

### `fill_english_gaps()`

**Purpose**: Align English words that appear inside Hindi songs, using the gaps left in the Hindi alignment timeline.

**Triggered when**: `language = "hi"` AND English words (`lang: "en"`) were found by the RefineHinglish analysis.

**Gap assignment strategy**:
- Gaps are sorted by duration (longest first) if there are more gaps than English word clusters.
- English words are distributed across gaps proportionally, preserving lyric order.

**Fallback — `_proportional_fallback()`**:
- Triggered per-gap if the English wav2vec2 aligner fails on a segment.
- Divides the gap duration evenly among the English words assigned to that gap.
- Produces `start` and `end` timestamps that are approximate but structurally valid — the sync data remains complete.

---

### `format_word_text()` / `process_hi` Merging

**Purpose**: Determines the final text string for each word in the output sync data.

**Priority order** (applied per word):

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | Word exists in `word_mapp` reverse-map | Restore the refined/original Hinglish spelling |
| 2 | Word is English (`lang: "en"`) | Pass through unchanged |
| 3 | `devanagari_output = True` | Provide the aligned Devanagari text |

This ensures that:
- User-supplied Hinglish spellings are **always honored** and output naturally.
- Auto-transcribed or Devanagari-provided lyrics get **natural Hinglish** output (via LLM mapping) when requested.
- English words in Hindi songs **appear as-is** in the output.

---

## Models Used

| Model | Source | Role |
|-------|--------|------|
| `openai/whisper-large-v3` | OpenAI / HuggingFace | Transcription for both EN and HI when lyrics not provided |
| `jonatasgrosman/wav2vec2-large-xlsr-53-english` | HuggingFace | Forced alignment for English lyrics/transcripts |
| `theainerd/Wav2Vec2-large-xlsr-hindi` | HuggingFace | Forced alignment for Hindi (Devanagari input only) |
| Cohere `command-a-03-2025` | Cohere (via LangChain) | Consolidated LLM tasks: correcting language tags, transliteration refinement, and English detection |

---

## Output Modes

### English
Always outputs in **Latin script**. No flags needed.

### Hindi

Controlled by the `devanagari_output` flag:

| Flag Value | Output Format | When to use |
|------------|---------------|-------------|
| `true` | Devanagari (`"तेरा प्यार"`) | Apps rendering Hindi natively; databases; NLP pipelines |
| `false` | Hinglish / Latin (`"tera pyaar"`) | Karaoke displays; apps without Devanagari font support; user-facing sync |

When lyrics are processed and `devanagari_output = false`, the `word_mapp` reverse-map guarantees realistic Hinglish representations, rather than rigid ITRANS literals.

---

## Edge Cases & Known Behaviors

| Scenario | Behavior |
|----------|----------|
| English words in Hindi song | Singled out by `RefineHinglish`, assigned `lang: "en"`, and handled gracefully by `fill_english_gaps()` logic |
| LLM API failure during refinement | Falls back to rule-based `indic_transliteration` defaults; accuracy degrades for irregular spellings |
| English aligner fails on a gap segment | `_proportional_fallback()` assigns equal time slices; word is not dropped |
| Mixed Devanagari + Hinglish in provided lyrics | Script detection checks the document. Words are parsed and routed efficiently to the `RefineHinglish` LLM step |
| Very short English word in a very short gap | English aligner may produce low-confidence alignment; proportional fallback is preferred |

---

## Data Flow Diagram

```
User Provides:  Audio + [optional lyrics] + language flag + devanagari_output flag
                     │
          ┌──────────┴───────────┐
        EN path               HI path
          │                       │
   Lyrics?                  Lyrics?
   ├─ NO: Whisper           ├─ NO: Whisper → Devanagari output
   │       ↓                │         ↓
   │  wav2vec2-EN align      │   wav2vec2-HI align
   │                         │         ↓
   └─ YES: wav2vec2-EN align  │   fill_english_gaps()
                              │         ↓
                              │   devanagari_output?
                              │   ├─ true  → export Devanagari
                              │   └─ false → ITRANS export
                              │
                              ├─ YES (Hinglish):
                              │   normalize_hinglish_with_llm()
                              │         ↓  [builds reverse-map]
                              │   wav2vec2-HI align (Devanagari)
                              │         ↓
                              │   fill_english_gaps()
                              │         ↓
                              │   format_word_text()
                              │   [reverse-map → Hinglish output]
                              │
                              └─ YES (Devanagari):
                                  wav2vec2-HI align
                                        ↓
                                  fill_english_gaps()
                                        ↓
                                  devanagari_output?
                                  ├─ true  → export Devanagari
                                  └─ false → ITRANS export
```

---

*Built for production use with real-world Hinglish lyrics, casual user input, and mixed-language songs.*
*LLM calls powered by Cohere `command-a-03-2025` via LangChain with structured output.*