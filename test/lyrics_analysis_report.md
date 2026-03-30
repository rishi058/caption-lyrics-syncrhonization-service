# Lyrics Transcription Breakdown Report

This document contains a side-by-side analysis of the original lyrics (`lyrics.txt`) versus the text extracted from the audio transcription (`Die_For_You_raw.json`). The text was reassembled by concatenating the `chunks -> text` arrays.

## Summary of Accuracy
- **Beginning (Verse 1 & Pre-Chorus 1):** Highly accurate. The transcription model performed well, capturing almost every word with only minor missed ad-libs.
- **Middle (Chorus 1 & Verse 2):** Severe degradation. The model began misinterpreting entire phrases, replacing the actual lyrics with completely different ("hallucinated") sentences. 
- **End (Bridge & Final Chorus):** Near-total failure. The original lyrics structure is completely lost, and the model outputs disjointed and nonsensical phrases, missing major blocks of the song completely.

---

## Detailed Side-by-Side Analysis

### Verse 1 & Pre-Chorus (High Accuracy)
The transcription does very well here, only missing some minor background vocals.

| Original Lyrics | JSON Transcription | Issues |
| --- | --- | --- |
| I just can't say I don't love you (Yeah) | I just can't say I don't love you. | **Lost:** The ad-lib `(Yeah)` is missing entirely. |
| Baby, let me tell the truth, yeah | Baby, let me tell the truth, yes | **Misinterpreted:** `yeah` was transcribed as `yes`. |
| I try to find a reason | I try to find reason | **Lost:** The article `a`. |
| And I know that you're worth it | But I know that you're worth it | **Misinterpreted:** `And` transcribed as `But`. |
| I can't walk away, oh | I can't walk away | **Lost:** The vocal ad-lib `oh`. |

---

### Chorus 1 (Heavy Misinterpretation & Loss)
The repetitive nature of the chorus causes the transcription model to collapse and merge lines together.

| Original Lyrics | JSON Transcription | Issues |
| --- | --- | --- |
| Just know that I would die for you / Baby, I would die for you, yeah | this long that I would die baby I would die baby I would die baby I would die baby | **Misinterpreted:** `Just know` became `this long`. The structure was compressed into a repetition of `baby I would die`. |
| The distance and the time between us / It'll never change my mind 'cause | *(None)* | **Completely Lost:** These two entire lines are completely missing from the transcription. |

---

### Verse 2 (Severe Hallucinations)
The model completely loses the thread here and generates phrases that do not exist in the song whatsoever.

| Original Lyrics | JSON Transcription | Issues |
| --- | --- | --- |
| I'm findin' ways to manipulate the feelin' you're goin' through | I'm finding ways to stay concentrated on what I gotta do | **Misinterpreted:** Entire phrase hallucinated / completely changed. |
| But, baby girl, I'm not blamin' you | Baby boy it's so hard on you | **Misinterpreted:** Meaning flipped entirely (`baby girl` -> `Baby boy`). |
| Just don't blame me, too, yeah | And yes, I'm blaming you | **Misinterpreted:** Literal opposite meaning transcribed. |
| 'Cause I can't take this pain forever | And you know I can't fake it now or never | **Misinterpreted:** `take this pain` -> `fake it now`. |
| And you won't find no one that's better | And you won't send you waiting | **Misinterpreted:** Entirely hallucinated text. |
| 'Cause I'm right for you, babe / I think I'm right for you, babe | But you think we might be better Better me and you Yeah, I know you do | **Misinterpreted:** Rhyme scheme completely guessed and re-written by the model. |

---

### Pre-Chorus 2 (Added Hallucinations)
| Original Lyrics | JSON Transcription | Issues |
| --- | --- | --- |
| And I know that you're worth it | And I know you deserve it | **Misinterpreted:** `worth it` -> `deserve it`. |
| I can't walk away, oh | I can't walk away In the power of God. | **Misinterpreted:** Added a completely hallucinated or misheard ad-lib (`In the power of God.`). |

---

### Chorus 2 (Loss of Structure)
| Original Lyrics | JSON Transcription | Issues |
| --- | --- | --- |
| Even though we're goin' through it | Moving in a mix of things. | **Misinterpreted:** `Even though` -> `Moving in`. |
| And it makes you feel alone | No one just knew | **Misinterpreted:** Completely incorrect text. |
| Just know that I would die for you | that I would die. | **Lost:** Partial line loss. |
| The distance and the time between us | Baby, the distance in the top. Between us, | **Misinterpreted:** `and the time` -> `in the top`. |
| It'll never change my mind 'cause | they'll never change the world. | **Misinterpreted:** `my mind 'cause` -> `the world`. |

---

### Bridge & Outro (Near Complete Failure)
The model struggles with the vocal layering and pitch jumps in the bridge and outro, leading to major omissions and inappropriate text generation.

| Original Lyrics | JSON Transcription | Issues |
| --- | --- | --- |
| I would die for you, I would lie for you | *(None)* | **Completely Lost:** The first delivery of this line is missing. |
| Keep it real with you, I would kill for you, my baby | Keep it real with you All we care for you won't be in pain | **Misinterpreted:** Severe hallucination in the second half of the phrase. |
| I'm just sayin', yeah / I would die for you, I would lie for you | Let me see how it's life in, how it's life in, | **Misinterpreted:** Massive breakdown in transcription logic. |
| Keep it real with you, I would kill for you, my baby | keep it real with you, I'm like you, fuck you, my baby. | **Misinterpreted:** Added non-existent explicit lyrics (`fuck you`). |
| Na-na-na, na-na-na, na-na-na | No, no, no, no, no. | **Misinterpreted:** Scatting/vocals misheard as words. |
| Even though we're goin' through it / And it makes you feel alone | And it'll get sick and long to the hour of the wind. | **Misinterpreted:** Completely incomprehensible hallucination. |
| Just know that I would die for you | *(None)* | **Completely Lost.** |
| Baby, I would die for you, yeah | Baby, I'm your child. | **Misinterpreted:** Meaning completely altered. |
| The distance and the time between us / It'll never change my mind 'cause / Baby, I would die for you / Baby, I would die for you, yeah (Oh, babe) / Die for you | *(None)* | **Completely Lost:** The entire ending sequence is completely missing from the JSON chunks. |
