"""
part2/ipa_converter.py — Hinglish → IPA Unified Representation

Standard G2P tools (like espeak-ng) fail on code-switched Hinglish because:
  - Hindi words are often written in Roman script (transliteration)
  - English phonology ≠ Hindi phonology (retroflex sounds, aspirates, etc.)
  - Word boundaries between Hindi and English are phonologically distinct

This module implements a custom Hinglish G2P:
  1. Language-tag each word using simple heuristics + LID context
  2. For English words:  use CMU dict / phonemizer (espeak)
  3. For Hindi words (in Roman): custom phoneme mapping for Hindi phonology

Key Hinglish phonological phenomena handled:
  - Retroflex consonants: ट ड ण → /ʈ ɖ ɳ/
  - Aspirated stops: ख घ → /kʰ ɡʱ/
  - Dental vs retroflex distinction
  - Schwa deletion (final vowel in many Hindi words)
  - Nasalisation anusvara /ã/
"""

import re
from typing import Optional


# ─────────────────────────── Hindi G2P Map ──────────────────────────────────

# Romanised Hindi → IPA (Hinglish phonology)
HINDI_ROMAN_TO_IPA = {
    # Consonants
    "kh":  "kʰ",  "gh":  "ɡʱ",  "ch": "tʃ",  "chh": "tʃʰ",
    "jh":  "dʒʱ", "th":  "t̪ʰ",  "dh": "d̪ʱ",  "nh":  "nʱ",
    "ph":  "pʰ",  "bh":  "bʱ",  "rh": "ɽ",   "lh":  "lʱ",
    "sh":  "ʃ",   "zh":  "ʒ",   "ng": "ŋ",
    # Retroflex
    "T":   "ʈ",   "D":   "ɖ",   "N":  "ɳ",   "Th": "ʈʰ",  "Dh": "ɖʱ",
    "R":   "ɽ",
    # Simple consonants
    "k":  "k",   "g":  "ɡ",   "c": "tʃ",  "j":  "dʒ",
    "t":  "t̪",   "d":  "d̪",   "n": "n",   "p":  "p",
    "b":  "b",   "m":  "m",   "y": "j",   "r":  "r",
    "l":  "l",   "v":  "ʋ",   "w": "ʋ",   "s":  "s",
    "h":  "ɦ",   "f":  "f",   "z": "z",   "x":  "x",
    "q":  "q",
    # Vowels
    "aa": "aː",  "ee": "iː",  "oo": "uː",  "ae": "ɛː",  "oe": "oː",
    "ai": "ɛ",   "au": "ɔ",   "an": "ãː",
    "a":  "ə",   "e":  "eː",  "i":  "ɪ",   "o":  "oː",  "u":  "ʊ",
}

# Common Hindi technical/function words in Roman → IPA
HINDI_WORD_IPA = {
    "hai":    "ɦɛː",
    "hain":   "ɦɛ̃ː",
    "ka":     "kəː",
    "ke":     "keː",
    "ko":     "koː",
    "ki":     "kiː",
    "mein":   "meːɪ̃",
    "yeh":    "jeː",
    "aur":    "ɔːr",
    "nahi":   "nəɦiː",
    "nahin":  "nəɦiː",
    "ek":     "eːk",
    "do":     "d̪oː",
    "toh":    "t̪oː",
    "bhi":    "bʱiː",
    "jo":     "dʒoː",
    "woh":    "ʋoː",
    "kya":    "kjɑː",
    "matlab": "mɐt̪ləb",
    "matlab": "mɐt̪ləb",
    "basically": "beɪsɪkəli",
    "toh":    "t̪oː",
    "wala":   "ʋɑːlɑː",
    "wali":   "ʋɑːliː",
    "isliye": "ɪslijeː",
    "lekin":  "leːkɪn",
    "phir":   "pʱɪr",
    "abhi":   "əbʱiː",
    "bilkul": "bɪlkʊl",
    "samajh": "səmɐdʒ",
    "dekho":  "d̪eːkʰoː",
    "suno":   "sʊnoː",
}

# ─────────────────────────── English G2P ────────────────────────────────────

def english_g2p(word: str) -> str:
    """
    English word → IPA using phonemizer (espeak-ng backend).
    Falls back to simple letter-by-letter if phonemizer unavailable.
    """
    try:
        from phonemizer import phonemize
        ipa = phonemize(
            word,
            backend="espeak",
            language="en-us",
            with_stress=True,
            strip=True,
        )
        return ipa.strip()
    except Exception:
        # Fallback: simple rule-based English IPA
        return _simple_english_ipa(word)


def _simple_english_ipa(word: str) -> str:
    """Very basic English IPA approximation when phonemizer not available."""
    rules = [
        (r"tion",  "ʃən"), (r"sion",  "ʒən"), (r"ck",   "k"),
        (r"ph",    "f"),   (r"gh",    ""),    (r"th",   "θ"),
        (r"sh",    "ʃ"),   (r"ch",    "tʃ"),  (r"wh",   "w"),
        (r"ng",    "ŋ"),   (r"ee|ea", "iː"),  (r"oo",   "uː"),
        (r"ai|ay", "eɪ"), (r"ow|ou", "aʊ"),  (r"oi|oy","ɔɪ"),
        (r"igh",   "aɪ"), (r"a_e",   "eɪ"),
        (r"a",     "æ"),   (r"e",    "ɛ"),    (r"i",   "ɪ"),
        (r"o",     "ɒ"),   (r"u",    "ʌ"),
    ]
    result = word.lower()
    for pattern, replacement in rules:
        result = re.sub(pattern, replacement, result)
    return result


# ─────────────────────────── Hindi (Roman) G2P ──────────────────────────────

def hindi_roman_g2p(word: str) -> str:
    """
    Romanised Hindi word → IPA.
    Uses dictionary lookup first, then rule-based mapping.
    """
    w = word.lower().strip()
    if w in HINDI_WORD_IPA:
        return HINDI_WORD_IPA[w]

    # Rule-based: match digraphs first, then unigraphs
    result = ""
    i = 0
    while i < len(word):
        # Try 3-char sequences first (e.g., "chh")
        if i + 3 <= len(word) and word[i:i+3].lower() in HINDI_ROMAN_TO_IPA:
            result += HINDI_ROMAN_TO_IPA[word[i:i+3].lower()]
            i += 3
        # Then 2-char
        elif i + 2 <= len(word) and word[i:i+2].lower() in HINDI_ROMAN_TO_IPA:
            result += HINDI_ROMAN_TO_IPA[word[i:i+2].lower()]
            i += 2
        # Then 1-char
        elif word[i].lower() in HINDI_ROMAN_TO_IPA:
            result += HINDI_ROMAN_TO_IPA[word[i].lower()]
            i += 1
        else:
            result += word[i]   # Unknown character — keep as-is
            i += 1

    # Schwa deletion: final short 'a' is often silent in Hindi
    result = re.sub(r"ə$", "", result)
    return result


# ─────────────────────────── Word Language Tagging ──────────────────────────

# Words that are strongly Hindi (in Roman script)
_HINDI_INDICATORS = set(HINDI_WORD_IPA.keys()) | {
    "matlab", "samajh", "toh", "yeh", "aur", "nahi", "kyun", "isliye",
    "woh", "kya", "abhi", "phir", "bhi", "lekin", "agar", "phle",
}

def detect_word_lang(word: str, lid_hint: Optional[str] = None) -> str:
    """
    Heuristically determine if a word is 'en' (English) or 'hi' (Hindi).
    lid_hint: 'en' or 'hi' from frame-level LID context.
    """
    w = word.lower().strip()

    # Known Hindi words
    if w in _HINDI_INDICATORS:
        return "hi"

    # Devanagari script → definitely Hindi
    if re.search(r"[\u0900-\u097F]", word):
        return "hi"

    # Purely ASCII, no diacritics → likely English if not a Hindi indicator
    if re.match(r"^[a-zA-Z]+$", word) and lid_hint == "en":
        return "en"

    return lid_hint or "en"     # Default to English


# ─────────────────────────── Main Converter ─────────────────────────────────

def text_to_ipa(
    text: str,
    lid_segments: Optional[list] = None,
) -> tuple[str, list]:
    """
    Convert Hinglish text to unified IPA string.

    Args:
        text:         Raw transcript (Hinglish / code-switched).
        lid_segments: Optional [(start, end, 'en'/'hi'), ...] from LID.

    Returns:
        ipa_string (str):   Full IPA representation of the utterance.
        word_ipa_list (list): [(word, lang, ipa), ...]
    """
    words = text.split()
    word_ipa = []

    # Build a word-index → language mapping from LID segments
    word_lang_map = {}
    if lid_segments:
        # Approximate: each word gets the language of its temporal region
        # (We use word position as proxy for time)
        n = len(words)
        for i, (start, end, lang) in enumerate(lid_segments):
            frac_start = start / (lid_segments[-1][1] + 1e-6)
            frac_end   = end   / (lid_segments[-1][1] + 1e-6)
            for wi in range(int(frac_start * n), min(int(frac_end * n) + 1, n)):
                word_lang_map[wi] = lang

    for i, word in enumerate(words):
        # Strip punctuation for IPA lookup
        clean = re.sub(r"[^a-zA-Z\u0900-\u097F']", "", word)
        if not clean:
            continue

        lang_hint = word_lang_map.get(i, None)
        lang = detect_word_lang(clean, lang_hint)

        if lang == "hi":
            ipa = hindi_roman_g2p(clean)
        else:
            ipa = english_g2p(clean)

        word_ipa.append((word, lang, ipa))

    ipa_string = " ".join(ipa for _, _, ipa in word_ipa)
    return ipa_string, word_ipa


# ─────────────────────────── Devanagari G2P ─────────────────────────────────

# Devanagari consonants → IPA
DEVANAGARI_TO_IPA = {
    "क": "k",  "ख": "kʰ", "ग": "ɡ",  "घ": "ɡʱ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ","ज": "dʒ", "झ": "dʒʱ","ञ": "ɲ",
    "ट": "ʈ",  "ठ": "ʈʰ", "ड": "ɖ",  "ढ": "ɖʱ", "ण": "ɳ",
    "त": "t̪",  "थ": "t̪ʰ", "द": "d̪",  "ध": "d̪ʱ", "न": "n",
    "प": "p",  "फ": "pʰ", "ब": "b",  "भ": "bʱ", "म": "m",
    "य": "j",  "र": "r",  "ल": "l",  "व": "ʋ",
    "श": "ʃ",  "ष": "ʂ",  "स": "s",  "ह": "ɦ",
    "क्ष": "kʃ","त्र": "tr̪","ज्ञ": "dʒɲ",
    # Vowels
    "अ": "ə",  "आ": "aː", "इ": "ɪ",  "ई": "iː",
    "उ": "ʊ",  "ऊ": "uː", "ए": "eː", "ऐ": "ɛ",
    "ओ": "oː", "औ": "ɔː", "ऋ": "rɪ",
    # Matras (vowel signs)
    "ा": "aː", "ि": "ɪ",  "ी": "iː", "ु": "ʊ",
    "ू": "uː", "े": "eː", "ै": "ɛ",  "ो": "oː",
    "ौ": "ɔː", "ं": "̃",   "ः": "h",  "्": "",
}


def devanagari_to_ipa(text: str) -> str:
    """Convert Devanagari script text to IPA."""
    result = ""
    for char in text:
        result += DEVANAGARI_TO_IPA.get(char, char)
    return result
