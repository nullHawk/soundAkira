"""Script-agnostic text helpers (work for Latin, Devanagari and other Indic scripts)."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from soundakira.types import Word

# Sentence-final punctuation across scripts: Latin, Devanagari danda (। ॥), CJK, Arabic.
SENTENCE_END = frozenset(".!?।॥。！？؟…")
CLAUSE_END = frozenset(",;:،、，；：—–")

# Scripts written without spaces between words.
UNSPACED_LANGUAGES = frozenset({"zh", "ja", "th", "lo", "my", "km", "yue", "bo"})

_LANGUAGE_ALIASES = {
    "english": "en",
    "eng": "en",
    "hindi": "hi",
    "hin": "hi",
    "hinglish": "hi-en",
    "bengali": "bn",
    "bangla": "bn",
    "ben": "bn",
    "tamil": "ta",
    "tam": "ta",
    "telugu": "te",
    "tel": "te",
    "marathi": "mr",
    "mar": "mr",
    "gujarati": "gu",
    "guj": "gu",
    "kannada": "kn",
    "kan": "kn",
    "malayalam": "ml",
    "mal": "ml",
    "punjabi": "pa",
    "pan": "pa",
    "odia": "or",
    "oriya": "or",
    "ori": "or",
    "urdu": "ur",
    "urd": "ur",
    "assamese": "as",
    "asm": "as",
}

# ISO 639-1 -> 639-2/B as used in container stream tags (e.g. Matroska "eng").
_ISO2_TO_ISO3 = {
    "en": "eng",
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "gu": "guj",
    "kn": "kan",
    "ml": "mal",
    "pa": "pan",
    "or": "ori",
    "ur": "urd",
    "as": "asm",
    "fr": "fre",
    "de": "ger",
    "es": "spa",
    "ja": "jpn",
    "zh": "chi",
}


def normalize_language(code: str | None) -> str | None:
    """Map user-facing language names/codes to short codes ('English' -> 'en')."""
    if not code:
        return None
    c = code.strip().lower().replace("_", "-")
    return _LANGUAGE_ALIASES.get(c, c)


def container_language_tags(code: str | None) -> list[str]:
    """Stream-tag spellings that correspond to a language code."""
    lang = normalize_language(code)
    if not lang:
        return []
    base = lang.split("-")[0]
    tags = {lang, base}
    if base in _ISO2_TO_ISO3:
        tags.add(_ISO2_TO_ISO3[base])
    return sorted(tags)


def join_words(words: Iterable[Word], include_events: bool = False) -> str:
    return "".join(w.text for w in words if include_events or w.kind == "word").strip()


def spoken_char_count(text: str) -> int:
    """Letters, combining marks (matras) and digits: a speaking-rate proxy that
    works for any script."""
    return sum(1 for ch in text if unicodedata.category(ch)[0] in "LMN")


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in text.split():
        tok = "".join(ch for ch in raw if unicodedata.category(ch)[0] in "LMN").lower()
        if tok:
            tokens.append(tok)
    return tokens


def repetition_ratio(text: str, n: int = 3) -> float:
    """Share of repeated word n-grams (0 = none). High values are a strong
    signal of ASR hallucination loops ("thank you thank you thank you...")."""
    tokens = tokenize(text)
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def ends_sentence(text: str) -> bool:
    t = text.rstrip().rstrip("\"'”’)]»")
    return bool(t) and t[-1] in SENTENCE_END


def ends_clause(text: str) -> bool:
    t = text.rstrip().rstrip("\"'”’)]»")
    return bool(t) and t[-1] in CLAUSE_END


def with_leading_space(token: str, language: str | None) -> str:
    """Adapt a bare token to the `Word.text` whitespace convention."""
    token = token.strip()
    lang = (normalize_language(language) or "").split("-")[0]
    return token if lang in UNSPACED_LANGUAGES else " " + token
