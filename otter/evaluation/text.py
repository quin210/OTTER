"""Small Vietnamese text utilities shared by validators, retrieval and filters."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

_WORD = re.compile(r"\w+", flags=re.UNICODE)
_QUOTED = re.compile(r"[“\"«]([^”\"»]{1,400})[”\"»]")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def norm(text: str) -> str:
    """Lower-cased NFC text with collapsed whitespace and no punctuation."""
    return " ".join(_WORD.findall(nfc(text).lower()))


def tokens(text: str) -> list[str]:
    """Syllable tokens (Vietnamese words are space-separated syllables)."""
    return _WORD.findall(nfc(text).lower())


def bigram_tokens(text: str) -> list[str]:
    """Syllable unigrams + bigrams; bigrams approximate Vietnamese compound words for BM25."""
    t = tokens(text)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = norm(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def template_of(question: str) -> str:
    """Replace quoted content and digits so templated questions collapse to one key."""
    q = _QUOTED.sub("“…”", nfc(question))
    q = re.sub(r"\d+([.,]\d+)?", "#", q)
    return " ".join(q.lower().split())


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def short_hash(parts: Iterable[str], n: int = 12) -> str:
    return sha1("\x1f".join(parts))[:n]
