"""Vietnamese-aware text helpers shared by the store and the builder.

Diacritics are meaningful in Vietnamese ("bào" != "bảo"), so the canonical key
keeps them (NFC + casefold); `ascii_fold` is only used to build readable ids.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WORD = re.compile(r"\w+", re.UNICODE)

# Small function-word list used only to build retrieval queries / overlap scores.
VI_STOPWORDS = frozenset("""
là của và các những có được trong cho với không gì nào này đó thì mà một hai khi nếu để từ theo về
ra vào bị do như cũng đã sẽ đang rất hay hoặc tại sao vì nên cần phải hãy em ta ở trên dưới
bao nhiêu thế làm sự việc người cái điều đây kia nhau hơn nhất lại vẫn còn chỉ đều mỗi
""".split())


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def norm_key(s: str) -> str:
    """Canonical comparison key: NFC, casefold, punctuation -> space, collapsed whitespace."""
    s = nfc(s).casefold()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def ascii_fold(s: str) -> str:
    s = nfc(s).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def slugify(s: str, max_len: int = 60) -> str:
    s = ascii_fold(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len].rstrip("_") or "x"


def short_hash(*parts: str, n: int = 12) -> str:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def syllables(s: str, drop_stopwords: bool = False) -> list[str]:
    toks = _WORD.findall(nfc(s).casefold())
    if drop_stopwords:
        toks = [t for t in toks if t not in VI_STOPWORDS]
    return toks


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def lexical_overlap(claim: str, evidence: str) -> float:
    """Share of the claim's content syllable uni+bigrams found in the evidence (0..1)."""
    c = syllables(claim)
    e = syllables(evidence)
    if not c:
        return 0.0
    cu = {(t,) for t in c if t not in VI_STOPWORDS} or {(t,) for t in c}
    cb = ngrams(c, 2)
    eu = {(t,) for t in e}
    eb = ngrams(e, 2)
    uni = len(cu & eu) / len(cu)
    if not cb:
        return uni
    bi = len(cb & eb) / len(cb)
    return 0.5 * uni + 0.5 * bi


def fts_query(text: str, max_terms: int = 24) -> str:
    """Build a safe FTS5 MATCH expression: OR over content syllables + adjacent bigram phrases."""
    toks = syllables(text)
    content = [t for t in toks if t not in VI_STOPWORDS] or toks
    terms: list[str] = []
    seen = set()
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if a in VI_STOPWORDS and b in VI_STOPWORDS:
            continue
        p = f'"{a} {b}"'
        if p not in seen:
            seen.add(p)
            terms.append(p)
    for t in content:
        p = f'"{t}"'
        if p not in seen:
            seen.add(p)
            terms.append(p)
    terms = terms[:max_terms]
    return " OR ".join(terms)
