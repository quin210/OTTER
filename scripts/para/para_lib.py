"""Shared helpers for ViEduHop-Para (post hoc; see docs/VIEDUHOP_PARA.md). No system output is used anywhere in this module."""
from __future__ import annotations
import json, re, sys, threading, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.storage.textutil import VI_STOPWORDS, nfc, syllables  # noqa: E402

LETTERS = "ABCD"
_CAP = re.compile(r"^[A-ZÀ-ỹ]")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def content_syll(text: str) -> list[str]:
    """Content syllables: lower-cased NFC word tokens without VI_STOPWORDS and without pure-digit tokens (digits are protected, not paraphrasable)."""
    return [t for t in syllables(text) if t not in VI_STOPWORDS and not t.isdigit()]


def mcq_text(question: str, options: dict) -> str:
    return question + " " + " ".join(options[k] for k in LETTERS)


def overlap(text: str, gold_text: str) -> float:
    """Share of DISTINCT content syllables of `text` that occur among the syllables of `gold_text` (the union of the gold spans)."""
    c = set(content_syll(text))
    if not c:
        return 0.0
    g = set(syllables(gold_text))
    return len(c & g) / len(c)


def entity_mask(text: str) -> list[tuple[str, bool]]:
    """(token, in_named_entity_or_digit) for every raw token; a named entity = run of >=1 capitalised tokens that is not the first token of a sentence, or any token with a digit."""
    toks = re.findall(r"\w+", nfc(text), flags=re.UNICODE)
    out = []
    for i, t in enumerate(toks):
        nxt_cap = i + 1 < len(toks) and bool(_CAP.match(toks[i + 1]))
        out.append((t.casefold(), bool(any(ch.isdigit() for ch in t)) or (bool(_CAP.match(t)) and (i > 0 or nxt_cap))))
    return out


def content_trigrams(text: str):
    """Trigrams over the content-syllable sequence (stopwords/digits removed). Returns list of (trigram, exempt) where exempt = all three tokens are protected
    (named entity run or digit token) in the paraphrase, i.e. unavoidable copies of entities/numbers."""
    seq = [(t, prot) for t, prot in entity_mask(text) if t not in VI_STOPWORDS and not t.isdigit()]
    return [(tuple(t for t, _ in seq[i:i + 3]), all(p for _, p in seq[i:i + 3])) for i in range(len(seq) - 2)]


def copied_trigrams(text: str, gold_text: str) -> list[tuple]:
    g = [t for t in syllables(gold_text) if t not in VI_STOPWORDS and not t.isdigit()]
    gset = {tuple(g[i:i + 3]) for i in range(len(g) - 2)}
    return [tg for tg, exempt in content_trigrams(text) if tg in gset and not exempt]


_corpus = None
def gold_text(span_ids: list[str]) -> str:
    global _corpus
    if _corpus is None:
        from otter.evaluation.corpus import SpanCorpus, core_book_filter
        _corpus = SpanCorpus.load(book_filter=core_book_filter)
    return "\n".join(_corpus.get(s).text for s in span_ids if _corpus.get(s))
