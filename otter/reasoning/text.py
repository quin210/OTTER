"""Small Vietnamese-aware text helpers used for lexical (heuristic) support checks.

These are *heuristics* only: lexical overlap never establishes source support by
itself (spec section 4); callers must disclose the method they used.
"""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[\w]+", re.UNICODE)

# Frequent Vietnamese function words / question scaffolding. Kept short on purpose.
STOPWORDS = frozenset("""
là của và các những một có được trong cho với khi thì này đó đây kia nào gì ai sao bao nhiêu
không chưa đã đang sẽ cũng như nếu vì nên do bởi để từ đến tới về theo trên dưới ra vào lại
hay hoặc mà thế vậy rằng nhất hơn rất quá đều mỗi mọi cả chỉ còn sau trước giữa ở tại bị
phát biểu sau đây đúng sai nhất câu đáp án mô tả phù hợp khái niệm nội dung nhận định điều
gọi người ta chủ yếu thường luôn có thể cần phải việc sự cái chiếc
the a an of and or to in is are be for on with by as at from that this which what
""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, *, drop_stop: bool = True) -> list[str]:
    toks = _WORD.findall(normalize(text))
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def bigrams(toks: list[str]) -> set[tuple[str, str]]:
    return set(zip(toks, toks[1:]))


def coverage(claim: str, evidence: str) -> float:
    """Fraction of content syllables (and bigrams, weighted) of `claim` found in `evidence`."""
    ct = tokens(claim)
    if not ct:
        return 0.0
    et = set(tokens(evidence, drop_stop=False))
    uni = sum(1 for t in ct if t in et) / len(ct)
    cb = bigrams(tokens(claim, drop_stop=False))
    if not cb:
        return uni
    eb = bigrams(tokens(evidence, drop_stop=False))
    bi = len(cb & eb) / len(cb)
    return round(0.6 * uni + 0.4 * bi, 4)


def contains_quote(quote: str, text: str, *, min_len: int = 8) -> bool:
    """True iff `quote` (normalized) occurs verbatim in `text`, or >=90% of its tokens
    occur as a contiguous-ish run. Used to check LLM-provided quotes against real spans."""
    q, t = normalize(quote), normalize(text)
    if len(q) < min_len:
        return bool(q) and q in t and len(q) >= 3
    if q in t:
        return True
    qt = tokens(q, drop_stop=False)
    tt = tokens(t, drop_stop=False)
    if len(qt) < 3:
        return False
    qb, tb = bigrams(qt), bigrams(tt)
    return len(qb & tb) / max(1, len(qb)) >= 0.9


def truncate(text: str, n: int = 400) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


NEGATIVE_QUESTION = re.compile(
    r"(không\s+đúng|không\s+phải|không\s+chính\s+xác|\bsai\b|ngoại\s+trừ|\btrừ\b|KHÔNG|không\s+thuộc|không\s+có\s+vai\s+trò)",
)


def is_negative_question(question: str) -> bool:
    return bool(NEGATIVE_QUESTION.search(question or ""))
