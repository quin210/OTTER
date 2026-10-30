"""OTTER-Hard seed sampling: tuples of 2-3 textbook spans that are *linked* but not redundant.

Mirrors the ViHERMES construction (semantic clustering for coverage, tuple sampling under
cross-unit and overlap constraints), adapted to textbooks:

* strategies: `same_lesson` (two units of one lesson), `cross_lesson` (same book, different lessons),
  `cross_book` (same subject, different grade/book), and 3-span `chain`s (A-B, B-C linked);
* links: shared ontology concepts (WS2 DB, if built) else bge-m3 neighbours + lexical concept overlap
  (shared high-IDF syllable bigrams);
* exclusions: exercise/question/activity spans, objectives, very short/long or OCR-garbage spans
  (gold evidence must be expository text, not textbook exercises -> no exercise leakage);
* diversity: anchors are spread round-robin over (spherical k-means or lesson) clusters, each span is
  used in at most `max_span_uses` seeds and two seeds share at most `tau` of their spans;
* a target hop type is assigned per seed from subject quotas and textual cues.
"""
from __future__ import annotations

import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.text import short_hash, tokens

SEEDS_VERSION = "otter_hard_seeds_v1"
HOP_TYPES = ("bridge", "comparison", "conditional_rule", "causal_chain", "calculation", "temporal")

# relative quotas of hop types per subject (normalised inside the sampler)
SUBJECT_HOPS: dict[str, dict[str, float]] = {
    "physics": {"calculation": 3, "conditional_rule": 2, "causal_chain": 2, "bridge": 2, "comparison": 1},
    "chemistry": {"calculation": 3, "conditional_rule": 2, "causal_chain": 2, "comparison": 2, "bridge": 1},
    "biology": {"causal_chain": 3, "conditional_rule": 2, "comparison": 2, "bridge": 2},
    "history": {"temporal": 3, "causal_chain": 3, "comparison": 2, "bridge": 2},
    "geography": {"causal_chain": 3, "comparison": 2, "bridge": 2, "conditional_rule": 1, "calculation": 1},
    "literature": {"comparison": 3, "bridge": 3, "causal_chain": 1, "temporal": 1},
}
STRATEGY_MIX = {"same_lesson": 0.35, "cross_lesson": 0.30, "cross_book": 0.20, "chain": 0.15}

_CUES = {
    "calculation": re.compile(r"(\d+[.,]?\d*\s*(%|kg|g|m|cm|km|s|h|N|J|W|V|A|Ω|mol|M|K|°C|Pa|Hz|lít|L)\b)|=|\$[^$]+\$|"
                              r"công thức|tính theo|bằng tích|bằng tỉ số", re.I),
    "temporal": re.compile(r"\b(1\d{3}|20\d{2})\b|thế kỉ|thế kỷ|năm \d|giai đoạn|thời kì|thời kỳ|trước công nguyên|TCN", re.I),
    "conditional_rule": re.compile(r"\b(nếu|khi|trong điều kiện|chỉ khi|với điều kiện|trường hợp)\b", re.I),
    "causal_chain": re.compile(r"(dẫn đến|làm cho|nguyên nhân|hệ quả|kết quả là|do đó|vì vậy|nên|gây ra|ảnh hưởng)", re.I),
    "comparison": re.compile(r"(khác với|giống|so với|hơn|trong khi|ngược lại|khác nhau)", re.I),
}
_OBJECTIVE = re.compile(r"(học xong bài này|sau bài học này|yêu cầu cần đạt|mục tiêu|em sẽ)", re.I)
_LAB = re.compile(r"(DỤNG CỤ|CHUẨN BỊ|TIẾN HÀNH|BÁO CÁO|MẪU BÁO CÁO|THIẾT BỊ THÍ NGHIỆM)", re.I)
_TOC = re.compile(r"(\.{5,}|…{3,})")
_STOP = set("và của là các có được trong cho với những một này đó khi thì để từ theo như đến vào ra không cũng "
            "đã sẽ đang bị do vì nên hay hoặc mà rất nhiều ít trên dưới sau trước giữa lại còn vẫn chỉ đều "
            "bài em hãy ta người hình ở".split())


@dataclass
class Seed:
    seed_id: str
    strategy: str
    span_ids: list[str]
    domain: str
    grade: Optional[int]
    grades: list[Optional[int]]
    book_ids: list[str]
    lesson_keys: list[str]
    target_hop_type: str
    compatible_hop_types: list[str]
    link: dict[str, Any] = field(default_factory=dict)
    version: str = SEEDS_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ------------------------------------------------------------------- eligibility
def eligible(s: Span, *, min_chars: int = 150, max_chars: int = 2200) -> bool:
    t = s.text.strip()
    if not (min_chars <= len(t) <= max_chars) or s.is_exercise() or not (s.section_id or s.heading_path):
        return False
    if s.pedagogical_type in ("objective", "figure_caption", "question", "exercise", "activity"):
        return False
    if _OBJECTIVE.search(t[:120]) or _TOC.search(t):
        return False
    if t.startswith("[") or t.startswith("!["):     # VLM description of a figure/map, not textbook prose
        return False
    if _LAB.search(" / ".join(s.heading_path[-1:])) or re.match(r"^\s*[-–]?\s*Bước \d", t):
        return False                                # lab equipment lists / procedure steps
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.55:          # tables of numbers / OCR garbage
        return False
    toks = tokens(t)
    return len(toks) >= 25 and len(set(toks)) / len(toks) > 0.3


def hop_cues(text: str) -> set[str]:
    return {h for h, rx in _CUES.items() if rx.search(text)} | {"bridge"}


# ------------------------------------------------------------------- linking
class TermIndex:
    """High-IDF syllable bigrams per subject approximate shared concepts when no ontology exists."""

    def __init__(self, spans: list[Span]):
        self.bigrams: dict[str, set[str]] = {}
        df: dict[str, Counter] = defaultdict(Counter)
        self.n_by_dom: Counter = Counter()
        for s in spans:
            t = [w for w in tokens(s.text)]
            bg = {f"{a} {b}" for a, b in zip(t, t[1:]) if not (a in _STOP or b in _STOP) and not a.isdigit()
                  and not b.isdigit() and len(a) + len(b) > 3}
            self.bigrams[s.span_id] = bg
            df[s.domain].update(bg)
            self.n_by_dom[s.domain] += 1
        self.df = df

    def idf(self, dom: str, term: str) -> float:
        return math.log((1 + self.n_by_dom[dom]) / (1 + self.df[dom][term]))

    def shared(self, a: Span, b: Span, *, max_df_frac: float = 0.04) -> list[tuple[str, float]]:
        n = self.n_by_dom[a.domain]
        lim = max(3, int(max_df_frac * n))
        common = self.bigrams.get(a.span_id, set()) & self.bigrams.get(b.span_id, set())
        out = [(t, self.idf(a.domain, t)) for t in common if 2 <= self.df[a.domain][t] <= lim]
        return sorted(out, key=lambda x: -x[1])


def load_span_concepts(ontology_db: Optional[Path]) -> dict[str, set[str]]:
    """span_id -> concept ids from WS2's ontology DB (evidence_links or concepts.json). Empty if absent."""
    out: dict[str, set[str]] = defaultdict(set)
    if not ontology_db or not Path(ontology_db).exists():
        return out
    try:
        with sqlite3.connect(f"file:{ontology_db}?mode=ro", uri=True) as c:
            tabs = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "evidence_links" in tabs:
                for kind, iid, sid in c.execute("SELECT item_kind, item_id, span_id FROM evidence_links"):
                    if kind == "concept":
                        out[sid].add(iid)
            if not out and "concepts" in tabs:
                for cid, js in c.execute("SELECT concept_id, json FROM concepts"):
                    for sid in json.loads(js).get("source_span_ids", []):
                        out[sid].add(cid)
            # facts/rules link concepts to further spans (concept mentioned in an assertion)
            if "fact_concepts" in tabs and "evidence_links" in tabs:
                q = ("SELECT fc.concept_id, el.span_id FROM fact_concepts fc JOIN evidence_links el "
                     "ON el.item_kind='fact' AND el.item_id=fc.assertion_id")
                for cid, sid in c.execute(q):
                    out[sid].add(cid)
    except sqlite3.Error:
        return defaultdict(set)
    return out


def spherical_kmeans(x: np.ndarray, k: int, *, iters: int = 25, seed: int = 0) -> np.ndarray:
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(n, size=k, replace=False)].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ cent.T, axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = x[lab == j]
            if len(m):
                v = m.mean(axis=0)
                cent[j] = v / (np.linalg.norm(v) or 1)
    return lab


# ------------------------------------------------------------------- sampler
class SeedSampler:
    def __init__(self, corpus: SpanCorpus, *, span_concepts: Optional[dict[str, set[str]]] = None,
                 seed: int = 13, max_span_uses: int = 2, tau: float = 0.5, min_link: float = 6.0, min_terms: int = 2,
                 dense_lo: float = 0.5, dense_hi: float = 0.93, eligible_kw: Optional[dict] = None):
        self.corpus = corpus
        self.rng = random.Random(seed)
        self.seed = seed
        self.spans = [s for s in corpus.spans if s.domain and eligible(s, **(eligible_kw or {}))]
        self.idx = {s.span_id: i for i, s in enumerate(self.spans)}
        self.terms = TermIndex(self.spans)
        self.concepts = span_concepts or {}
        self.max_span_uses, self.tau, self.min_link, self.min_terms = max_span_uses, tau, min_link, min_terms
        self.dense_lo, self.dense_hi = dense_lo, dense_hi
        self.dense: Optional[np.ndarray] = None
        full = corpus.dense_matrix() if corpus._embedder is not None else None
        if full is not None and self.spans:
            pos = {s.span_id: i for i, s in enumerate(corpus.spans)}
            self.dense = full[[pos[s.span_id] for s in self.spans]]
        self.uses: Counter = Counter()
        self.taken: list[frozenset] = []
        self.order_in_book = {}
        for b, ss in self._group(lambda s: s.book_id).items():
            for k, s in enumerate(sorted(ss, key=lambda s: (s.pdf_page_index or 0, s.span_id))):
                self.order_in_book[s.span_id] = k

    def _group(self, key) -> dict[Any, list[Span]]:
        g: dict[Any, list[Span]] = defaultdict(list)
        for s in self.spans:
            g[key(s)].append(s)
        return g

    # -- link scoring
    def dense_sim(self, a: Span, b: Span) -> Optional[float]:
        if self.dense is None:
            return None
        return float(self.dense[self.idx[a.span_id]] @ self.dense[self.idx[b.span_id]])

    def link(self, a: Span, b: Span) -> Optional[dict[str, Any]]:
        if a.span_id == b.span_id:
            return None
        if a.book_id == b.book_id and abs(self.order_in_book[a.span_id] - self.order_in_book[b.span_id]) <= 1:
            return None                        # adjacent paragraphs: usually one continuous statement
        sim = self.dense_sim(a, b)
        if sim is not None and not (self.dense_lo <= sim <= self.dense_hi):
            return None
        shared_c = sorted(self.concepts.get(a.span_id, set()) & self.concepts.get(b.span_id, set()))
        terms = self.terms.shared(a, b)
        lex = sum(w for _, w in terms[:6])
        score = lex + 3.0 * len(shared_c) + (4.0 * sim if sim is not None else 0.0)
        if not shared_c and (lex < self.min_link or len(terms) < self.min_terms):
            return None
        return {"score": round(score, 3), "shared_concepts": shared_c[:8], "shared_terms": [t for t, _ in terms[:8]],
                "dense_sim": None if sim is None else round(sim, 4),
                "link_source": "ontology" if shared_c else ("dense+lexical" if sim is not None else "lexical")}

    def _ok_tuple(self, ids: list[str]) -> bool:
        if any(self.uses[i] >= self.max_span_uses for i in ids):
            return False
        f = frozenset(ids)
        return all(len(f & t) / len(f) <= self.tau for t in self.taken)

    def _commit(self, ids: list[str]) -> None:
        self.uses.update(ids)
        self.taken.append(frozenset(ids))

    # -- candidate partners for an anchor under a strategy
    def _partners(self, a: Span, strategy: str, pool: list[Span]) -> list[tuple[Span, dict]]:
        if strategy == "same_lesson":
            cand = [b for b in pool if b.book_id == a.book_id and b.lesson_key == a.lesson_key]
        elif strategy == "cross_lesson":
            cand = [b for b in pool if b.book_id == a.book_id and b.lesson_key != a.lesson_key]
        elif strategy == "cross_book":
            cand = [b for b in pool if b.domain == a.domain and b.book_id != a.book_id]
        else:
            cand = [b for b in pool if b.domain == a.domain]
        out = []
        for b in cand:
            lk = self.link(a, b)
            if lk:
                out.append((b, lk))
        out.sort(key=lambda x: -x[1]["score"])
        return out

    def anchors(self, cell: list[Span], n_clusters: int) -> list[Span]:
        """Round-robin over clusters (spherical k-means on bge-m3 if available, else lessons)."""
        if self.dense is not None and len(cell) > n_clusters:
            lab = spherical_kmeans(self.dense[[self.idx[s.span_id] for s in cell]], n_clusters, seed=self.seed)
            groups: dict[Any, list[Span]] = defaultdict(list)
            for s, l in zip(cell, lab):
                groups[int(l)].append(s)
        else:
            groups = defaultdict(list)
            for s in cell:
                groups[s.lesson_key].append(s)
        lists = [self.rng.sample(v, len(v)) for _, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]
        self.rng.shuffle(lists)
        out = []
        while any(lists):
            for l in lists:
                if l:
                    out.append(l.pop())
        return out

    def assign_hop(self, domain: str, texts: list[str], counts: Counter) -> tuple[str, list[str]]:
        quota = SUBJECT_HOPS.get(domain, {h: 1 for h in HOP_TYPES})
        any_cues = set.union(*[hop_cues(t) for t in texts])
        compat = [h for h in quota if h in any_cues]
        if not compat:
            compat = ["bridge"] if "bridge" in quota else list(quota)[:1]
        tot = sum(quota.values())
        n = sum(counts.values()) + 1
        best = max(compat, key=lambda h: (quota[h] / tot - counts[h] / n, quota[h], self.rng.random()))
        return best, compat

    def sample(self, *, per_cell: int = 36, n_clusters: int = 8, mix: Optional[dict[str, float]] = None,
               per_domain: Optional[dict[str, int]] = None) -> list[Seed]:
        mix = mix or STRATEGY_MIX
        cells = self._group(lambda s: (s.domain, s.grade))
        by_domain = self._group(lambda s: s.domain)
        seeds: list[Seed] = []
        for (dom, grade), cell in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
            hop_counts: Counter = Counter()
            quotas = {k: int(round((per_domain or {}).get(dom, per_cell) * v)) for k, v in mix.items()}
            if len({s.book_id for s in by_domain[dom]}) < 2:          # no second book in this subject
                quotas["cross_lesson"] += quotas.pop("cross_book", 0)
            made: Counter = Counter()
            anchors = self.anchors(cell, n_clusters)
            for strategy in sorted(quotas, key=lambda k: -quotas[k]):
                for a in anchors:
                    if made[strategy] >= quotas[strategy]:
                        break
                    if self.uses[a.span_id] >= self.max_span_uses:
                        continue
                    pool = by_domain[dom]
                    if strategy == "chain":
                        sd = self._chain(a, pool)
                    else:
                        sd = self._pair(a, strategy, pool)
                    if sd is None:
                        continue
                    ids, links = sd
                    texts = [self.corpus.get(i).text for i in ids]
                    hop, compat = self.assign_hop(dom, texts, hop_counts)
                    hop_counts[hop] += 1
                    ss = [self.corpus.get(i) for i in ids]
                    grades = [s.grade for s in ss]
                    seeds.append(Seed(seed_id="seed_" + short_hash(ids), strategy=strategy, span_ids=ids, domain=dom,
                                      grade=max((g for g in grades if g is not None), default=None), grades=grades,
                                      book_ids=list(dict.fromkeys(s.book_id for s in ss)),
                                      lesson_keys=list(dict.fromkeys(s.lesson_key for s in ss)),
                                      target_hop_type=hop, compatible_hop_types=compat, link={"pairs": links}))
                    self._commit(ids)
                    made[strategy] += 1
        return seeds

    def _pair(self, a: Span, strategy: str, pool: list[Span]):
        for b, lk in self._partners(a, strategy, pool)[:25]:
            ids = [a.span_id, b.span_id]
            if self._ok_tuple(ids):
                return ids, [lk]
        return None

    def _chain(self, a: Span, pool: list[Span]):
        p1 = [x for x in self._partners(a, "chain_any", pool)[:12] if self.uses[x[0].span_id] < self.max_span_uses]
        for b, l1 in p1:
            for c, l2 in self._partners(b, "chain_any", pool)[:12]:
                if c.span_id in (a.span_id, b.span_id):
                    continue
                sim_ac = self.dense_sim(a, c)
                if sim_ac is not None and sim_ac > self.dense_hi:
                    continue
                ids = [a.span_id, b.span_id, c.span_id]
                if len({s.lesson_key for s in (a, b, c)}) >= 2 and self._ok_tuple(ids):
                    return ids, [l1, l2]
        return None


def write_seeds(seeds: Iterable[Seed], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(s.to_json() + "\n")


def load_seeds(path: Path) -> list[Seed]:
    return [Seed(**json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def seed_stats(seeds: list[Seed]) -> dict[str, Any]:
    return {"n": len(seeds), "by_domain": dict(Counter(s.domain for s in seeds)),
            "by_grade": {str(k): v for k, v in Counter(s.grade for s in seeds).items()},
            "by_strategy": dict(Counter(s.strategy for s in seeds)),
            "by_target_hop": dict(Counter(s.target_hop_type for s in seeds)),
            "by_link_source": dict(Counter(p["link_source"] for s in seeds for p in s.link.get("pairs", []))),
            "n_spans_used": len({i for s in seeds for i in s.span_ids})}
