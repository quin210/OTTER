"""Span alignment between two editions of the same corpus (knowledge-maintenance add-on).

Given the span set of the OLD edition (the one the ontology was built on) and of a NEW edition, decide for
every old span what happened to it. Span ids are NOT trusted (a new edition re-numbers pages/paragraphs);
the alignment key is (book, heading path, text) and falls back to fuzzy matching:

  1. identical      same book + heading path + exact text_sha1 (paired in reading order inside a group)
  2. identical_norm same, after NFC + whitespace normalisation (typesetting-only change)
  3. moved          identical text, different heading path, found in the same book
  4. modified       fuzzy 1-1 match: syllable uni+bigram Jaccard, confirmed by bge-m3 cosine when borderline
  5. split_merged   old text re-flowed into several new spans (or several old spans merged into one)
  6. ambiguous      plausible but not decisive candidates: NEVER forced (consumers must treat as unresolved)
  7. gone           no plausible counterpart

Thresholds are fixed a priori (AlignConfig) and were not tuned on the simulation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from otter.storage.textutil import nfc, syllables

SAME_STATUSES = ("identical", "identical_norm", "moved")
CHANGED_STATUSES = ("modified", "split_merged")


@dataclass
class SpanRec:
    span_id: str
    book_id: str
    text: str
    heading_path: tuple = ()
    order: int = 0                      # reading-order index inside the book
    sha1: str = ""
    nsha1: str = ""
    raw: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.sha1:
            self.sha1 = hashlib.sha1(self.text.encode("utf-8")).hexdigest()
        if not self.nsha1:
            self.nsha1 = hashlib.sha1(" ".join(nfc(self.text).split()).casefold().encode("utf-8")).hexdigest()


@dataclass
class AlignConfig:
    jac_accept: float = 0.55            # lexical Jaccard sufficient (with a margin) for a 1-1 match
    jac_floor: float = 0.25             # below this lexically: no candidate unless dense says paraphrase
    margin: float = 0.12                # best - second best (Jaccard) required to accept
    cos_accept: float = 0.90            # dense confirmation of a borderline lexical match
    cos_paraphrase: float = 0.93        # dense-only evidence => ambiguous (never an automatic match)
    min_syllables_fuzzy: int = 4        # shorter old spans: exact matching only (+ strict same-heading unigram rule)
    short_jac: float = 0.75
    split_cover: float = 0.80
    split_precision: float = 0.50
    max_split_parts: int = 4
    cand_top: int = 5


@dataclass
class AlignRow:
    old_id: str
    new_ids: list[str]
    status: str
    method: str = ""
    score: float = 0.0                  # Jaccard (lexical) of the accepted/best candidate
    cosine: Optional[float] = None
    note: str = ""

    def as_dict(self) -> dict:
        return {"old_span_id": self.old_id, "new_span_ids": self.new_ids, "status": self.status,
                "method": self.method, "jaccard": round(self.score, 4),
                "cosine": None if self.cosine is None else round(self.cosine, 4), "note": self.note}


def span_category(status: str) -> str:
    if status in SAME_STATUSES:
        return "same"
    if status in CHANGED_STATUSES:
        return "changed"
    if status == "gone":
        return "gone"
    return "unresolved"


# ---------------------------------------------------------------------------------------------- loading
def _hp_key(hp) -> tuple:
    return tuple(" ".join(nfc(str(x)).split()).casefold() for x in (hp or []))


def _rec_from_dict(d: dict, order: int) -> SpanRec:
    hp = d.get("heading_path")
    if hp is None and d.get("heading_path_json"):
        try:
            hp = json.loads(d["heading_path_json"])
        except (TypeError, json.JSONDecodeError):
            hp = []
    return SpanRec(span_id=d["span_id"], book_id=d["book_id"], text=d.get("text") or "", heading_path=_hp_key(hp),
                   order=order, raw=d)


def _finish(rows: list[dict]) -> list[SpanRec]:
    by_book: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r.get("text") or "").strip():
            by_book[r["book_id"]].append(r)
    out: list[SpanRec] = []
    for bid, rs in by_book.items():
        rs.sort(key=lambda r: (r.get("pdf_page_index") if r.get("pdf_page_index") is not None
                               else r.get("source_page_index") or 0, r.get("start_offset") or 0, r["span_id"]))
        out += [_rec_from_dict(r, i) for i, r in enumerate(rs)]
    return out


def load_span_records(src: str | Path, *, book_prefix: Optional[str] = None,
                      book_ids: Optional[Iterable[str]] = None) -> list[SpanRec]:
    """Load spans from a spans.jsonl file, a directory (`*/spans.jsonl`), or a sqlite DB with a `spans` table."""
    p = Path(src)
    keep = set(book_ids) if book_ids else None

    def ok(bid: str) -> bool:
        return (not book_prefix or bid.startswith(book_prefix)) and (keep is None or bid in keep)

    rows: list[dict] = []
    if p.is_dir():
        files = sorted(p.glob("*/spans.jsonl")) or sorted(p.glob("spans.jsonl"))
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    if ok(d["book_id"]):
                        rows.append(d)
    elif p.suffix in (".jsonl", ".json"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if ok(d["book_id"]):
                    rows.append(d)
    else:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        for r in con.execute("SELECT * FROM spans"):
            d = dict(r)
            if ok(d["book_id"]) and (d.get("text") or "").strip() != "":
                rows.append(d)
        con.close()
    return _finish(rows)


# ---------------------------------------------------------------------------------------------- features
def _grams(text: str) -> frozenset:
    s = syllables(text)
    g = {(t,) for t in s}
    g |= {(s[i], s[i + 1]) for i in range(len(s) - 1)}
    return frozenset(g)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _bigrams(g: frozenset) -> frozenset:
    return frozenset(x for x in g if len(x) == 2)


# ---------------------------------------------------------------------------------------------- core
def align_spans(old: list[SpanRec], new: list[SpanRec], *, embedder=None,
                old_vecs: Optional[dict[str, np.ndarray]] = None, cfg: Optional[AlignConfig] = None
                ) -> tuple[dict[str, AlignRow], list[str]]:
    """Returns ({old_span_id: AlignRow}, [new span ids with no old counterpart (added content)])."""
    cfg = cfg or AlignConfig()
    rows: dict[str, AlignRow] = {}
    claimed: set[str] = set()                      # new ids matched 1-1 (identical/moved/modified)
    new_by_book: dict[str, list[SpanRec]] = defaultdict(list)
    for s in new:
        new_by_book[s.book_id].append(s)
    old_by_book: dict[str, list[SpanRec]] = defaultdict(list)
    for s in old:
        old_by_book[s.book_id].append(s)

    for bid, olds in old_by_book.items():
        news = new_by_book.get(bid, [])
        _align_book(olds, news, rows, claimed, cfg, embedder, old_vecs)
    # old spans in books absent from the new edition
    for s in old:
        if s.span_id not in rows:
            rows[s.span_id] = AlignRow(s.span_id, [], "gone", "no_book", note="book_absent_in_new_edition")
    used = set(claimed)
    for r in rows.values():
        if r.status == "split_merged":
            used.update(r.new_ids)
    added = [s.span_id for s in new if s.span_id not in used]
    return rows, added


def _pair_group(olds: list[SpanRec], news: list[SpanRec], status: str, method: str, rows, claimed, note_dup=True):
    """Pair two lists of spans sharing a content key, in reading order."""
    olds = sorted(olds, key=lambda s: s.order)
    news = sorted(news, key=lambda s: s.order)
    n = min(len(olds), len(news))
    for i in range(n):
        rows[olds[i].span_id] = AlignRow(olds[i].span_id, [news[i].span_id], status, method, 1.0)
        claimed.add(news[i].span_id)
    # surplus old copies: the text still exists in the new edition (content-addressed) -> same knowledge
    for o in olds[n:]:
        tgt = news[-1] if news else None
        if tgt is not None:
            rows[o.span_id] = AlignRow(o.span_id, [tgt.span_id], status, method + "_dup_count_changed", 1.0,
                                       note="duplicate text; copy count differs")


def _align_book(olds: list[SpanRec], news: list[SpanRec], rows, claimed, cfg: AlignConfig, embedder, old_vecs):
    # 1. exact (book, heading path, sha1)
    for field_, status, method in (("sha1", "identical", "exact"), ("nsha1", "identical_norm", "exact_norm")):
        go: dict[tuple, list[SpanRec]] = defaultdict(list)
        gn: dict[tuple, list[SpanRec]] = defaultdict(list)
        for o in olds:
            if o.span_id not in rows:
                go[(o.heading_path, getattr(o, field_))].append(o)
        for n in news:
            if n.span_id not in claimed:
                gn[(n.heading_path, getattr(n, field_))].append(n)
        for k, os_ in go.items():
            if k in gn:
                _pair_group(os_, gn[k], status, method, rows, claimed)
    # 3. moved: identical text, other heading path (same book)
    gn2: dict[str, list[SpanRec]] = defaultdict(list)
    for n in news:
        if n.span_id not in claimed:
            gn2[n.nsha1].append(n)
    go2: dict[str, list[SpanRec]] = defaultdict(list)
    for o in olds:
        if o.span_id not in rows:
            go2[o.nsha1].append(o)
    for k, os_ in go2.items():
        if k in gn2:
            cands = gn2[k]
            _pair_group(os_, cands, "moved", "exact_other_heading", rows, claimed)
            if len(cands) > 1 or len(os_) > 1:
                for o in os_:
                    if o.span_id in rows:
                        rows[o.span_id].note = (rows[o.span_id].note + " moved_ambiguous").strip()
    rem_old = [o for o in olds if o.span_id not in rows]
    if not rem_old:
        return
    rem_new = [n for n in news if n.span_id not in claimed]
    if not rem_new:
        for o in rem_old:
            rows[o.span_id] = AlignRow(o.span_id, [], "gone", "no_candidates")
        return
    _fuzzy(rem_old, rem_new, rows, claimed, cfg, embedder, old_vecs)


def _fuzzy(rem_old: list[SpanRec], rem_new: list[SpanRec], rows, claimed, cfg: AlignConfig, embedder, old_vecs):
    g_old = {o.span_id: _grams(o.text) for o in rem_old}
    g_new = {n.span_id: _grams(n.text) for n in rem_new}
    # inverted index over bigrams (unigrams for very short texts) to avoid all-pairs scoring
    inv: dict[tuple, list[str]] = defaultdict(list)
    for n in rem_new:
        for x in g_new[n.span_id]:
            if len(x) == 2:
                inv[x].append(n.span_id)
    by_id_new = {n.span_id: n for n in rem_new}
    cands: dict[str, list[tuple[float, str]]] = {}
    for o in rem_old:
        go = g_old[o.span_id]
        cnt: dict[str, int] = defaultdict(int)
        for x in go:
            if len(x) == 2:
                for nid in inv.get(x, ()):
                    cnt[nid] += 1
        pool = sorted(cnt, key=lambda k: -cnt[k])[: max(cfg.cand_top * 3, 15)]
        if len(syllables(o.text)) < cfg.min_syllables_fuzzy:
            pool = [n.span_id for n in rem_new if n.heading_path == o.heading_path]      # short: same heading only
        sc = sorted(((jaccard(go, g_new[nid]), nid) for nid in pool), reverse=True)[: cfg.cand_top]
        cands[o.span_id] = sc
    # dense vectors only for the (few) spans that need a decision
    need_dense = [o for o in rem_old if cands[o.span_id] and cands[o.span_id][0][0] < cfg.jac_accept] + \
                 [o for o in rem_old if not cands[o.span_id] or cands[o.span_id][0][0] < cfg.jac_floor]
    vecs_o: dict[str, np.ndarray] = {}
    vecs_n: dict[str, np.ndarray] = {}
    dense_ok = embedder is not None and embedder.available()
    if dense_ok and need_dense:
        want_old = list(dict.fromkeys(o.span_id for o in need_dense))
        miss = [i for i in want_old if not old_vecs or i not in old_vecs]
        if miss:
            txt = {o.span_id: o.text for o in rem_old}
            m = embedder.embed([txt[i] for i in miss])
            for i, v in zip(miss, m):
                vecs_o[i] = _unit(v)
        for i in want_old:
            if old_vecs and i in old_vecs:
                vecs_o[i] = _unit(old_vecs[i])
        want_new = sorted({nid for o in need_dense for _, nid in cands[o.span_id]} |
                          {n.span_id for o in need_dense for n in rem_new if n.heading_path == o.heading_path})
        if want_new:
            m = embedder.embed([by_id_new[i].text for i in want_new])
            for i, v in zip(want_new, m):
                vecs_n[i] = _unit(v)

    tentative: dict[str, tuple[str, float, Optional[float], str]] = {}
    for o in rem_old:
        sc = cands[o.span_id]
        short = len(syllables(o.text)) < cfg.min_syllables_fuzzy
        j1, n1 = sc[0] if sc else (0.0, None)
        j2 = sc[1][0] if len(sc) > 1 else 0.0
        cos1: Optional[float] = None
        if dense_ok and o.span_id in vecs_o and n1 in vecs_n:
            cos1 = float(vecs_o[o.span_id] @ vecs_n[n1])
        if short:
            if n1 is not None and j1 >= cfg.short_jac and j1 - j2 >= cfg.margin:
                tentative[o.span_id] = (n1, j1, cos1, "fuzzy_short")
            elif n1 is not None and j1 >= 0.3:
                rows[o.span_id] = AlignRow(o.span_id, [], "ambiguous", "short_text", j1, cos1,
                                           "short text, weak lexical evidence")
            else:
                rows[o.span_id] = AlignRow(o.span_id, [], "gone", "no_candidates", j1)
            continue
        if n1 is not None and j1 >= cfg.jac_accept and j1 - j2 >= cfg.margin:
            tentative[o.span_id] = (n1, j1, cos1, "fuzzy_jaccard")
        elif n1 is not None and j1 >= cfg.jac_floor:
            if cos1 is not None and cos1 >= cfg.cos_accept and j1 - j2 >= cfg.margin / 2:
                tentative[o.span_id] = (n1, j1, cos1, "fuzzy_jaccard+dense")
            else:
                done = _try_split(o, rem_new, g_old, g_new, cfg)
                if done:
                    rows[o.span_id] = AlignRow(o.span_id, done[0], "split_merged", done[1], j1, cos1, done[2])
                else:
                    rows[o.span_id] = AlignRow(o.span_id, [], "ambiguous", "borderline", j1, cos1,
                                               "borderline lexical/dense evidence; not forced")
        else:
            done = _try_split(o, rem_new, g_old, g_new, cfg)
            if done:
                rows[o.span_id] = AlignRow(o.span_id, done[0], "split_merged", done[1], j1, cos1, done[2])
                continue
            best_cos = None
            if dense_ok and o.span_id in vecs_o:
                same = [vecs_n[n.span_id] for n in rem_new if n.heading_path == o.heading_path and n.span_id in vecs_n]
                if same:
                    best_cos = float(max(vecs_o[o.span_id] @ v for v in same))
            if best_cos is not None and best_cos >= cfg.cos_paraphrase:
                rows[o.span_id] = AlignRow(o.span_id, [], "ambiguous", "dense_only", j1, best_cos,
                                           "semantically close but lexically different (paraphrase?)")
            else:
                rows[o.span_id] = AlignRow(o.span_id, [], "gone", "no_candidates", j1, best_cos)
    # one-to-one: a new span claimed by several old spans is only kept for a decisive best claimant
    by_new: dict[str, list[str]] = defaultdict(list)
    for oid, (nid, *_rest) in tentative.items():
        by_new[nid].append(oid)
    for nid, oids in by_new.items():
        oids.sort(key=lambda i: -tentative[i][1])
        win = oids[0]
        if len(oids) > 1 and tentative[win][1] - tentative[oids[1]][1] < cfg.margin:
            losers = oids
        else:
            losers = oids[1:]
            _accept(win, tentative[win], rows, claimed)
        for oid in losers:
            _, j, c, _m = tentative[oid]
            rows[oid] = AlignRow(oid, [], "ambiguous", "contested", j, c, "several old spans compete for one new span")


def _accept(oid: str, t, rows, claimed):
    nid, j, c, method = t
    rows[oid] = AlignRow(oid, [nid], "modified", method, j, c)
    claimed.add(nid)


def _try_split(o: SpanRec, rem_new: list[SpanRec], g_old, g_new, cfg: AlignConfig):
    """Old span re-flowed into several consecutive new spans (or contained in one larger new span)."""
    go = _bigrams(g_old[o.span_id]) or g_old[o.span_id]
    if not go:
        return None
    scored = []
    for n in rem_new:
        gn = _bigrams(g_new[n.span_id]) or g_new[n.span_id]
        inter = len(go & gn)
        if inter:
            scored.append((inter / len(go), inter / max(len(gn), 1), n))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return None
    # merge: one (larger) new span contains almost all of the old span
    c, p, n = scored[0]
    if c >= cfg.split_cover and (len(scored) == 1 or scored[1][0] < 0.3):
        return [n.span_id], "merged_into_one", f"containment={c:.2f}"
    parts = [(cc, pp, nn) for cc, pp, nn in scored[: cfg.max_split_parts] if pp >= cfg.split_precision and cc >= 0.1]
    if len(parts) >= 2:
        union = set()
        for _, _, nn in parts:
            union |= (_bigrams(g_new[nn.span_id]) or g_new[nn.span_id]) & go
        cover = len(union) / len(go)
        if cover >= cfg.split_cover:
            parts.sort(key=lambda x: x[2].order)
            orders = [x[2].order for x in parts]
            if max(orders) - min(orders) <= len(parts) + 1:         # consecutive in reading order
                return [x[2].span_id for x in parts], "split_into_parts", f"cover={cover:.2f}"
    return None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def summarize(rows: dict[str, AlignRow]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows.values():
        out[r.status] += 1
    return dict(out)
