"""Extend the span retrieval indexes with NEW spans without rebuilding the existing ones (knowledge-ingest add-on).

Dense (bge-m3): `ingest-books` already writes `embeddings(kind='span')` rows for the new books into the new ontology file, and
the old file's span vectors are copied unchanged by the sqlite backup. Nothing else is needed: `SqliteOntologyStore._dense_index`
loads the vectors of the file it opens. For an external vector matrix (e.g. a text-RAG cache) use `append_dense`.

Sparse (BM25): `rank_bm25.BM25Okapi` has no `add`. `IncrementalBM25` keeps the same statistics (document frequency, lengths,
average length) and recomputes the idf table lazily after an `add`, which is O(vocabulary), not O(corpus). Scores are identical to
a full `BM25Okapi` rebuild over old+new documents (tested to float precision), because BM25Okapi's idf depends only on those
counts. The sqlite FTS5 table `spans_fts` is extended by the same ingest (INSERT of the new rows, no rebuild).

This module is not imported by the frozen method code; callers opt in.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Optional, Sequence

import numpy as np


class IncrementalBM25:
    """BM25Okapi-compatible scorer (k1=1.5, b=0.75, epsilon=0.25) with `add(docs)`."""

    def __init__(self, docs: Iterable[Sequence[str]] = (), k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)       # term -> {doc index: tf}
        self._len_sum = 0
        self._idf: Optional[dict[str, float]] = None
        self.add(docs)

    @property
    def corpus_size(self) -> int:
        return len(self.doc_len)

    def add(self, docs: Iterable[Sequence[str]]) -> int:
        n0 = len(self.doc_len)
        for doc in docs:
            i = len(self.doc_len)
            self.doc_len.append(len(doc))
            self._len_sum += len(doc)
            for term, tf in Counter(doc).items():
                self.postings[term][i] = tf
        self._idf = None
        return len(self.doc_len) - n0

    def _calc_idf(self) -> dict[str, float]:
        n = self.corpus_size
        idf, total, neg = {}, 0.0, []
        for term, post in self.postings.items():
            v = math.log(n - len(post) + 0.5) - math.log(len(post) + 0.5)
            idf[term] = v
            total += v
            if v < 0:
                neg.append(term)
        avg = total / len(idf) if idf else 0.0
        for term in neg:
            idf[term] = self.epsilon * avg
        return idf

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        if self._idf is None:
            self._idf = self._calc_idf()
        n = self.corpus_size
        score = np.zeros(n)
        if n == 0:
            return score
        avgdl = self._len_sum / n
        dl = np.asarray(self.doc_len, dtype=float)
        for q in query:
            post = self.postings.get(q)
            if not post:
                continue
            ids = np.fromiter(post.keys(), dtype=int, count=len(post))
            tf = np.fromiter(post.values(), dtype=float, count=len(post))
            score[ids] += (self._idf.get(q) or 0) * (tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl[ids] / avgdl)))
        return score


def append_dense(matrix: np.ndarray, new_vectors: np.ndarray) -> np.ndarray:
    """Row-append L2-normalised vectors to an existing (n, d) matrix (no re-embedding of old rows)."""
    new_vectors = np.asarray(new_vectors, dtype=np.float32)
    if matrix.size == 0:
        return new_vectors
    if matrix.shape[1] != new_vectors.shape[1]:
        raise ValueError(f"dimension mismatch {matrix.shape[1]} vs {new_vectors.shape[1]}")
    return np.vstack([matrix, new_vectors])
