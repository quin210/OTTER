"""bge-m3 embeddings through the OpenAI-compatible endpoint with a persistent sqlite cache.

`MockEmbedder` (hashing trick over syllable n-grams) is used offline / in tests.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from otter.evaluation.text import bigram_tokens, sha1

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "runs" / "cache" / "embeddings.sqlite3"


class _Cache:
    def __init__(self, path: Optional[Path], model: str):
        self.path = Path(path) if path else None
        self.model = model
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as c:
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("CREATE TABLE IF NOT EXISTS emb(key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")

    def key(self, text: str) -> str:
        return sha1(self.model + "\x1f" + text)

    def get_many(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        if not self.path or not texts:
            return {}
        out: dict[str, np.ndarray] = {}
        keys = [self.key(t) for t in texts]
        with self._lock, sqlite3.connect(self.path) as c:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
                for k, v in c.execute(q, chunk):
                    out[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        if not self.path or not items:
            return
        with self._lock, sqlite3.connect(self.path) as c:
            c.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?)",
                          [(k, int(v.size), v.astype(np.float32).tobytes()) for k, v in items.items()])


class Embedder:
    model: str = ""

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def __init__(self, cache_path: Optional[Path] = DEFAULT_CACHE, batch_size: int = 32):
        self.cache = _Cache(cache_path, self.model)
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised float32 matrix (len(texts), dim)."""
        texts = [t if t.strip() else " " for t in texts]
        cached = self.cache.get_many(texts)
        missing = [t for t in dict.fromkeys(texts) if self.cache.key(t) not in cached]
        new: dict[str, np.ndarray] = {}
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            vecs = self._embed_raw(batch)
            for t, v in zip(batch, vecs):
                n = np.linalg.norm(v)
                new[self.cache.key(t)] = (v / n if n > 0 else v).astype(np.float32)
            self.cache.put_many({k: new[k] for k in [self.cache.key(t) for t in batch]})
        allv = {**cached, **new}
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([allv[self.cache.key(t)] for t in texts])


class OpenAIEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, cache_path: Optional[Path] = DEFAULT_CACHE,
                 batch_size: int = 32, max_chars: int = 6000):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
        self.max_chars = max_chars
        super().__init__(cache_path, batch_size)

    def _embed_raw(self, texts):
        r = self.client.embeddings.create(model=self.model, input=[t[: self.max_chars] for t in texts])
        return np.asarray([d.embedding for d in r.data], dtype=np.float32)


class MockEmbedder(Embedder):
    """Deterministic hashing-trick embedder: lexical overlap -> cosine similarity."""
    model = "mock-hash-embedder"

    def __init__(self, dim: int = 256, cache_path: Optional[Path] = None):
        self.dim = dim
        super().__init__(cache_path)

    def _embed_raw(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in bigram_tokens(t):
                h = int(sha1(tok)[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return out


def get_embedder(config: Optional[dict] = None, cache_path: Optional[Path] = DEFAULT_CACHE) -> Embedder:
    if os.environ.get("OTTER_LLM_MOCK") == "1" or os.environ.get("OTTER_EMB_MOCK") == "1":
        return MockEmbedder()
    from otter.llm.interface import load_config
    cfg = (config or load_config())["embedding"]
    return OpenAIEmbedder(cfg["base_url"], cfg["model"], cache_path=cache_path)
