"""Dense embeddings: bge-m3 via the OpenAI-compatible :8002 server, with fallbacks.

* `Bgem3Embedder` - real server; `available()` probes with a short timeout and caches failures.
* `HashEmbedder` - deterministic offline char-n-gram hashing embedder for tests (NOT for experiments).
* `cosine_topk` - pure-numpy cosine search over L2-normalised float32 matrices.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Optional, Protocol

import numpy as np

from otter.storage.textutil import norm_key


class Embedder(Protocol):
    model: str

    def available(self) -> bool: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Bgem3Embedder:
    def __init__(self, base_url: str, model: str = "BAAI/bge-m3", timeout: float = 60.0, batch_size: int = 64,
                 max_chars: int = 6000):
        self.base_url, self.model, self.timeout = base_url, model, timeout
        self.batch_size, self.max_chars = batch_size, max_chars
        self._client = None
        self._ok: Optional[bool] = None
        self._checked_at = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "Bgem3Embedder":
        from otter.llm.interface import load_config
        cfg = (config or load_config())["embedding"]
        return cls(cfg["base_url"], cfg["model"])

    def _cli(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=self.timeout, max_retries=1)
        return self._client

    def available(self) -> bool:
        if os.environ.get("OTTER_LLM_MOCK") == "1" or os.environ.get("OTTER_NO_DENSE") == "1":
            return False
        with self._lock:
            if self._ok is not None and (self._ok or time.time() - self._checked_at < 60):
                return self._ok
            try:
                import urllib.request
                with urllib.request.urlopen(self.base_url.rstrip("/") + "/models", timeout=2) as r:
                    self._ok = r.status == 200
            except Exception:
                self._ok = False
            self._checked_at = time.time()
            return self._ok

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [(t or " ")[: self.max_chars] for t in texts[i:i + self.batch_size]]
            r = self._cli().embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        if not out:
            return np.zeros((0, 1024), dtype=np.float32)
        return _l2(np.array(out, dtype=np.float32))


class HashEmbedder:
    """Offline deterministic embedder (char 3-grams hashed into `dim` buckets). Test use only."""
    model = "hash-ngram-v1"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f"  {norm_key(t)}  "
            for j in range(len(s) - 2):
                h = int.from_bytes(hashlib.md5(s[j:j + 3].encode()).digest()[:4], "little")
                m[i, h % self.dim] += 1.0
        return _l2(m)


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    sims = matrix @ query.reshape(-1)
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
