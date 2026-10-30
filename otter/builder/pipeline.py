"""End-to-end offline Textbook Builder (spec 3.4): ingest -> segment -> extract -> normalize -> validate
-> persist -> report. Deterministic orchestration; the LLM is only used for extraction and the separate
support check (by a different model)."""
from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import logging
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, get_args

from otter.builder import extract as X
from otter.builder import validate as V
from otter.builder.ingest import load_books, load_sections, load_spans
from otter.builder.normalize import normalize
from otter.builder.persist import EmbeddingCache, persist
from otter.builder.report import compute_report, write_report
from otter.builder.segment import segment_book
from otter.llm.interface import CONFIG_PATH, LLM, get_llm, load_config
from otter.schemas.ontology import Subject

log = logging.getLogger("otter.builder")
ROOT = CONFIG_PATH.parents[1]
_SUBJECTS = set(get_args(Subject))
VERIFIER_FOR = {"main": ["small"], "small": ["main"]}       # never alt / index (reserved roles)


@dataclass
class BuildConfig:
    db: Path = ROOT / "data/textbooks/ontology.sqlite3"            # provenance DB (WS1, read-only)
    out: Path = ROOT / "data/ontology/ontology.sqlite3"            # ontology DB (WS2)
    export_dir: Path = ROOT / "data/ontology"
    cache_dir: Path = ROOT / "data/ontology/cache"
    parsed_dir: Path = ROOT / "data/textbooks/parsed"
    registry_dir: Path = ROOT / "data/textbooks/registry"
    books: Optional[list[str]] = None                             # ids or fnmatch patterns; None = all with spans
    backbone: str = "main"                                         # extractor role
    verifier: Optional[str] = None                                 # verifier role (default: a different model)
    workers: int = 48
    verify_workers: int = 48
    min_tokens: int = 1500
    max_tokens: int = 3000
    include_exercises: bool = False
    limit_chunks: Optional[int] = None
    sim_threshold: float = 0.93
    no_verify: bool = False
    no_embed: bool = False
    embed_spans: bool = True
    max_output_tokens: int = 6144
    report_dir: Optional[Path] = None                              # default: export_dir
    corpus_manifest: Optional[Path] = ROOT / "runs/corpus/CORPUS_V1.json"
    max_failed_frac: float = 0.2                                   # abort before persist above this
    require_verifier: bool = False                                 # abort instead of lexical-only fallback

    def jsonable(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


def endpoint_up(role: str, config: Optional[dict] = None) -> bool:
    try:
        cfg = (config or load_config())["llm"][role]
        with urllib.request.urlopen(cfg["base_url"].rstrip("/") + "/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def pick_verifier(backbone: str, explicit: Optional[str], config: dict) -> Optional[str]:
    ext_model = config["llm"][backbone]["model"]
    cands = [explicit] if explicit else VERIFIER_FOR.get(backbone, ["small"])
    for role in cands:
        if role in ("alt", "index"):
            continue
        if role and role in config["llm"] and config["llm"][role]["model"] != ext_model and endpoint_up(role, config):
            return role
    return None


def select_books(all_ids: list[str], patterns: Optional[list[str]]) -> list[str]:
    if not patterns:
        return sorted(all_ids)
    out = []
    for p in patterns:
        if p == "core":
            p = "nxbgd_*"
        out += [b for b in sorted(all_ids) if fnmatch.fnmatch(b, p) and b not in out]
    return out


def build(cfg: BuildConfig, *, extractor_llm: Optional[LLM] = None, verifier_llm: Optional[LLM] = None,
          embedder=None, config: Optional[dict] = None) -> dict[str, Any]:
    t_all = time.time()
    config = config or load_config()
    timings: dict[str, float] = {}
    # ------------------------------------------------------------------ ingest
    t = time.time()
    books_all = load_books(cfg.db, None, cfg.registry_dir)
    spans_by_book = load_spans(cfg.db, list(books_all), cfg.parsed_dir)
    with_spans = [b for b, s in spans_by_book.items() if s]
    book_ids = select_books(with_spans, cfg.books)
    if not book_ids:
        raise SystemExit(f"no books with spans match {cfg.books!r} (books with spans: {with_spans})")
    books = {b: books_all[b] for b in book_ids}
    sections = load_sections(cfg.db, book_ids, cfg.parsed_dir)
    timings["ingest"] = round(time.time() - t, 2)
    corpus: dict[str, Any] = {"corpus_version": None, "spans_sha256": None}
    if cfg.corpus_manifest and Path(cfg.corpus_manifest).exists():
        man = json.loads(Path(cfg.corpus_manifest).read_text(encoding="utf-8"))
        per = man.get("per_book") or {}
        corpus = {"corpus_version": man.get("corpus_version"), "spans_sha256": man.get("spans_sha256"),
                  "manifest": str(cfg.corpus_manifest), "manifest_books": man.get("books"),
                  "manifest_spans": man.get("spans"),
                  "span_count_mismatch": {b: {"manifest": per.get(b), "loaded": len(spans_by_book[b])}
                                          for b in book_ids if per.get(b) is not None
                                          and per.get(b) != len(spans_by_book[b])},
                  "books_not_in_manifest": [b for b in book_ids if per and b not in per]}
    # ------------------------------------------------------------------ segment
    chunks, chunk_stats, spans = [], {}, {}
    for b in book_ids:
        info = books[b]
        subj = info.subject if info.subject in _SUBJECTS else "other"
        bs = spans_by_book[b]
        spans.update({s.span_id: s for s in bs})
        cs = segment_book(b, bs, sections, subject=subj, grade=info.grade, book_title=info.title,
                          min_tokens=cfg.min_tokens, max_tokens=cfg.max_tokens,
                          include_exercises=cfg.include_exercises)
        chunk_stats[b] = {"spans": len(bs), "chunks": len(cs), "chunk_tokens_est": sum(c.tokens for c in cs)}
        chunks += cs
    if cfg.limit_chunks:
        chunks = chunks[: cfg.limit_chunks]
    log.info("%d books, %d spans, %d chunks (~%d tokens)", len(book_ids), len(spans), len(chunks),
             sum(c.tokens for c in chunks))
    # ------------------------------------------------------------------ extract
    t = time.time()
    llm = extractor_llm or get_llm(cfg.backbone, config)
    extractor = X.Extractor(llm, cfg.cache_dir / "extract", max_tokens=cfg.max_output_tokens)
    results = extractor.run(chunks, workers=cfg.workers)
    timings["extract"] = round(time.time() - t, 2)
    live = [r for r in results if not r.cached]
    extraction = {
        "model": getattr(llm, "model", ""), "prompt_version": X.PROMPT_VERSION, "chunks": len(results),
        "llm_chunks": sum(1 for r in live), "cached_chunks": sum(1 for r in results if r.cached),
        "failed_chunks": sum(1 for r in results if r.status in ("parse_failed", "llm_error")),
        "failures": [{"chunk_id": r.chunk.chunk_id, "status": r.status, "error": (r.error or "")[:300]}
                     for r in results if r.status in ("parse_failed", "llm_error")][:50],
        "prompt_tokens": sum(r.prompt_tokens for r in live), "completion_tokens": sum(r.completion_tokens for r in live),
        "cached_prompt_tokens": sum(r.prompt_tokens for r in results if r.cached),
        "cached_completion_tokens": sum(r.completion_tokens for r in results if r.cached),
        "item_schema_errors": sum(len(r.item_errors) for r in results),
    }
    if live and timings["extract"] > 0:
        extraction["throughput"] = {
            "chunks_per_min": round(60 * len(live) / timings["extract"], 2),
            "completion_tok_per_s": round(extraction["completion_tokens"] / timings["extract"], 1),
            "prompt_tok_per_s": round(extraction["prompt_tokens"] / timings["extract"], 1),
            "mean_latency_s": round(sum(r.latency_ms for r in live) / len(live) / 1000, 2)}
    if results and extraction["failed_chunks"] / len(results) > cfg.max_failed_frac:
        raise SystemExit(f"{extraction['failed_chunks']}/{len(results)} chunks failed (> {cfg.max_failed_frac:.0%}); "
                         f"not persisting. Successful chunks are cached; re-run to resume. "
                         f"First failures: {extraction['failures'][:3]}")
    # ------------------------------------------------------------------ normalize
    t = time.time()
    emb = None
    if not cfg.no_embed:
        if embedder is None:
            from otter.storage.embeddings import Bgem3Embedder
            embedder = Bgem3Embedder.from_config(config)
        if embedder.available():
            emb = EmbeddingCache(embedder, cfg.cache_dir / "emb_cache.sqlite3")
        else:
            log.warning("embedding server unavailable: string-only canonicalisation, no dense vectors")
    onto = normalize(results, spans, embedder=emb, sim_threshold=cfg.sim_threshold)
    timings["normalize"] = round(time.time() - t, 2)
    # ------------------------------------------------------------------ validate
    t = time.time()
    V.structural_checks(onto, cfg.db)
    V.missing_guard_check(onto, spans)
    if verifier_llm is None and not cfg.no_verify:
        role = pick_verifier(cfg.backbone, cfg.verifier, config)
        if role is None and cfg.require_verifier:
            raise SystemExit("no verifier endpoint (different model) is up; extraction is cached - re-run later")
        if role is None:
            log.warning("no verifier endpoint different from the extractor is up: lexical-only support check")
        else:
            verifier_llm = get_llm(role, config)
    if verifier_llm is not None and getattr(verifier_llm, "model", "") == getattr(llm, "model", "") \
            and getattr(llm, "model", "") not in ("", "mock"):
        raise ValueError("support verifier must be a different model than the extractor")
    checker = V.SupportChecker(None if cfg.no_verify else verifier_llm, cfg.cache_dir / "verify")
    checker.run(onto, spans, workers=cfg.verify_workers)
    timings["validate"] = round(time.time() - t, 2)
    verification = {"model": getattr(checker.llm, "model", None), "verify_version": V.VERIFY_VERSION,
                    "mode": "lexical+llm" if checker.llm is not None else "lexical_only", **checker.stats}
    # ------------------------------------------------------------------ persist
    t = time.time()
    build_id = f"onto-{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{cfg.backbone}"
    run_cfg = cfg.jsonable() | {"extractor": extraction["model"], "verifier": verification["model"],
                                "prompt_version": X.PROMPT_VERSION, "verify_version": V.VERIFY_VERSION,
                                "book_ids": book_ids, **corpus}
    pinfo = persist(onto, cfg.out, build_id=build_id, extractor=f"{extraction['model']}|{X.PROMPT_VERSION}",
                    verifier=f"{verification['model']}|{V.VERIFY_VERSION}|{verification['mode']}",
                    embedder=emb, embed_spans=cfg.embed_spans, spans=spans, config=run_cfg,
                    stats={"concepts": len(onto.concepts), "relations": len(onto.relations), "facts": len(onto.facts),
                           "rules": len(onto.rules), "objectives": len(onto.objectives),
                           "corpus_version": corpus["corpus_version"], "spans_sha256": corpus["spans_sha256"]},
                    notes=f"corpus_version={corpus['corpus_version']} spans_sha256={corpus['spans_sha256']}",
                    export_dir=cfg.export_dir / "export")
    if emb is not None:
        pinfo["embedding_cache"] = {"hits": emb.hits, "misses": emb.misses}
    timings["persist"] = round(time.time() - t, 2)
    timings["total"] = round(time.time() - t_all, 2)
    rep = compute_report(onto, books=books, chunk_stats=chunk_stats, extraction=extraction, verification=verification,
                         timings=timings, config=run_cfg, persist_info=pinfo, build_id=build_id)
    write_report(rep, cfg.report_dir or cfg.export_dir)
    return rep
