"""python -m otter.builder {build,audit-sample,audit-score,revalidate,ingest-books} ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m otter.builder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the ontology DB from WS1 spans")
    b.add_argument("--db", default="data/textbooks/ontology.sqlite3", help="provenance DB (read-only)")
    b.add_argument("--out", default="data/ontology/ontology.sqlite3", help="ontology DB to (re)write atomically")
    b.add_argument("--export-dir", default="data/ontology")
    b.add_argument("--cache-dir", default="data/ontology/cache")
    b.add_argument("--books", nargs="*", help="book ids / fnmatch patterns / 'core' (default: all books with spans)")
    b.add_argument("--backbone", default="main", choices=["main", "small"], help="extractor LLM role")
    b.add_argument("--verifier", default=None, choices=["main", "small"],
                   help="support-check LLM role (default: main->small, small->main)")
    b.add_argument("--workers", type=int, default=48)
    b.add_argument("--verify-workers", type=int, default=48)
    b.add_argument("--min-tokens", type=int, default=1500)
    b.add_argument("--max-tokens", type=int, default=3000)
    b.add_argument("--limit-chunks", type=int, default=None)
    b.add_argument("--include-exercises", action="store_true")
    b.add_argument("--no-verify", action="store_true", help="lexical-only support check")
    b.add_argument("--no-embed", action="store_true")
    b.add_argument("--no-span-embeddings", action="store_true")
    b.add_argument("--report-dir", default=None)
    b.add_argument("--corpus-manifest", default="runs/corpus/CORPUS_V1.json")
    b.add_argument("--max-failed-frac", type=float, default=0.2)
    b.add_argument("--require-verifier", action="store_true", help="fail instead of lexical-only support check")
    b.add_argument("--url", action="append", default=[], metavar="ROLE=BASE_URL",
                   help="override an LLM role endpoint for this run only, e.g. main=http://127.0.0.1:8004/v1")
    b.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("audit-sample", help="stratified human-audit sheet (CSV + HTML)")
    s.add_argument("--onto", default="data/ontology/ontology.sqlite3")
    s.add_argument("--db", default="data/textbooks/ontology.sqlite3")
    s.add_argument("--out-dir", default="annotation/ontology_audit")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=2027)

    sc = sub.add_parser("audit-score", help="precision from a filled audit sheet")
    sc.add_argument("sheet", nargs="?", default="annotation/ontology_audit/ontology_audit.csv")
    sc.add_argument("--out", default=None)

    rv = sub.add_parser("revalidate", help="knowledge maintenance: re-validate an ontology build against a new edition "
                                           "of the corpus; writes a NEW version file + changelog (old file untouched)")
    rv.add_argument("--old-db", default="data/ontology/ontology.sqlite3", help="ontology DB of the old build (read-only)")
    rv.add_argument("--old-prov", default=None, help="provenance DB of the OLD edition (default: paths.db)")
    rv.add_argument("--new-spans", required=True, help="spans.jsonl | dir of <book>/spans.jsonl | sqlite DB with a spans table")
    rv.add_argument("--out", required=True, help="NEW ontology DB file (must differ from --old-db)")
    rv.add_argument("--report-dir", default=None, help="changelog/alignment/summary dir (default: next to --out)")
    rv.add_argument("--book-prefix", default=None)
    rv.add_argument("--verifier", default="small", choices=["main", "small"], help="support-check LLM role (default small :8001)")
    rv.add_argument("--workers", type=int, default=8)
    rv.add_argument("--lost-span-policy", default="recheck", choices=["recheck", "strict"],
                    help="item with SOME cited spans gone: re-check against the rest (default) | orphan it")
    rv.add_argument("--lexical-only", action="store_true", help="no LLM: lexical support label is final")
    rv.add_argument("--lexical-shortcut", action="store_true",
                    help="skip the LLM for items whose claim/evidence lexical overlap did not drop")
    rv.add_argument("--no-embed", action="store_true", help="do not write span vectors / do not call :8002")
    rv.add_argument("--no-provenance", action="store_true", help="do not embed books/sections/spans of the new edition")
    rv.add_argument("--reason", default="new edition of the corpus")
    rv.add_argument("--build-id", default=None)
    rv.add_argument("--cache-dir", default=None, help="verifier batch cache (default <out>.cache); makes the run resumable")
    rv.add_argument("--dry-run", action="store_true", help="align + classify + count LLM work; write nothing")
    rv.add_argument("-v", "--verbose", action="store_true")


    ib = sub.add_parser("ingest-books", help="knowledge maintenance: ingest NEW books (other publishers / editions) into an existing "
                                             "ontology; writes a NEW version file + changelog/merge log/conflict list (old file untouched)")
    ib.add_argument("--old-db", default="data/ontology/ontology.sqlite3", help="existing ontology DB (read-only)")
    ib.add_argument("--old-prov", default=None, help="provenance DB of the OLD books (default: paths.db)")
    ib.add_argument("--new-spans", required=True, help="dir of <book>/spans.jsonl | spans.jsonl | sqlite DB with a spans table")
    ib.add_argument("--books", nargs="*", help="new book ids / fnmatch patterns (default: every book found in --new-spans)")
    ib.add_argument("--out", required=True, help="NEW ontology DB file (must differ from --old-db)")
    ib.add_argument("--report-dir", default=None, help="changelog / merges / conflicts / summary dir (default: next to --out)")
    ib.add_argument("--book-meta", default=None, help="JSON file {book_id | '*': {publisher, series, edition, edition_year, subject, grade, title}}")
    ib.add_argument("--publisher", default=None)
    ib.add_argument("--series", default=None)
    ib.add_argument("--edition", default=None)
    ib.add_argument("--edition-year", type=int, default=None)
    ib.add_argument("--extractor", default="main", choices=["main", "small"], help="extraction LLM role (cache is keyed by model)")
    ib.add_argument("--verifier", default="small", choices=["main", "small"], help="support-check LLM role")
    ib.add_argument("--judge", default="small", choices=["main", "small"], help="claim agree/conflict LLM role (uncertain pairs only)")
    ib.add_argument("--workers", type=int, default=4, help="extraction workers (max 6)")
    ib.add_argument("--verify-workers", type=int, default=3)
    ib.add_argument("--judge-workers", type=int, default=3)
    ib.add_argument("--limit-chunks", type=int, default=None, help="smoke test: only the first N chunks")
    ib.add_argument("--sim-threshold", type=float, default=0.93)
    ib.add_argument("--max-judge-pairs", type=int, default=800)
    ib.add_argument("--include-conflicts", action="store_true",
                    help="keep conflicting items visible as 'alternative_source' (default: hidden, listed for review)")
    ib.add_argument("--claim-rules", default="v2", choices=["v1", "v2"], help="claim comparison rule set (v1 = first release, kept for reproducibility)")
    ib.add_argument("--no-cross-kind", action="store_true", help="compare facts only with facts and rules only with rules")
    ib.add_argument("--no-confirm-conflicts", action="store_true", help="do not LLM-check lexical (same wording, other number) conflicts")
    ib.add_argument("--lexical-only", action="store_true", help="no claim LLM: uncertain pairs become 'ambiguous'")
    ib.add_argument("--no-verify", action="store_true", help="lexical-only support check (flagged)")
    ib.add_argument("--no-accumulate", action="store_true", help="agreeing items: do NOT add the new book's spans/aliases to the old item")
    ib.add_argument("--no-embed", action="store_true")
    ib.add_argument("--no-old-provenance", action="store_true", help="do not copy the old books' spans into the new file")
    ib.add_argument("--allow-reingest", action="store_true")
    ib.add_argument("--force", action="store_true", help="replace an existing --out of a previous ingest")
    ib.add_argument("--reason", default="new books ingested")
    ib.add_argument("--build-id", default=None)
    ib.add_argument("--cache-dir", default=None, help="extract/ verify/ judge/ emb_cache.sqlite3 (default <out>.cache); reuse = resumable")
    ib.add_argument("--dry-run", action="store_true", help="segment + count chunks (cached vs to extract); write nothing")
    ib.add_argument("--url", action="append", default=[], metavar="ROLE=BASE_URL", help="override an LLM role endpoint for this run")
    ib.add_argument("-v", "--verbose", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(a, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "build":
        from otter.builder.pipeline import BuildConfig, build
        cfg = BuildConfig(db=Path(a.db), out=Path(a.out), export_dir=Path(a.export_dir), cache_dir=Path(a.cache_dir),
                          books=a.books or None, backbone=a.backbone, verifier=a.verifier, workers=a.workers,
                          verify_workers=a.verify_workers, min_tokens=a.min_tokens, max_tokens=a.max_tokens,
                          include_exercises=a.include_exercises, limit_chunks=a.limit_chunks,
                          no_verify=a.no_verify, no_embed=a.no_embed, embed_spans=not a.no_span_embeddings,
                          report_dir=Path(a.report_dir) if a.report_dir else None,
                          corpus_manifest=Path(a.corpus_manifest) if a.corpus_manifest else None,
                          max_failed_frac=a.max_failed_frac, require_verifier=a.require_verifier)
        from otter.llm.interface import load_config
        conf = load_config()
        for ov in a.url:
            role, url = ov.split("=", 1)
            conf["llm"][role]["base_url"] = url
        rep = build(cfg, config=conf)
        print(json.dumps({"build_id": rep["build_id"], "totals": rep["totals"], "support": rep["support"],
                          "timings_s": rep["timings_s"]}, ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "revalidate":
        from otter.builder.revalidate import RevalConfig, revalidate
        from otter.llm.interface import get_llm
        from otter.storage.embeddings import Bgem3Embedder
        out = Path(a.out)
        cfg = RevalConfig(old_db=Path(a.old_db), new_spans=Path(a.new_spans), out=out,
                          old_prov=Path(a.old_prov) if a.old_prov else None, book_prefix=a.book_prefix,
                          verifier_role=a.verifier, workers=a.workers, lost_span_policy=a.lost_span_policy,
                          lexical_only=a.lexical_only, lexical_shortcut=a.lexical_shortcut, embed=not a.no_embed,
                          embed_provenance=not a.no_provenance, reason=a.reason, build_id=a.build_id,
                          report_dir=Path(a.report_dir) if a.report_dir else None,
                          cache_dir=Path(a.cache_dir) if a.cache_dir else out.with_name(out.stem + ".cache"),
                          dry_run=a.dry_run)
        verifier = None if (a.lexical_only or a.dry_run) else get_llm(a.verifier)
        emb = None
        if not a.no_embed:
            e = Bgem3Embedder.from_config()
            emb = e if e.available() else None
        res = revalidate(cfg, verifier=verifier, embedder=emb)
        print(json.dumps({k: res[k] for k in ("build_id", "parent_build_id", "totals", "invalid_items", "alignment", "cost",
                                              "added_spans_without_old_counterpart") if k in res}, ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "ingest-books":
        from otter.builder.ingest_books import IngestConfig, ingest_books
        from otter.llm.interface import load_config
        conf = load_config()
        for ov in a.url:
            role, url = ov.split("=", 1)
            conf["llm"][role]["base_url"] = url
        meta = json.loads(Path(a.book_meta).read_text(encoding="utf-8")) if a.book_meta else {}
        star = {k: v for k, v in (("publisher", a.publisher), ("series", a.series), ("edition", a.edition),
                                  ("edition_year", a.edition_year)) if v is not None}
        if star:
            meta["*"] = {**meta.get("*", {}), **star}
        out = Path(a.out)
        cfg = IngestConfig(old_db=Path(a.old_db), new_spans=Path(a.new_spans), out=out,
                           old_prov=Path(a.old_prov) if a.old_prov else None, books=a.books or None, book_meta=meta,
                           extractor_role=a.extractor, verifier_role=a.verifier, judge_role=a.judge, workers=a.workers,
                           verify_workers=a.verify_workers, judge_workers=a.judge_workers, limit_chunks=a.limit_chunks,
                           sim_threshold=a.sim_threshold, max_judge_pairs=a.max_judge_pairs,
                           include_conflicts=a.include_conflicts, claim_rules=a.claim_rules, cross_kind=not a.no_cross_kind, confirm_conflicts=not a.no_confirm_conflicts,
                           lexical_only=a.lexical_only, no_verify=a.no_verify, accumulate_provenance=not a.no_accumulate,
                           merge_aliases=not a.no_accumulate, embed=not a.no_embed,
                           embed_old_provenance=not a.no_old_provenance, allow_reingest=a.allow_reingest, force=a.force,
                           reason=a.reason, build_id=a.build_id, report_dir=Path(a.report_dir) if a.report_dir else None,
                           cache_dir=Path(a.cache_dir) if a.cache_dir else None, dry_run=a.dry_run)
        if cfg.old_prov is None:
            from otter.llm.interface import CONFIG_PATH
            cfg.old_prov = CONFIG_PATH.parents[1] / conf["paths"]["db"]
        res = ingest_books(cfg, config=conf)
        keys = ("build_id", "parent_build_id", "verdicts", "handling", "concept_merge_decisions", "cost", "extraction", "verification",
                "claim_judge", "old_items_touched", "conflict_rows", "old_db_unchanged", "dry_run", "chunks", "chunks_cached",
                "chunks_to_extract", "books", "spans")
        print(json.dumps({k: res[k] for k in keys if k in res}, ensure_ascii=False, indent=1, default=str))
        return 0
    if a.cmd == "audit-sample":
        from otter.builder.audit import make_audit
        print(json.dumps(make_audit(Path(a.onto), Path(a.db), Path(a.out_dir), n=a.n, seed=a.seed),
                         ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "audit-score":
        from otter.builder.audit import score_audit
        res = score_audit(Path(a.sheet))
        txt = json.dumps(res, ensure_ascii=False, indent=1)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf-8")
        print(txt)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
