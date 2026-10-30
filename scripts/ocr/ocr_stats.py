#!/usr/bin/env python
"""Aggregate OCR statistics -> runs/ocr/ocr_stats.json (from parsed/<book>/ocr_raw + pages.jsonl)."""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARSED = ROOT / "data/textbooks/parsed"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    per_book, raws, pages_all, mtimes = {}, [], [], []
    for d in sorted(PARSED.iterdir()):
        rd = d / "ocr_raw"
        if not rd.is_dir():
            continue
        rs = []
        for f in rd.glob("p*.json"):
            try:
                rs.append(json.loads(f.read_text(encoding="utf-8"))); mtimes.append(f.stat().st_mtime)
            except ValueError:
                pass
        pf = d / "pages.jsonl"
        pages = [json.loads(l) for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()] if pf.exists() else []
        pages = [p for p in pages if p.get("extraction_method") == "vlm_ocr"]
        raws += rs; pages_all += pages
        per_book[d.name] = {
            "ocr_pages": len(rs), "assembled_pages": len(pages),
            "errors": sum(bool(r.get("error")) for r in rs),
            "degenerate": sum(bool(r.get("degenerate")) for r in rs),
            "manual_review": sum(bool(p.get("requires_manual_review")) for p in pages),
            "mean_quality": round(st.mean(p["quality_score"] for p in pages if p.get("quality_score") is not None), 4) if pages else None,
            "printed_label_verified": sum(bool(p.get("printed_page_verified")) for p in pages),
            "ocr_source": dict(Counter(p.get("ocr_source", "qwen") for p in pages)),
        }
    lat = [r["latency_s"] for r in raws if r.get("latency_s")]
    toks = [r["completion_tokens"] for r in raws if r.get("completion_tokens")]
    span_min = (max(mtimes) - min(mtimes)) / 60 if len(mtimes) > 1 else None
    n = len(raws)
    both = [r for r in raws if r.get("printed_page_label_tag") is not None or r.get("printed_page_label_crop") is not None]
    out = {
        "model": Counter(r.get("model") for r in raws).most_common(3),
        "ocr_versions": dict(Counter(r.get("ocr_version") for r in raws)),
        "books": len(per_book), "pages_ocr": n, "pages_assembled": len(pages_all),
        "throughput_pages_per_min_wallclock": round(n / span_min, 1) if span_min else None,
        "throughput_note": "pages / (last - first raw-file mtime), includes restarts; 3 x Qwen3-VL-8B (1 per RTX 6000 Ada), "
                           "48 concurrent page requests per GPU",
        "latency_s": {"mean": round(st.mean(lat), 2) if lat else None, "p50": pct(lat, .5), "p95": pct(lat, .95), "max": max(lat) if lat else None},
        "completion_tokens": {"p50": pct(toks, .5), "p95": pct(toks, .95), "p99": pct(toks, .99), "max": max(toks) if toks else None},
        "failures": sum(bool(r.get("error")) for r in raws),
        "degenerate_repeats": sum(bool(r.get("degenerate")) for r in raws),
        "repetition_tail_trimmed": sum(bool(r.get("trimmed_repetition")) for r in raws),
        "retried_pages": sum((r.get("attempts") or 1) > 1 for r in raws),
        "page_tag_missing_first_attempt": sum(any("tag_missing" in x for x in r.get("notes", [])) for r in raws),
        "manual_review_pct": round(100 * sum(bool(p.get("requires_manual_review")) for p in pages_all) / max(1, len(pages_all)), 2),
        "status": dict(Counter(p.get("status") for p in pages_all)),
        "mean_quality_score": round(st.mean(p["quality_score"] for p in pages_all if p.get("quality_score") is not None), 4) if pages_all else None,
        "printed_page_labels": {
            "pages_with_any_read": len(both),
            "tag_crop_agree": sum(r.get("printed_page_label_tag") is not None and r.get("printed_page_label_tag") == r.get("printed_page_label_crop") for r in raws),
            "verified_in_pages": sum(bool(p.get("printed_page_verified")) for p in pages_all),
        },
        "per_book": per_book,
    }
    dst = ROOT / "runs/ocr/ocr_stats.json"; dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    json.dump({k: v for k, v in out.items() if k != "per_book"}, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
