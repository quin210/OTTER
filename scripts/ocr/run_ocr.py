#!/usr/bin/env python
"""Resumable VLM OCR of all scanned textbook PDFs -> data/textbooks/parsed/<book_id>/pages.jsonl.

  .venv/bin/python scripts/ocr/run_ocr.py --endpoints http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1

Order: the cataloged books (registry/books.jsonl with a local_file) first, then the
unmatched local PDFs (registry/unmatched_files.jsonl) under sha-derived book ids
``local_<sha256[:10]>`` (see register_unmatched.py for metadata from the cover).
Per-page raw results are cached in parsed/<book_id>/ocr_raw/pNNNN.json, so the
job can be killed and restarted at any time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.textbooks.extractor import EXTRACTOR_VERSION, extract_page_images  # noqa: E402
from otter.textbooks.ocr import OCR_VERSION, VisionLLMOCRAdapter  # noqa: E402
from otter.textbooks.pages import assemble_book_pages, load_jobs  # noqa: E402

PARSED = ROOT / "data/textbooks/parsed"


def _extract(args):
    book_id, pdf = args
    imgs = extract_page_images(Path(pdf), PARSED / book_id / "images")
    return book_id, [(i.pdf_page_index, str(i.path), i.width, i.height, i.method) for i in imgs]


async def main_async(a):
    jobs = load_jobs(ROOT, a.books)
    if a.max_books:
        jobs = jobs[:a.max_books]
    print(f"{len(jobs)} books queued", flush=True)
    adapter = VisionLLMOCRAdapter(a.endpoints.split(","), model=a.model, per_endpoint_concurrency=a.concurrency,
                                  max_tokens=a.max_tokens)
    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=a.extract_workers)
    ext_futs = {j["book_id"]: loop.run_in_executor(pool, _extract, (j["book_id"], j["local_file"])) for j in jobs}
    stats = {"pages_done": 0, "pages_skipped": 0, "errors": 0, "t0": time.time()}
    progress_path = ROOT / "runs/ocr/progress.json"; progress_path.parent.mkdir(parents=True, exist_ok=True)

    async def do_page(job, idx, path, w, h, method):
        out = PARSED / job["book_id"] / "ocr_raw" / f"p{idx:04d}.json"
        if out.exists():
            try:
                if not json.loads(out.read_text(encoding="utf-8")).get("error"):
                    stats["pages_skipped"] += 1; return
            except ValueError:
                pass
        res = await adapter.ocr_page(Path(path))
        rec = asdict(res) | {"book_id": job["book_id"], "pdf_page_index": idx, "image_path": os.path.relpath(path, ROOT),
                             "image_width": w, "image_height": h, "image_method": method,
                             "ocr_version": OCR_VERSION, "extractor_version": EXTRACTOR_VERSION}
        tmp = out.with_suffix(".tmp"); out.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8"); tmp.replace(out)
        stats["pages_done"] += 1; stats["errors"] += bool(res.error)
        n = stats["pages_done"]
        if n % 25 == 0:
            el = time.time() - stats["t0"]
            msg = {"pages_done": n, "skipped": stats["pages_skipped"], "errors": stats["errors"], "elapsed_s": round(el),
                   "pages_per_min": round(60 * n / el, 1), "last": f"{job['book_id']}:p{idx:04d}"}
            print(json.dumps(msg), flush=True); progress_path.write_text(json.dumps(msg | {"total_pages": stats.get("total")}))

    async def do_book(job):
        book_id, pages = await ext_futs[job["book_id"]]
        stats["total"] = stats.get("total", 0) + len(pages)
        await asyncio.gather(*(do_page(job, *p) for p in pages))
        n = assemble_book_pages(ROOT, job)
        print(f"[book done] {book_id}: {n} pages -> {PARSED / book_id / 'pages.jsonl'}", flush=True)

    # cataloged books first: limit how many books are in flight so earlier books finish first
    sem = asyncio.Semaphore(a.books_in_flight)
    async def guarded(job):
        async with sem:
            try:
                await do_book(job)
            except Exception as e:  # keep going with other books
                print(f"[book error] {job['book_id']}: {type(e).__name__}: {e}", flush=True)
    await asyncio.gather(*(guarded(j) for j in jobs))
    pool.shutdown()
    el = time.time() - stats["t0"]
    print(json.dumps({"finished": True, "pages_done": stats["pages_done"], "skipped": stats["pages_skipped"],
                      "errors": stats["errors"], "elapsed_s": round(el)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoints", default="http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--books", default="all", help="all | core | unmatched | comma-separated book ids")
    p.add_argument("--concurrency", type=int, default=48, help="in-flight requests per endpoint")
    p.add_argument("--books-in-flight", type=int, default=14)
    p.add_argument("--extract-workers", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-books", type=int, default=0)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
