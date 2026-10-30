#!/usr/bin/env python
"""Build the OCR review sheet annotation/ocr_review/ (~20 stratified pages for human correction).

  .venv/bin/python scripts/ocr/make_ocr_review.py            # select pages (once; keeps an existing manifest)
  .venv/bin/python scripts/ocr/make_ocr_review.py --refresh  # only (re)write system outputs (e.g. new Gemini pages)

Selection (deterministic, seed 2027): one page per cataloged book (18 books) with the page *type* rotating
over {plain text, table, formula, figure-heavy, low quality score}, plus 2 pages from unmatched local
files; front matter / near-empty pages are excluded. Per item: pages/<id>.png (page image),
pages/<id>.qwen.md (Qwen3-VL OCR, raw), pages/<id>.gemini.md (if a Gemini page exists).
The annotator writes pages/<id>.corrected.md; then run ``python -m otter.textbooks.ocr_eval``.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.textbooks.gemini_import import load_imported  # noqa: E402

PARSED = ROOT / "data/textbooks/parsed"
OUT = ROOT / "annotation/ocr_review"
STRATA = ["plain_text", "table", "formula", "figure_heavy", "low_quality"]

README = """# OCR review sheet (WS1)

Purpose: measure OCR quality (CER / WER / diacritic accuracy) of Qwen3-VL-8B (and Gemini, when available)
on {n} stratified textbook pages.

For each item in `manifest.jsonl` (`id`, book, pdf page, stratum):
1. Open `pages/<id>.png` (the scanned page).
2. Copy `pages/<id>.qwen.md` to `pages/<id>.corrected.md` and correct it so it matches the page EXACTLY:
   every word and diacritic, reading order (left column before right), headings as Markdown `#`,
   formulas in `$...$`, tables as Markdown tables, figure captions as `[Hình x.y. ...]`.
   Do NOT add text that is not printed on the page; drop running headers/footers and the page number.
3. Leave `qwen.md` / `gemini.md` untouched.

Score: `.venv/bin/python -m otter.textbooks.ocr_eval annotation/ocr_review --json runs/ocr/ocr_eval.json`
(only items with a `corrected.md` are scored; `--view raw` keeps Markdown markup).
"""


def page_rows(d: Path) -> list[dict]:
    f = d / "pages.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def stratum_ok(p: dict, stratum: str) -> bool:
    m, t = p.get("ocr_metrics") or {}, p.get("text") or ""
    if p.get("status") != "ok" or m.get("n_chars", 0) < 400:
        return False
    if stratum == "table":
        return m.get("n_table_rows", 0) >= 4
    if stratum == "formula":
        return m.get("n_latex", 0) >= 3 or t.count("→") >= 2
    if stratum == "figure_heavy":
        return t.count("Hình") >= 2 and m.get("n_chars", 0) < 1800
    if stratum == "low_quality":
        return (p.get("quality_score") or 1) < 0.95 or bool(p.get("requires_manual_review"))
    return m.get("n_table_rows", 0) == 0 and m.get("n_latex", 0) == 0 and m.get("n_chars", 0) >= 1200


def select(seed: int = 2027) -> list[dict]:
    rng = random.Random(seed)
    reg = [json.loads(l) for l in (ROOT / "data/textbooks/registry/books.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    core = [b["book_id"] for b in reg if (PARSED / b["book_id"] / "pages.jsonl").exists() and b.get("local_file")]
    local = sorted(d.name for d in PARSED.iterdir() if d.name.startswith("local_") and (d / "pages.jsonl").exists())
    items, k = [], 0
    for bid in core:
        rows = [p for p in page_rows(PARSED / bid) if p["pdf_page_index"] >= 4]
        for shift in range(len(STRATA)):            # rotate strata; fall back to the next stratum if none fits
            st = STRATA[(k + shift) % len(STRATA)]
            cands = [p for p in rows if stratum_ok(p, st)]
            if cands:
                p = rng.choice(cands); items.append((bid, p, st)); break
        k += 1
    for bid in rng.sample(local, min(2, len(local))):
        rows = [p for p in page_rows(PARSED / bid) if p["pdf_page_index"] >= 4 and stratum_ok(p, "plain_text")]
        if rows:
            items.append((bid, rng.choice(rows), "plain_text_unmatched"))
    out = []
    for i, (bid, p, st) in enumerate(items, 1):
        out.append({"id": f"r{i:02d}_{bid}_p{p['pdf_page_index']:04d}", "book_id": bid, "pdf_page_index": p["pdf_page_index"],
                    "page_span_id": p["span_id"], "stratum": st, "image_path": p.get("image_path"),
                    "printed_page_label": p.get("printed_page_label"), "qwen_quality_score": p.get("quality_score")})
    return out


def write_outputs(items: list[dict]) -> None:
    pages = OUT / "pages"; pages.mkdir(parents=True, exist_ok=True)
    for it in items:
        d = PARSED / it["book_id"]
        raw = d / "ocr_raw" / f"p{it['pdf_page_index']:04d}.json"
        if raw.exists():
            r = json.loads(raw.read_text(encoding="utf-8"))
            (pages / f"{it['id']}.qwen.md").write_text((r.get("text") or "") + "\n", encoding="utf-8")
        g = load_imported(d).get(it["pdf_page_index"])
        if g and not g.get("error"):
            (pages / f"{it['id']}.gemini.md").write_text((g.get("text") or "") + "\n", encoding="utf-8")
        img = ROOT / (it.get("image_path") or "")
        if img.is_file() and not (pages / f"{it['id']}.png").exists():
            shutil.copyfile(img, pages / f"{it['id']}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    man = OUT / "manifest.jsonl"
    if man.exists():
        items = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        if a.refresh:
            sys.exit("no manifest yet")
        items = select()
        OUT.mkdir(parents=True, exist_ok=True)
        man.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        (OUT / "README.md").write_text(README.format(n=len(items)), encoding="utf-8")
    write_outputs(items)
    print(f"{len(items)} review items in {OUT}")
    for it in items:
        print(it["id"], it["stratum"])


if __name__ == "__main__":
    main()
