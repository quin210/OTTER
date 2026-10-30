"""OCR quality against human-corrected pages: CER, WER, diacritic accuracy.

  .venv/bin/python -m otter.textbooks.ocr_eval annotation/ocr_review            # all systems found
  .venv/bin/python -m otter.textbooks.ocr_eval annotation/ocr_review --json out.json

annotation/ocr_review/manifest.jsonl lists the review items; for each item ``<id>``:
  pages/<id>.qwen.md        Qwen3-VL OCR (the file the annotator copies and corrects)
  pages/<id>.gemini.md      Gemini OCR (optional, when available)
  pages/<id>.corrected.md   human-corrected reference (created by the annotator)
Only items with a corrected file are scored. Text is compared in two views:
  * ``text``  - NFC, Markdown markup removed (#, **, table pipes, $), whitespace collapsed (main metric);
  * ``raw``   - NFC + whitespace collapsed only.
Metrics: CER = char edit distance / reference chars; WER = word edit distance / reference words;
CER_base = CER after stripping all diacritics from both sides; diacritic accuracy = among aligned word
pairs whose base letters match, the share that also match with diacritics.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Sequence


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_diacritics(s: str) -> str:
    s = nfc(s).replace("đ", "d").replace("Đ", "D")
    return nfc("".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"))


def normalize(text: str, view: str = "text") -> str:
    t = nfc(text or "")
    t = re.sub(r"<<\s*S[ỐO]_?\s*TRANG[^>]*>>", " ", t)
    if view == "text":
        t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)          # headings
        t = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", " ", t, flags=re.M)  # table rules
        t = t.replace("**", "").replace("__", "").replace("|", " ").replace("$", "")
        t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)            # bullets
        t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)           # md images -> caption
        t = t.replace("[", "").replace("]", "")
    return " ".join(t.split())


def edit_distance(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if ref[i - 1] == hyp[j - 1] else (0.5 if strip_diacritics(ref[i - 1]) == strip_diacritics(hyp[j - 1]) else 1)
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0 if ref[i - 1] == hyp[j - 1] else (0.5 if strip_diacritics(ref[i - 1]) == strip_diacritics(hyp[j - 1]) else 1)
            if d[i][j] == d[i - 1][j - 1] + sub:
                out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1; continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def score_pair(ref_text: str, hyp_text: str, view: str = "text") -> dict:
    r, h = normalize(ref_text, view), normalize(hyp_text, view)
    rw, hw = r.split(), h.split()
    cer = edit_distance(r, h) / max(1, len(r))
    wer = edit_distance(rw, hw) / max(1, len(rw))
    cer_base = edit_distance(strip_diacritics(r), strip_diacritics(h)) / max(1, len(r))
    pairs = [(a, b) for a, b in align_words(rw, hw) if a is not None and b is not None
             and strip_diacritics(a).lower() == strip_diacritics(b).lower()]
    with_dia = [(a, b) for a, b in pairs if a != strip_diacritics(a)]
    dia_acc = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
    dia_acc_marked = sum(a == b for a, b in with_dia) / len(with_dia) if with_dia else None
    return {"ref_chars": len(r), "ref_words": len(rw), "cer": round(cer, 4), "wer": round(wer, 4),
            "cer_base": round(cer_base, 4), "diacritic_acc": None if dia_acc is None else round(dia_acc, 4),
            "diacritic_acc_marked_words": None if dia_acc_marked is None else round(dia_acc_marked, 4)}


def evaluate_dir(review_dir: Path, view: str = "text") -> dict:
    manifest = [json.loads(l) for l in (review_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    per_item, systems = [], {}
    for it in manifest:
        ref_f = review_dir / "pages" / f"{it['id']}.corrected.md"
        if not ref_f.exists():
            continue
        ref = ref_f.read_text(encoding="utf-8")
        row = {"id": it["id"], "book_id": it.get("book_id"), "pdf_page_index": it.get("pdf_page_index"), "stratum": it.get("stratum")}
        for sys_f in sorted((review_dir / "pages").glob(f"{it['id']}.*.md")):
            name = sys_f.name[len(it["id"]) + 1:-3]
            if name == "corrected":
                continue
            sc = score_pair(ref, sys_f.read_text(encoding="utf-8"), view)
            row[name] = sc
            systems.setdefault(name, []).append(sc)
        per_item.append(row)

    def agg(scores: list[dict]) -> dict:
        chars = sum(s["ref_chars"] for s in scores) or 1
        words = sum(s["ref_words"] for s in scores) or 1
        das = [s["diacritic_acc"] for s in scores if s["diacritic_acc"] is not None]
        return {"pages": len(scores),
                "cer_micro": round(sum(s["cer"] * s["ref_chars"] for s in scores) / chars, 4),
                "wer_micro": round(sum(s["wer"] * s["ref_words"] for s in scores) / words, 4),
                "cer_base_micro": round(sum(s["cer_base"] * s["ref_chars"] for s in scores) / chars, 4),
                "cer_macro": round(sum(s["cer"] for s in scores) / len(scores), 4),
                "diacritic_acc_macro": round(sum(das) / len(das), 4) if das else None}
    return {"view": view, "n_corrected": len(per_item), "systems": {k: agg(v) for k, v in systems.items()}, "items": per_item}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("review_dir", nargs="?", default="annotation/ocr_review")
    p.add_argument("--view", default="text", choices=["text", "raw"])
    p.add_argument("--json")
    a = p.parse_args(argv)
    res = evaluate_dir(Path(a.review_dir), a.view)
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"view": res["view"], "n_corrected": res["n_corrected"], "systems": res["systems"]}, ensure_ascii=False, indent=1))
    if res["n_corrected"] == 0:
        print("no *.corrected.md files yet", file=sys.stderr)


if __name__ == "__main__":
    main()
