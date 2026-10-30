"""Assemble per-page OCR results into PageRecord JSONL (data/textbooks/parsed/<book_id>/pages.jsonl).

Printed page labels: each page has up to two independent reads (tag at the end of
the full-page transcription, and a margin-crop read). A label is kept only if it
was read from the page; ``printed_page_verified`` is True only when
  (a) both reads agree, and
  (b) the label is arithmetically consistent with at least one neighbouring page
      (|delta pdf index| <= 3) whose label implies the same pdf->printed offset.
Otherwise the best candidate is kept with ``printed_page_verified=False``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from otter.schemas.textbook import PageRecord
from otter.textbooks.ocr import quality_metrics, quality_score

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def page_span_id(book_id: str, idx: int) -> str:
    return f"{book_id}:p{idx:04d}"


def local_book_id(sha256: str) -> str:
    return f"local_{sha256[:10]}"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jobs(root: Path, which: str = "all") -> list[dict]:
    """Cataloged books first (registry order), then unmatched local files."""
    reg = root / "data/textbooks/registry"
    core, extra = [], []
    for line in (reg / "books.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        b = json.loads(line)
        if b.get("local_file") and (root / b["local_file"]).exists():
            job = {"book_id": b["book_id"], "local_file": str(root / b["local_file"]), "sha256": b.get("sha256"), "kind": "core"}
            if str(b["book_id"]).startswith("local_"):
                job["kind"] = "unmatched"; extra.append(job)
            else:
                core.append(job)
    seen = {j["book_id"] for j in core + extra}
    um = reg / "unmatched_files.jsonl"
    if um.exists():
        for line in um.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            bid = local_book_id(r["sha256"])
            if bid in seen or not (root / r["local_path"]).exists():
                continue
            extra.append({"book_id": bid, "local_file": str(root / r["local_path"]), "sha256": r["sha256"], "kind": "unmatched",
                          "filename": r["filename"]})
            seen.add(bid)
    # smaller unmatched files first (Chuyên đề books) so more books complete early
    extra.sort(key=lambda j: Path(j["local_file"]).stat().st_size)
    if which == "core":
        return core
    if which == "unmatched":
        return extra
    if which == "all":
        return core + extra
    wanted = set(which.split(","))
    return [j for j in core + extra if j["book_id"] in wanted]


def _as_int(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    if label.isdigit():
        return int(label)
    return None


def resolve_page_labels(raw: dict[int, dict]) -> dict[int, tuple[Optional[str], bool, str]]:
    """{idx: (label, verified, how)} using the agreement + neighbour-offset rule."""
    cand: dict[int, tuple[Optional[str], Optional[str]]] = {i: (r.get("printed_page_label_tag"), r.get("printed_page_label_crop")) for i, r in raw.items()}
    offsets: dict[int, set[int]] = {}
    for i, (a, b) in cand.items():
        offsets[i] = {v - i for v in (_as_int(a), _as_int(b)) if v is not None}
    out = {}
    for i, (a, b) in cand.items():
        neigh = set()
        for j in range(i - 3, i + 4):
            if j != i and j in offsets:
                neigh |= offsets[j]
        agreed = a is not None and a == b
        if agreed:
            ai = _as_int(a)
            consistent = ai is not None and (ai - i) in neigh
            out[i] = (a, consistent, "tag+crop agree" + (", neighbour-consistent" if consistent else ", no neighbour support"))
            continue
        # disagreement or single read: keep a candidate that neighbours support; never verify
        for lab, src in ((b, "crop"), (a, "tag")):
            li = _as_int(lab)
            if lab is not None and li is not None and (li - i) in neigh:
                out[i] = (lab, False, f"{src} only (neighbour-consistent, reads disagree)"); break
        else:
            lab = b if b is not None else a
            out[i] = (lab, False, "single unconfirmed read" if lab else "no printed page number read")
    return out


def _candidate_quality(r: Optional[dict]) -> Optional[tuple[float, bool, list[str], dict]]:
    if r is None or r.get("error") or not (r.get("text") or "").strip():
        return None
    fin = r.get("finish_reason")
    fin = "length" if str(fin).upper() in ("LENGTH", "MAX_TOKENS") else fin
    m = quality_metrics(r.get("text") or "")
    score, review, reasons = quality_score(m, finish_reason=fin, degenerate=r.get("degenerate", False))
    return score, review, reasons, m


def select_candidate(q: Optional[dict], g: Optional[dict]) -> tuple[str, str]:
    """Return ("qwen"|"gemini"|"none", reason). Prefer Gemini when present, error-free and quality >= Qwen's;
    except when Gemini is much shorter than a high-quality Qwen page (likely truncated/skipped content)."""
    qq, gq = _candidate_quality(q), _candidate_quality(g)
    if gq is None and qq is None:
        return ("qwen" if q is not None else "gemini" if g is not None else "none"), "no usable candidate"
    if gq is None:
        why = "gemini missing" if g is None else ("gemini error" if g.get("error") else "gemini empty")
        return "qwen", why
    if qq is None:
        return "gemini", "qwen missing/failed"
    lq, lg = len(q.get("text") or ""), len(g.get("text") or "")
    if lg < 0.6 * lq and qq[0] >= 0.9 and lq > 300:
        return "qwen", f"gemini much shorter ({lg} vs {lq} chars)"
    if gq[0] >= qq[0]:
        return "gemini", f"quality gemini {gq[0]} >= qwen {qq[0]}"
    return "qwen", f"quality qwen {qq[0]} > gemini {gq[0]}"


def assemble_book_pages(root: Path, job: dict, use_gemini: bool = False) -> int:
    """Gemini pages are only a comparison candidate (OCR table); they replace Qwen text only with
    ``use_gemini=True`` (off by default: span corpus v1 is frozen on the Qwen3-VL text)."""
    from otter.textbooks.gemini_import import load_imported
    parsed = root / "data/textbooks/parsed" / job["book_id"]
    raw_dir = parsed / "ocr_raw"
    raw = {}
    for f in sorted(raw_dir.glob("p*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        raw[r["pdf_page_index"]] = r
    gem = load_imported(parsed)
    sha = job.get("sha256") or _sha(Path(job["local_file"]))
    labels = resolve_page_labels(raw)
    lines = []
    for idx in sorted(set(raw) | set(gem)):
        q, g = raw.get(idx), gem.get(idx)
        choice, why = select_candidate(q, g) if use_gemini else (("qwen", "qwen3-vl text (gemini comparison only)") if q is not None else ("none", "no qwen page"))
        r = g if choice == "gemini" else (q if q is not None else g)
        text = r.get("text") or ""
        m = quality_metrics(text)
        fin = r.get("finish_reason")
        fin = "length" if str(fin).upper() in ("LENGTH", "MAX_TOKENS") else fin
        score, review, reasons = quality_score(m, finish_reason=fin, degenerate=r.get("degenerate", False))
        if r.get("error"):
            status, review = "ocr_failed", True; reasons.append("ocr_error")
        elif m["n_chars"] < 20:
            status = "empty_page"
        else:
            status = "ok"
        if r.get("trimmed_repetition"):
            reasons.append("repetition_tail_trimmed"); review = True
        label, verified, how = labels.get(idx, (None, False, "no read"))
        g_label = g.get("printed_page_label") if g else None
        if label is None and g_label:
            label, verified, how = str(g_label), False, "gemini read only (unconfirmed)"
        if choice == "gemini":
            ext_version = f"gemini-import:{g.get('prompt_version')}+{g.get('model')}"
            sel = f"scanned PDF (no text layer); Gemini OCR {g.get('model')} selected: {why}"
        else:
            ext_version = f"{r.get('ocr_version')}+{r.get('model')}+{r.get('extractor_version')}"
            sel = (f"scanned PDF (no text layer); {r.get('model')} attempt {r.get('attempts')}, image {r.get('image_method')} "
                   f"{r.get('image_width')}x{r.get('image_height')}; {why}")
        rec = PageRecord(
            book_id=job["book_id"], span_id=page_span_id(job["book_id"], idx), pdf_page_index=idx, source_page_index=idx,
            printed_page_label=label, printed_page_verified=verified, section_id=None, text=text,
            extraction_method="vlm_ocr", extractor_version=ext_version,
            quality_score=score, requires_manual_review=review, selection_reason=sel,
            source_file_sha256=sha, status=status)
        row = rec.model_dump()
        # backward-compatible fields of the earlier parser output
        row.update({"start_offset": 0, "end_offset": len(text), "extraction_confidence": score,
                    "provenance_status": "candidate_extraction" if status == "ok" else "manual_review_needed",
                    "page_label_evidence": how,
                    "printed_page_label_reads": {"tag": q.get("printed_page_label_tag") if q else None,
                                                 "crop": q.get("printed_page_label_crop") if q else None, "gemini": g_label},
                    "quality_reasons": reasons, "ocr_metrics": m,
                    "image_path": (q or {}).get("image_path") or f"data/textbooks/parsed/{job['book_id']}/images/p{idx:04d}.png",
                    "ocr_source": choice, "ocr_model": r.get("model"),
                    "ocr_candidates": {"qwen": bool(q), "gemini": bool(g)},
                    "ocr_latency_s": r.get("latency_s"), "ocr_finish_reason": r.get("finish_reason"),
                    "ocr_completion_tokens": r.get("completion_tokens")})
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    tmp = parsed / "pages.jsonl.tmp"
    tmp.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    tmp.replace(parsed / "pages.jsonl")
    return len(lines)
