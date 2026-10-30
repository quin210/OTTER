from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

def _stable(prefix: str, *parts: str) -> str:
    return prefix + hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]

def _pages(path: Path):
    suffix=path.suffix.lower()
    if suffix in {".txt", ".md"}:
        yield 0, path.read_text(encoding="utf-8"), "native_text"
        return
    if suffix == ".pdf":
        if shutil.which("pdftotext") and shutil.which("pdfinfo"):
            meta=subprocess.run(["pdfinfo",str(path)],capture_output=True,text=True,check=True).stdout
            encrypted=re.search(r"^Encrypted:\s+(\w+)",meta,re.M)
            if encrypted and encrypted.group(1).lower()=="yes": raise ValueError("Encrypted PDF is unsupported; do not bypass DRM")
            pages_match=re.search(r"^Pages:\s+(\d+)",meta,re.M)
            if not pages_match: raise ValueError("Could not verify PDF page count")
            page_count=int(pages_match.group(1))
            proc=subprocess.run(["pdftotext","-layout","-enc","UTF-8",str(path),"-"],capture_output=True,check=True)
            extracted=proc.stdout.decode("utf-8","replace").split("\f")
            if len(extracted)>page_count and not extracted[-1].strip(): extracted=extracted[:-1]
            extracted=(extracted+[""]*page_count)[:page_count]
            any_text=any(x.strip() for x in extracted)
            method="native_pdf_text" if any_text else "scanned_pdf_no_text"
            for i,text in enumerate(extracted): yield i,text,method
            return
        try: from pypdf import PdfReader
        except ImportError as e: raise RuntimeError("PDF parsing requires pdftotext/pdfinfo or optional dependency pypdf") from e
        reader=PdfReader(str(path), strict=True)
        if reader.is_encrypted: raise ValueError("Encrypted PDF is unsupported; do not bypass DRM")
        for i,page in enumerate(reader.pages): yield i, page.extract_text() or "", "native_pdf_text"
        return
    raise ValueError(f"Unsupported input type: {suffix}")

def _printed_label(text: str) -> tuple[str|None,bool]:
    # A page label is recorded only when explicitly printed in extracted text;
    # matching a digit alone is not enough to establish the printed page.
    matches=re.findall(r"(?im)^\s*(?:trang|page)\s+(\d{1,4})\s*$", text)
    # Extracted page text is a candidate locator. Only a reviewer who compares
    # against the rendered book may flip printed_page_verified to true.
    return (matches[-1], False) if matches else (None, False)

def _heading(text: str) -> str|None:
    for line in text.splitlines():
        line=" ".join(line.split())
        if line and (re.match(r"^(?:CHƯƠNG|BÀI|Chương|Bài)\s+\S+",line) or line.startswith("#")):
            return line.lstrip("# ")[:240]
    return None

def parse_book(book: dict, source_file: Path, output_dir: Path, sha256: str) -> tuple[int,int]:
    """Write page- and span-level JSONL with explicit unknown printed pages."""
    output_dir.mkdir(parents=True,exist_ok=True)
    pages=[]; sections={}; current_section=None
    for index,text,method in _pages(source_file):
        heading=_heading(text)
        if heading:
            current_section=_stable("sec_",book["book_id"],sha256,heading)
            sections.setdefault(current_section,{"book_id":book["book_id"],"section_id":current_section,"title":heading,"lesson":heading,"pdf_page_start":index if method in {"native_pdf_text","scanned_pdf_no_text"} else None,"pdf_page_end":index if method in {"native_pdf_text","scanned_pdf_no_text"} else None,"source_page_start":index,"source_page_end":index,"printed_page_start":None,"printed_page_end":None,"source_span_ids":[]})
        if current_section and current_section in sections:
            sections[current_section]["source_page_end"]=index
            if method in {"native_pdf_text","scanned_pdf_no_text"}: sections[current_section]["pdf_page_end"]=index
        span_id=_stable("span_",book["book_id"],sha256,str(index),text)
        label,verified=_printed_label(text)
        if current_section: sections[current_section]["source_span_ids"].append(span_id)
        pages.append({"book_id":book["book_id"],"pdf_page_index":index if method in {"native_pdf_text","scanned_pdf_no_text"} else None,"source_page_index":index,"printed_page_label":label,"printed_page_verified":verified,"section_id":current_section,"span_id":span_id,"text":text,"start_offset":0,"end_offset":len(text),"extraction_method":method,"extraction_confidence":1.0 if text.strip() and method=="native_pdf_text" else None,"source_file_sha256":sha256,"provenance_status":"candidate_extraction" if text.strip() else "manual_review_needed"})
    for name,rows in (("pages.jsonl",pages),("sections.jsonl",sections.values())):
        (output_dir/name).write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in rows),encoding="utf-8")
    return len(pages),len(sections)
