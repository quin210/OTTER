from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .catalog import Book, Catalog

def _log(registry: Path, book_id: str, status: str, reason: str | None=None, **extra) -> None:
    registry.mkdir(parents=True,exist_ok=True)
    row={"book_id":book_id,"status":status,"timestamp":datetime.now(timezone.utc).isoformat(),"reason":reason,**extra}
    with (registry/"acquisition_log.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")

def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def validate_file(path: Path) -> str:
    head=path.read_bytes()[:16]
    if head.startswith(b"%PDF-"): return "pdf"
    if head.startswith(b"PK\x03\x04") and path.suffix.lower()==".epub":
        try:
            with zipfile.ZipFile(path) as z:
                if "mimetype" in z.namelist() and z.read("mimetype").startswith(b"application/epub+zip"): return "epub"
        except (OSError,zipfile.BadZipFile): pass
    if path.suffix.lower() in {".txt",".md"}:
        path.read_text(encoding="utf-8"); return path.suffix.lower()[1:]
    raise ValueError("File signature is not a supported PDF, EPUB/container, UTF-8 TXT, or Markdown")

def import_manual(catalog: Catalog, book_id: str, source: Path, raw_root: Path, registry: Path) -> Book:
    books=catalog.load(); matches=[b for b in books if b.book_id==book_id]
    if not matches: raise ValueError("book_id is not in the verified catalog")
    kind=validate_file(source); destdir=raw_root/book_id; destdir.mkdir(parents=True,exist_ok=True)
    dest=destdir/source.name
    if source.resolve()!=dest.resolve():
        try: os.link(source,dest)
        except OSError:
            import shutil; shutil.copy2(source,dest)
    book=matches[0]; book.local_file=str(dest); book.sha256=file_hash(dest); book.file_type=kind; book.local_file_scope="sample_indicated_by_filename" if "mau" in source.stem.casefold() or "mẫu" in source.stem.casefold() else "unknown"; book.download_status="user_supplied"; book.access_status="user_supplied"; book.license_notes="File supplied locally by the user; source authorization and permitted downstream use have not been independently verified."
    catalog.save(books); _log(registry,book_id,"manual_import_checksum_verified",file=str(dest),sha256=book.sha256,file_type=kind)
    return book

def download_permitted(catalog: Catalog, book_id: str, raw_root: Path, registry: Path, max_bytes: int=80_000_000, delay_seconds: float=2.0) -> Book:
    books=catalog.load(); matches=[b for b in books if b.book_id==book_id]
    if not matches: raise ValueError("Unknown book_id")
    b=matches[0]
    if not b.official_download_url or not b.download_permission_verified:
        _log(registry,book_id,"rights_review","No verified standard download URL and permission; no request made")
        raise PermissionError("Download requires both an observed official standard download URL and verified download permission")
    if urlparse(b.official_download_url).hostname!="taphuan.nxbgd.vn": raise ValueError("Download host is not the official portal")
    time.sleep(max(delay_seconds,0))
    req=urllib.request.Request(b.official_download_url,headers={"User-Agent":"OTTER authorized acquisition"})
    destdir=raw_root/book_id; destdir.mkdir(parents=True,exist_ok=True)
    part=destdir/(book_id+".download.part")
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            if response.status!=200: raise OSError(f"download HTTP {response.status}")
            ctype=response.headers.get("Content-Type","").lower()
            total=0; first=b""
            with part.open("wb") as out:
                while chunk:=response.read(1024*1024):
                    total+=len(chunk)
                    if total>max_bytes: raise ValueError("download exceeded configured max_bytes")
                    if len(first)<16: first+=chunk[:16-len(first)]
                    out.write(chunk)
    except Exception:
        part.unlink(missing_ok=True); raise
    suffix=".pdf" if first.startswith(b"%PDF-") else ".epub" if first.startswith(b"PK\x03\x04") else ""
    if suffix==".epub":
        try:
            with zipfile.ZipFile(part) as z:
                if "mimetype" not in z.namelist() or not z.read("mimetype").startswith(b"application/epub+zip"): suffix=""
        except (OSError,zipfile.BadZipFile): suffix=""
    if not suffix or "text/html" in ctype:
        part.unlink(missing_ok=True)
        raise ValueError("Response is not a validated PDF/EPUB; probable error or login page")
    dest=destdir/(Path(urlparse(b.official_download_url).path).name or (book_id+suffix))
    if dest.suffix.lower()!=suffix: dest=dest.with_suffix(suffix)
    part.replace(dest)
    b.local_file=str(dest); b.sha256=file_hash(dest); b.file_type=suffix[1:]; b.download_status="checksum_verified"; b.access_status="downloaded"
    catalog.save(books); _log(registry,book_id,"downloaded_checksum_verified",file=str(dest),sha256=b.sha256,file_type=b.file_type,bytes=total)
    return b
