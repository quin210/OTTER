from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SUBJECTS = {
    "vật lí": "physics", "vật lý": "physics", "physics": "physics",
    "hóa học": "chemistry", "hoá học": "chemistry", "chemistry": "chemistry",
    "sinh học": "biology", "biology": "biology", "lịch sử": "history", "history": "history",
    "địa lí": "geography", "địa lý": "geography", "geography": "geography",
    "ngữ văn": "literature", "literature": "literature",
}

def normalize_subject(value: str) -> str:
    return SUBJECTS.get(" ".join(unicodedata.normalize("NFC", value).casefold().split()), "unknown")

@dataclass
class Book:
    book_id: str
    publisher: str
    source_platform: str
    curriculum: str
    series: str | None
    original_series_label: str | None
    subject_original: str
    subject_normalized: str
    grade: int
    title: str
    volume: str | None
    material_type: str
    edition_year: int | None
    item_page_url: str
    official_download_url: str | None = None
    download_permission_verified: bool = False
    access_status: str = "cataloged"
    download_status: str = "not_attempted"
    license_notes: str = "Portal states free for teaching/learning and not for commercial use; individual download permission unverified."
    discovered_at: str = "2026-10-23T00:00:00+08:00"
    last_checked_at: str = "2026-10-23T00:00:00+08:00"
    local_file: str | None = None
    sha256: str | None = None
    file_type: str | None = None
    local_file_scope: str | None = None
    page_count: int | None = None
    evidence_source_url: str | None = None

    @classmethod
    def observed(cls, *, title: str, subject: str, grade: int, item_url: str,
                 evidence_url: str, series: str | None = None, original_series_label: str | None = None, volume: str | None = None,
                 material_type: str = "student_textbook") -> "Book":
        digest = hashlib.sha256(item_url.encode()).hexdigest()[:8]
        observed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        slug = {"physics":"physics", "chemistry":"chemistry", "biology":"biology", "history":"history", "geography":"geography", "literature":"literature"}.get(normalize_subject(subject), "unknown")
        return cls(f"nxbgd_g{grade}_{slug}_{digest}", "Nhà xuất bản Giáo dục Việt Nam", "taphuan.nxbgd.vn", "GDPT 2018", series, original_series_label, subject, normalize_subject(subject), grade, title, volume, material_type, None, item_url, discovered_at=observed_at, last_checked_at=observed_at, evidence_source_url=evidence_url)

class Catalog:
    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.jsonl = registry_dir / "books.jsonl"

    def load(self) -> list[Book]:
        if not self.jsonl.exists(): return []
        rows=[]
        for line in self.jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row=json.loads(line); row.setdefault("original_series_label",None); row.setdefault("local_file_scope",None); rows.append(Book(**row))
        return rows

    def save(self, books: Iterable[Book]) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        rows = sorted({b.book_id: b for b in books}.values(), key=lambda b: (b.grade, b.subject_normalized, b.title))
        self.jsonl.write_text("".join(json.dumps(asdict(b), ensure_ascii=False, sort_keys=True)+"\n" for b in rows), encoding="utf-8")
        fields = list(Book.__dataclass_fields__)
        with (self.registry_dir / "books.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for b in rows: w.writerow(asdict(b))
        self.write_queue(rows)

    def write_queue(self, books: Iterable[Book]) -> None:
        fields = ["book_id","title","grade","subject","series","item_page_url","block_reason","expected_local_directory","instructions","status"]
        with (self.registry_dir / "download_queue.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for b in books:
                if not b.local_file and b.download_status not in {"downloaded", "checksum_verified"}:
                    w.writerow({"book_id":b.book_id,"title":b.title,"grade":b.grade,"subject":b.subject_original,"series":b.series or "","item_page_url":b.item_page_url,"block_reason":"permissions_unknown: detail page exposes electronic-book versions, but no authorized downloadable action could be verified during this run; use publisher's normal interface and download only if offered","expected_local_directory":f"data/textbooks/raw/{b.book_id}/","instructions":"Open verified official item page; use the standard ‘Tải xuống’ action only if shown and authorized. Otherwise request an authorized source copy.","status":"manual_download_possible"})
