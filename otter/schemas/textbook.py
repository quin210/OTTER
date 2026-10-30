"""Textbook provenance contracts: pages, structural nodes and source spans.

Shared by the OCR/ingestion pipeline (producer) and every downstream module
(builder, tools, citation resolver, evaluation). Field names are a contract;
add optional fields rather than renaming.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ExtractionMethod = Literal["native_text", "vlm_ocr", "tesseract_ocr", "paddle_ocr", "fixture", "manual"]
NodeLevel = Literal["book", "chapter", "lesson", "section", "subsection"]
PedagogicalType = Literal[
    "objective", "definition", "concept_explanation", "principle", "law", "formula",
    "condition", "exception", "process", "example", "worked_example", "exercise",
    "question", "activity", "summary", "figure_caption", "table", "reading_text", "other", "unknown",
]


class PageRecord(BaseModel):
    """One line of data/textbooks/parsed/<book_id>/pages.jsonl."""
    book_id: str
    span_id: str                                  # page-level span id: "<book_id>:p<pdf_page_index:04d>"
    pdf_page_index: int                           # 0-based index in the PDF file
    source_page_index: int
    printed_page_label: Optional[str] = None      # page number printed on the page, only if read from the page
    printed_page_verified: bool = False
    section_id: Optional[str] = None
    text: str = ""                                # Markdown transcription (headings, $LaTeX$, tables)
    extraction_method: ExtractionMethod = "native_text"
    extractor_version: str = ""
    quality_score: Optional[float] = None         # 0..1 heuristic (diacritics, garbage ratio, density)
    requires_manual_review: bool = False
    selection_reason: Optional[str] = None
    source_file_sha256: str
    status: str = "ok"                            # ok | scanned_pdf_no_text | ocr_failed | empty_page


class StructureNode(BaseModel):
    """Book -> chapter -> lesson -> section hierarchy (sections.jsonl)."""
    section_id: str                               # "<book_id>:ch02:l07:s03" style, stable per source sha
    book_id: str
    level: NodeLevel
    title: str
    parent_id: Optional[str] = None
    ordinal: Optional[str] = None                 # printed numbering: "CHƯƠNG II", "Bài 7", "I.", "1."
    pdf_page_start: Optional[int] = None
    pdf_page_end: Optional[int] = None
    printed_page_start: Optional[str] = None
    printed_page_end: Optional[str] = None
    heading_path: list[str] = Field(default_factory=list)
    link_status: Literal["resolved", "unresolved"] = "resolved"


class SourceSpan(BaseModel):
    """Paragraph-level evidence unit (TKU text). Citations always point here."""
    span_id: str                                  # "<book_id>:p<page:04d>:u<k:03d>"
    book_id: str
    section_id: Optional[str] = None
    heading_path: list[str] = Field(default_factory=list)
    pdf_page_index: int
    printed_page_label: Optional[str] = None
    printed_page_verified: bool = False
    start_offset: int                             # char offsets into the page text
    end_offset: int
    text: str                                     # exact extracted passage
    pedagogical_type: PedagogicalType = "unknown"
    source_file_sha256: str
    extractor_version: str = ""
    evidence_status: Literal["extracted", "ocr_unverified", "human_verified"] = "ocr_unverified"
