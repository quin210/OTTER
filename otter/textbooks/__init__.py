"""Textbook discovery, acquisition, parsing, and provenance support."""

from .catalog import Book, Catalog, normalize_subject
from .parser import parse_book
from .provenance import OntologyStore, CitationResolver

__all__ = ["Book", "Catalog", "normalize_subject", "parse_book", "OntologyStore", "CitationResolver"]
