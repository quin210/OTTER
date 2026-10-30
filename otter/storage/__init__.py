"""WS2 storage: read-only SqliteOntologyStore (OntologyReader), writer, fixtures."""
from otter.storage.sqlite_store import SqliteOntologyStore
from otter.storage.writer import OntologyWriter, is_strict_executable

__all__ = ["SqliteOntologyStore", "OntologyWriter", "is_strict_executable"]
