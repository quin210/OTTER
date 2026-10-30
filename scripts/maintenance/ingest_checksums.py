"""SHA-256 manifest of the saved ingest results (summaries, changelogs, conflict lists, eval JSON, tables, the new ontology files, edition copies).
  .venv/bin/python scripts/maintenance/ingest_checksums.py  -> runs/knowledge_ingest/MANIFEST.sha256
(scripts/eval/archive_runs.py is built for answer/score runs and copies code tarballs; it does not fit these outputs.)"""
import hashlib
from pathlib import Path

KI = Path(__file__).resolve().parents[2] / "runs/knowledge_ingest"
SKIP_DIRS = ("cache_main", "cache_ingest_only", "cache", "__pycache__")
lines = []
for p in sorted(KI.rglob("*")):
    if not p.is_file() or p.name == "MANIFEST.sha256" or any(part.startswith(("cache", ".")) or part in SKIP_DIRS or part.endswith(".cache") for part in p.relative_to(KI).parts[:-1]):
        continue
    if p.suffix in (".tmp", ".building"):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    lines.append(f"{h.hexdigest()}  {p.relative_to(KI)}")
(KI / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "files")
