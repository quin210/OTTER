"""Reference build: all 18 core books rebuilt from the extraction CACHE (no extraction calls), same code as `otter.builder build`.
Used as the apples-to-apples target of the leave-one-book-out comparison, because the cache does not reproduce
data/ontology/ontology.sqlite3 item-for-item (see docs/KNOWLEDGE_MAINTENANCE.md)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OTTER_RUN_TAG", "knowledge_ingest__reference")
from otter.builder.pipeline import BuildConfig, build

d = ROOT / "runs/knowledge_ingest/reference_full"
d.mkdir(parents=True, exist_ok=True)
books = sorted(p.name for p in (ROOT / "data/textbooks/parsed").glob("nxbgd_*") if (p / "spans.jsonl").exists())
cfg = BuildConfig(db=ROOT / "data/textbooks/ontology.sqlite3", out=d / "reference.sqlite3", export_dir=d / "export",
                  cache_dir=ROOT / "runs/knowledge_ingest/cache_main", books=books, backbone="main", verifier="small", workers=2,
                  verify_workers=3, report_dir=d / "report", require_verifier=True)
rep = build(cfg)
(d / "summary.json").write_text(json.dumps({k: rep[k] for k in ("build_id", "totals", "support", "extraction", "verification")}, ensure_ascii=False, indent=1))
print(json.dumps({"totals": rep["totals"], "extraction": rep["extraction"]["llm_chunks"], "verification": rep["verification"]}, indent=1))
