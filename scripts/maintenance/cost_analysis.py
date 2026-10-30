"""Cost of keeping knowledge fresh: ours vs rebuilding (counted, partly estimated; see README for what is measured)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.builder.ingest import load_books, load_sections, load_spans
from otter.builder.segment import segment_book

KM = ROOT / "runs/knowledge_maintenance"
db = ROOT / "data/textbooks/ontology.sqlite3"
books = {b: i for b, i in load_books(db, None, ROOT / "data/textbooks/registry").items() if b.startswith("nxbgd_")}
spans_by_book = load_spans(db, list(books), ROOT / "data/textbooks/parsed")
sections = load_sections(db, list(books), ROOT / "data/textbooks/parsed")
chunks = []
for b, info in sorted(books.items()):
    subj = info.subject
    chunks += segment_book(b, spans_by_book[b], sections, subject=subj, grade=info.grade, book_title=info.title)
docs = [json.loads(l) for l in (ROOT / "runs/baselines/corpus/v1core_g2400.jsonl").read_text().splitlines() if l.strip()]
kept = {b: [s.span_id for s in spans_by_book[b] if len(s.text.strip()) >= 20] for b in books}
doc_spans = []
for d in docs:
    first, n = d["doc_id"].rsplit("+", 1) if "+" in d["doc_id"] else (d["doc_id"], "0")
    ids = kept[d["book_id"]]
    k = ids.index(first) if first in ids else 0
    doc_spans.append(set(ids[k:k + int(n) + 1]))
build = json.loads((ROOT / "data/ontology/build_report.json").read_text())
idx = {r["system"]: r for r in __import__("csv").DictReader(open(ROOT / "runs/tables/index_cost.csv"))}
out = {"n_chunks": len(chunks), "n_docs": len(docs), "extraction_calls_full": build["extraction"]["llm_chunks"],
       "extraction_tokens_full": build["extraction"]["prompt_tokens"] + build["extraction"]["completion_tokens"],
       "verify_calls_full": build["verification"]["llm_calls"] + build["verification"]["cached_batches"],
       "verify_tokens_full": build["verification"]["prompt_tokens"] + build["verification"]["completion_tokens"],
       "graph_index_tokens": {s: int(idx[s]["prompt_tokens"]) + int(idx[s]["completion_tokens"]) for s in ("hipporag", "lightrag") if s in idx},
       "seeds": {}}
for s in (0, 1, 2):
    ch = json.loads((KM / "editions" / f"seed{s}" / "changes.json").read_text())
    changed = set(ch["deleted"]) | set(ch["edited"])
    summ = json.loads((KM / "versions" / f"seed{s}.summary.json").read_text())
    tc = [c for c in chunks if any(sp.span_id in changed for sp in c.spans)]
    dt = [d for d in doc_spans if d & changed]
    c = summ["cost"]
    out["seeds"][s] = {"changed_spans": len(changed), "chunks_touched": len(tc), "chunk_frac": len(tc) / len(chunks),
                       "docs_touched": len(dt), "doc_frac": len(dt) / len(docs),
                       "ours_llm_calls": c["llm_calls"], "ours_tokens": c["prompt_tokens"] + c["completion_tokens"],
                       "items_rechecked": c["items_rechecked"], "items_total": c["items_total"],
                       "batches_full_revalidation": c["batches_full_revalidation"],
                       "est_tokens_full_revalidation": round((c["prompt_tokens"] + c["completion_tokens"]) *
                                                              c["prompt_chars_full_revalidation"] / max(c["prompt_chars_planned"], 1)),
                       "wall_s": summ.get("wall_s")}
(KM / "reports" / "cost.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
