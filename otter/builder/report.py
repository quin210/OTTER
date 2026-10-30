"""Builder step 7: build report (data/ontology/build_report.{json,md}).

Counts by subject / book / item type, rejection reasons, support distribution, strict-executable rules,
validation notes, canonicalisation merges, extraction + verification cost (tokens, wall time, throughput).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from otter.builder.normalize import Ontology
from otter.storage.writer import is_strict_executable

KINDS = ("concept", "relation", "fact", "rule", "objective")


def compute_report(onto: Ontology, *, books: dict, chunk_stats: dict, extraction: dict, verification: dict,
                   timings: dict, config: dict, persist_info: dict, build_id: str) -> dict[str, Any]:
    by_subject: dict[str, Counter] = defaultdict(Counter)
    by_book: dict[str, Counter] = defaultdict(Counter)
    support: dict[str, Counter] = defaultdict(Counter)
    support_by_subject: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    status: dict[str, Counter] = defaultdict(Counter)
    notes: Counter = Counter()
    for kind, iid, obj in onto.items():
        subj = getattr(obj, "subject", None) or (onto.concepts[obj.subject_id].subject
                                                 if kind == "relation" and obj.subject_id in onto.concepts else "other")
        by_subject[subj][kind] += 1
        for b in onto.item_books.get(iid, []):
            by_book[b][kind] += 1
        v = getattr(obj, "validation", None)
        if v is not None:
            support[kind][v.source_support] += 1
            support_by_subject[subj][kind][v.source_support] += 1
            for n in v.notes:
                notes[n.split(":")[0]] += 1
        if hasattr(obj, "status"):
            status[kind][obj.status] += 1
    rule_types = Counter(r.rule_type for r in onto.rules.values())
    strict = sum(1 for r in onto.rules.values() if is_strict_executable(r))
    strict_by_subject = Counter(r.subject for r in onto.rules.values() if is_strict_executable(r))
    rejected = Counter((r.get("kind"), r.get("reason")) for r in onto.rejected)
    rej_by_kind: dict[str, dict[str, int]] = defaultdict(dict)
    for (k, reason), n in rejected.items():
        rej_by_kind[str(k)][str(reason)] = n
    merges = Counter(m.get("method") for m in onto.merges)
    concept_types = Counter(c.concept_type for c in onto.concepts.values())
    rel_types = Counter(r.relation_type for r in onto.relations.values())
    total = {k: sum(c[k] for c in by_subject.values()) for k in KINDS}
    return {
        "build_id": build_id,
        "config": config,
        "books": {b: {"subject": info.subject, "grade": info.grade, "title": info.title,
                      **chunk_stats.get(b, {}), "items": dict(by_book.get(b, {}))} for b, info in books.items()},
        "totals": total | {"pedagogical_links": len(onto.links), "strict_executable_rules": strict,
                           "rejected": len(onto.rejected), "merges": len(onto.merges)},
        "by_subject": {s: dict(c) for s, c in sorted(by_subject.items())},
        "support": {k: dict(c) for k, c in support.items()},
        "support_by_subject": {s: {k: dict(c) for k, c in d.items()} for s, d in sorted(support_by_subject.items())},
        "status": {k: dict(c) for k, c in status.items()},
        "rule_types": dict(rule_types),
        "strict_executable_by_subject": dict(strict_by_subject),
        "concept_types": dict(concept_types.most_common()),
        "relation_types": dict(rel_types.most_common()),
        "validation_notes": dict(notes.most_common()),
        "rejections": dict(rej_by_kind),
        "merges": dict(merges),
        "extraction": extraction,
        "verification": verification,
        "timings_s": timings,
        "persist": persist_info,
    }


def _table(rows: list[list[Any]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(rep: dict[str, Any]) -> str:
    L = [f"# OTTER ontology build report — `{rep['build_id']}`", ""]
    c = rep["config"]
    L.append(f"Extractor: `{c.get('extractor')}` · Support verifier: `{c.get('verifier')}` · "
             f"Embeddings: `{(rep['persist'].get('embeddings') or {}).get('model')}`")
    L.append(f"Corpus: `{c.get('corpus_version')}` · spans_sha256 `{c.get('spans_sha256')}`  ")
    L.append(f"Prompt: `{c.get('prompt_version')}` · Verify prompt: `{c.get('verify_version')}` · "
             f"Books: {len(rep['books'])}")
    L.append("")
    t = rep["totals"]
    L.append("## Totals")
    L.append(_table([[k, v] for k, v in t.items()], ["item", "count"]))
    L.append("")
    L.append("## By subject")
    subs = rep["by_subject"]
    L.append(_table([[s, *[d.get(k, 0) for k in KINDS], rep["strict_executable_by_subject"].get(s, 0)]
                     for s, d in subs.items()], ["subject", *KINDS, "strict rules"]))
    L.append("")
    L.append("## By book")
    rows = []
    for b, d in sorted(rep["books"].items()):
        it = d.get("items", {})
        rows.append([b, d.get("subject"), d.get("grade"), d.get("spans", 0), d.get("chunks", 0),
                     *[it.get(k, 0) for k in KINDS]])
    L.append(_table(rows, ["book", "subject", "grade", "spans", "chunks", *KINDS]))
    L.append("")
    L.append("## Source-support distribution (separate check; heuristic + LLM, see validate.py)")
    labels = ["supported", "partially_supported", "unsupported", "needs_review"]
    L.append(_table([[k, *[d.get(x, 0) for x in labels]] for k, d in rep["support"].items()], ["kind", *labels]))
    L.append("")
    L.append("## Rejections")
    rows = [[k, r, n] for k, d in rep["rejections"].items() for r, n in sorted(d.items(), key=lambda x: -x[1])]
    L.append(_table(rows, ["kind", "reason", "count"]) if rows else "none")
    L.append("")
    L.append("## Validation notes")
    L.append(_table([[k, v] for k, v in rep["validation_notes"].items()], ["note", "count"]))
    L.append("")
    L.append(f"Rule types: {rep['rule_types']}  ")
    L.append(f"Concept merges: {rep['merges']}")
    L.append("")
    L.append("## Cost")
    e, v, tm = rep["extraction"], rep["verification"], rep["timings_s"]
    L.append(_table([
        ["extraction", e.get("llm_chunks", 0), e.get("cached_chunks", 0), e.get("failed_chunks", 0),
         e.get("prompt_tokens", 0), e.get("completion_tokens", 0), tm.get("extract", 0)],
        ["support check", v.get("llm_calls", 0), v.get("cached_batches", 0), v.get("failed_batches", 0),
         v.get("prompt_tokens", 0), v.get("completion_tokens", 0), tm.get("validate", 0)],
    ], ["stage", "LLM calls", "cached", "failed", "prompt tok", "completion tok", "wall s"]))
    L.append("")
    L.append(f"Timings (s): {tm}")
    if e.get("throughput"):
        L.append(f"Extraction throughput: {e['throughput']}")
    L.append("")
    L.append("Notes: token counts are from the vLLM usage field of calls made in this run (cached chunks report "
             "the tokens of their original call separately in `extraction.cached_*`). Support labels are "
             "automatic (lexical overlap + entailment by a different model) and are NOT human verification; "
             "see annotation/ontology_audit/ for the human audit.")
    return "\n".join(L) + "\n"


def write_report(rep: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pj, pm = out_dir / "build_report.json", out_dir / "build_report.md"
    pj.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pm.write_text(render_markdown(rep), encoding="utf-8")
    return pj, pm
