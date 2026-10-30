"""Gold-evidence annotation for an MCQ benchmark without span ids (OTTER_MCQS_300).

Per question: hybrid retrieval (BM25 + bge-m3, RRF) over the subject's spans using the question
plus its keyed answer, then an LLM selects the candidate spans that support the keyed answer, or
`not_in_corpus`. LLM selections are *silver* until checked on the human sheet; the merged file
records `gold_source` accordingly.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from otter.evaluation.corpus import SpanCorpus
from otter.evaluation.mcq import LETTERS, MCQItem
from otter.evaluation.sheets import span_block, write_sheet
from otter.llm.interface import LLM, parse_json_loose

EVIDENCE_PROMPT_VERSION = "mcq_evidence_select_v1"
VERDICTS = ("supported", "partial", "not_in_corpus")

_PROMPT = """Bạn là chuyên gia giáo dục, đang gắn bằng chứng sách giáo khoa cho câu hỏi trắc nghiệm.
Cho câu hỏi, đáp án đúng và các ĐOẠN TRÍCH ứng viên từ sách giáo khoa (mã [S1], [S2], ...).
Hãy chọn TẤT CẢ và CHỈ những đoạn trích mà nội dung của chúng cho phép xác định đáp án đúng
(cùng nhau đủ để suy ra đáp án, hoặc mỗi đoạn cung cấp một phần cần thiết). Không chọn đoạn chỉ
cùng chủ đề nhưng không chứa thông tin cần thiết. Chỉ dựa vào đoạn trích, không dùng kiến thức ngoài.
- "supported": các đoạn đã chọn đủ để suy ra đáp án đúng.
- "partial": các đoạn đã chọn chỉ hỗ trợ một phần.
- "not_in_corpus": không đoạn nào chứa thông tin cần thiết (khi đó "supporting" = []).

Môn: {subject}, lớp {grade}
Câu hỏi: {question}
{options}
Đáp án đúng: {answer}. {answer_text}
{explanation}
ĐOẠN TRÍCH ỨNG VIÊN:
{cands}

Trả về JSON: {{"verdict": "supported|partial|not_in_corpus", "supporting": ["S.."], "rationale": "<ngắn gọn>"}}"""


def query_for(item: MCQItem) -> str:
    return f"{item.question} {item.answer_text}"


def retrieve_candidates(item: MCQItem, corpus: SpanCorpus, *, k_same_grade: int = 8, k_any: int = 4,
                        exclude_exercises: bool = True) -> list[dict[str, Any]]:
    q = query_for(item)
    hits = corpus.search(q, top_k=k_same_grade, domain=item.domain, grades=[item.grade] if item.grade else None,
                         exclude_exercises=exclude_exercises)
    seen = {h[0].span_id for h in hits}
    extra = [h for h in corpus.search(q, top_k=k_same_grade + k_any, domain=item.domain, exclude_exercises=exclude_exercises)
             if h[0].span_id not in seen][:k_any]
    out = []
    for rank, (s, score, rk) in enumerate(hits + extra, 1):
        out.append({"span_id": s.span_id, "rank": rank, "score": round(score, 5), **rk,
                    "same_grade": s.grade == item.grade})
    return out


def select_evidence(item: MCQItem, cands: list[dict], corpus: SpanCorpus, llm: LLM, *,
                    max_span_chars: int = 1200) -> dict[str, Any]:
    labels = [f"S{i + 1}" for i in range(len(cands))]
    lab2id = dict(zip(labels, [c["span_id"] for c in cands]))
    blocks = []
    for lab, c in zip(labels, cands):
        s = corpus.get(c["span_id"])
        hp = " › ".join(s.heading_path[-2:]) if s.heading_path else ""
        blocks.append(f"[{lab}] (lớp {s.grade}; {hp}) {s.text[:max_span_chars]}")
    opts = "\n".join(f"{k}. {item.options[k]}" for k in LETTERS)
    prompt = _PROMPT.format(subject=item.subject or item.domain, grade=item.grade, question=item.question,
                            options=opts, answer=item.answer, answer_text=item.answer_text,
                            explanation=f"Giải thích (của người ra đề): {item.explanation}\n" if item.explanation else "",
                            cands="\n\n".join(blocks) if blocks else "(không có)")
    schema = {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "supporting": {"type": "array", "items": {"type": "string", "enum": labels or ["S1"]}, "maxItems": 6},
        "rationale": {"type": "string"}}, "required": ["verdict", "supporting", "rationale"]}
    from otter.evaluation.hard_gen import chat_json
    obj, r = chat_json(llm, [{"role": "user", "content": prompt}], schema, max_tokens=400,
                       validate=lambda o: "verdict" in o)
    sup = obj.get("supporting") or []
    sup = sup if isinstance(sup, list) else [sup]
    import re as _re
    sel = [lab2id[m.group(1)] for x in sup if (m := _re.search(r"(S\d+)", str(x).upper())) and m.group(1) in lab2id]
    obj["verdict"] = str(obj.get("verdict", "")).strip().lower()
    verdict = obj.get("verdict") if obj.get("verdict") in VERDICTS else "not_in_corpus"
    if verdict != "not_in_corpus" and not sel:
        verdict = "not_in_corpus"
    if verdict == "not_in_corpus":
        sel = []
    return {"selected_span_ids": list(dict.fromkeys(sel)), "verdict": verdict, "rationale": obj.get("rationale", ""),
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}


def annotate(items: list[MCQItem], corpus: SpanCorpus, llm: LLM, out_path: Path, *, workers: int = 8,
             retrieval_cfg: Optional[dict] = None, corpus_version: Optional[str] = None) -> Path:
    """Resumable: question ids already in out_path (without error) are skipped."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add(r["id"])
    lock = threading.Lock()
    cfg = retrieval_cfg or {}

    def one(it: MCQItem):
        rec: dict[str, Any] = {"id": it.id, "domain": it.domain, "grade": it.grade, "question": it.question,
                               "answer": it.answer, "answer_text": it.answer_text, "model": llm.model,
                               "prompt_version": EVIDENCE_PROMPT_VERSION, "corpus_version": corpus_version,
                               "retrieval": {"method": "bm25+bge-m3 rrf" if corpus._embedder else "bm25", **cfg}}
        try:
            rec["candidates"] = retrieve_candidates(it, corpus, **cfg)
            rec.update(select_evidence(it, rec["candidates"], corpus, llm))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.bm25_scores("khởi tạo")                       # build shared indexes before threading
    for it in {(i.domain, i.grade) for i in items}:
        corpus._mask(it[0], [it[1]] if it[1] else None, None, True)
        corpus._mask(it[0], None, None, True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, [it for it in items if it.id not in done]))
    return out_path


def load_evidence(path: Path) -> dict[str, dict]:
    res = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                res[r["id"]] = r
    return res


def write_evidence_sheet(items: list[MCQItem], ev: dict[str, dict], corpus: SpanCorpus, out_dir: Path,
                         name: str = "mcq300_evidence_check", n_other: int = 4) -> tuple[Path, Path]:
    rows = []
    for it in items:
        e = ev.get(it.id)
        if e is None:
            continue
        sel = e.get("selected_span_ids", [])
        others = [c["span_id"] for c in e.get("candidates", []) if c["span_id"] not in sel][:n_other]
        rows.append({
            "id": it.id, "domain": it.domain, "question": it.question, "answer": f"{it.answer}. {it.answer_text}",
            "llm_verdict": e.get("verdict"), "llm_span_ids": sel, "other_candidate_ids": others,
            "__blocks": [
                {"kind": "text", "label": f"{it.subject} · lớp {it.grade} · {it.topic or ''}", "value": it.question},
                {"kind": "options", "label": "Options (keyed answer highlighted)", "value": it.options, "answer": it.answer},
                {"kind": "text", "label": "LLM verdict", "value": f"{e.get('verdict')} — {e.get('rationale', '')}"},
                span_block("LLM-selected evidence spans", [corpus.get(s) for s in sel]),
                span_block("Other retrieved candidates (not selected)", [corpus.get(s) for s in others], max_chars=600),
            ]})
    fields = [
        {"name": "evidence_ok", "label": "Selected spans support the key?", "type": "choice",
         "options": ["yes", "partial", "no", "none_selected_correctly"]},
        {"name": "in_corpus", "label": "Is the needed knowledge in the textbook?", "type": "choice",
         "options": ["yes", "no", "unsure"]},
        {"name": "answer_key_ok", "label": "Answer key correct?", "type": "choice", "options": ["yes", "no", "unsure"]},
        {"name": "corrected_span_ids", "label": "Corrected span ids (comma-sep., optional)", "type": "text", "required": False},
        {"name": "notes", "label": "Notes", "type": "text", "required": False},
    ]
    instr = ("For each question, check whether the LLM-selected textbook spans (or, if it said not_in_corpus, any "
             "of the other candidates) support the keyed answer. Use only the span text. If a better span among the "
             "'other candidates' supports the key, paste its span id into 'Corrected span ids'. Mark 'Answer key "
             "correct?' = no if the keyed option is wrong according to the textbook. Silver labels: "
             f"model + prompt `{EVIDENCE_PROMPT_VERSION}` (see benchmarks/mcq300_evidence.jsonl).")
    return write_sheet(rows, context_cols=["id", "domain", "question", "answer", "llm_verdict", "llm_span_ids",
                                           "other_candidate_ids"], fields=fields, out_dir=out_dir, name=name,
                       title="MCQ300 gold-evidence check", instructions=instr)


def merge_into_bench(items: list[MCQItem], ev: dict[str, dict], out_path: Path,
                     human: Optional[dict[str, dict]] = None) -> dict[str, int]:
    """Write a copy of the bench with gold_span_ids (+ gold_source) from silver or human labels."""
    stats = {"human": 0, "silver": 0, "none": 0}
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            row = it.model_dump()
            h = (human or {}).get(it.id)
            e = ev.get(it.id, {})
            if h and h.get("evidence_ok") in ("yes", "partial", "no", "none_selected_correctly"):
                corr = [x.strip() for x in (h.get("corrected_span_ids") or "").split(",") if x.strip()]
                ids = corr or (e.get("selected_span_ids", []) if h["evidence_ok"] in ("yes", "partial") else [])
                row.update(gold_span_ids=ids, gold_source="human_checked", evidence_verdict_human=h["evidence_ok"],
                           in_corpus_human=h.get("in_corpus"), answer_key_ok_human=h.get("answer_key_ok"))
                stats["human"] += 1
            elif e:
                row.update(gold_span_ids=e.get("selected_span_ids", []), gold_source=f"llm_silver:{e.get('model')}",
                           corpus_version=e.get("corpus_version"),
                           evidence_verdict=e.get("verdict"))
                stats["silver"] += 1
            else:
                row.update(gold_span_ids=[], gold_source=None)
                stats["none"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats
