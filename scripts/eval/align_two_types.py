"""Align data/2_types_questions/OTTER_2types_400.csv to the ontology and, through it, to textbook spans.

For every item: retrieve candidate spans (hybrid BM25 + bge-m3, same store as OTTER) and ontology items (concepts, facts, rules),
then ask Qwen3-32B (:8005) whether one candidate span settles the statement / contains the blank's answer. The CSV is never edited;
output is a NEW file with candidate gold spans, matched ontology item ids and a consistency flag against the stated key.
Status is 'auto_aligned' only; nothing here is expert- or teacher-validated.

  .venv/bin/python scripts/eval/align_two_types.py [--limit N] [--workers 3]
"""
import argparse, csv, json, re, sys, concurrent.futures as cf, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.agents.systems import default_store  # noqa: E402

SUBJ = {"Hóa học": "chemistry", "Sinh học": "biology", "Địa lí": "geography", "Lịch sử": "history", "Ngữ văn": "literature", "Vật lí": "physics"}
URL, MODEL = "http://127.0.0.1:8005/v1/chat/completions", "Qwen/Qwen3-32B"
NEG = re.compile(r"^\s*Không đúng khi cho rằng\s*:\s*", re.I)


def statement(r):
    if r["type"] == "fill_blank":
        return re.sub(r"_{2,}", r["answer"], r["question"])
    return NEG.sub("", r["question"])


def llm(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", t, re.S)
    return json.loads(m.group(0)) if m else {}


def work(r, store):
    st = statement(r)
    q = r["question"] if r["type"] == "fill_blank" else st
    sub, gr = SUBJ[r["subject"]], int(r["grade"])
    spans = [s for s, _ in store.search_spans(q + " " + (r["answer"] if r["type"] == "fill_blank" else ""), subject=sub, grade=gr, top_k=6)]
    kn = store.search_knowledge(q, subject=sub, grade=gr, kinds=("concept", "fact", "rule"), top_k=5)
    ctx = "\n".join(f"[{s.span_id}] {s.text[:600]}" for s in spans)
    if r["type"] == "true_false":
        task = (f"Phát biểu: \"{st}\"\n"
                "Chọn một đoạn trong danh sách đủ thông tin để quyết định phát biểu này đúng hay sai theo sách giáo khoa, rồi tự kết luận.")
    else:
        task = (f"Câu điền khuyết: \"{r['question']}\"\n"
                "Chọn một đoạn trong danh sách chứa thuật ngữ điền vào chỗ trống, và nêu thuật ngữ đó.")
    p = (f"Môn {r['subject']} lớp {r['grade']}.\n{task}\n\nCác đoạn sách:\n{ctx}\n\n"
         'Chỉ trả về JSON: {"span_id": "id đoạn tốt nhất hoặc null", "verdict": "true|false|not_enough (với câu điền khuyết: found|not_enough)", "term": "thuật ngữ điền (chỉ câu điền khuyết)", "quote": "trích nguyên văn ngắn từ đoạn đó", "note": "tối đa 15 từ"}.')
    try:
        j = llm(p)
    except Exception as e:
        j = {"error": str(e)[:120]}
    sid, v = j.get("span_id"), j.get("verdict")
    valid = sid in {s.span_id for s in spans}
    quote_ok = bool(valid and j.get("quote") and re.sub(r"\s+", " ", j["quote"]).strip().lower() in
                    re.sub(r"\s+", " ", next(s.text for s in spans if s.span_id == sid)).lower())
    if r["type"] == "true_false":
        stmt_true = (r["answer"] == "TRUE") != bool(NEG.match(r["question"]))
        consistent = None if v not in ("true", "false") else ((v == "true") == stmt_true)
    else:
        def nn(x):
            import unicodedata
            return re.sub(r"[^\w\s]", "", unicodedata.normalize("NFC", str(x or "")).lower()).replace("axit", "acid").strip()
        t, g = nn(j.get("term")), nn(r["answer"])
        consistent = None if v not in ("found", "true") else bool(t and g and (t == g or t in g or g in t))
    status = ("auto_aligned" if (valid and quote_ok and consistent is True) else
              "key_conflict" if (valid and consistent is False) else "not_aligned")
    return {"id": r["id"], "type": r["type"], "subject": r["subject"], "grade": r["grade"], "pair_id": r["pair_id"],
            "candidate_span_ids": [s.span_id for s in spans], "best_span_id": sid if valid else None, "llm_verdict": v,
            "quote": j.get("quote"), "quote_found_in_span": quote_ok, "consistent_with_key": consistent,
            "ontology_items": [{"kind": h.get("kind"), "id": h.get("id") or h.get("item_id"), "label": str(h.get("label_or_text"))[:140],
                                "source_span_ids": list(h.get("source_span_ids") or [])[:3], "support": h.get("source_support")} for h in kn],
            "alignment_status": status, "note": j.get("note") or j.get("error")}


def load_rows(src):
    """Return rows in the schema of OTTER_2types_400.csv (type, question, answer, pair_id ...)."""
    if src == "400":
        return list(csv.DictReader(open(ROOT / "data/2_types_questions/OTTER_2types_400.csv", encoding="utf-8-sig")))
    out = []
    for r in csv.DictReader(open(ROOT / "data/2_types_questions/OTTER_EXTRA_200.csv", encoding="utf-8-sig")):
        qt = r["question_type"]
        d = {"id": r["id"], "subject": r["subject"], "grade": r["grade"], "pair_id": r["fact_group_id"], "src_type": qt}
        if qt == "fill_in_blank":
            d.update(type="fill_blank", question=r["question"], answer=r["answer"] or r["answer_text"])
        elif qt == "true_false":
            d.update(type="true_false", question=r["question"], answer="TRUE" if r["answer"].strip().lower() in ("đúng", "true") else "FALSE")
        else:   # multiple_choice: align the sentence completed with the correct option (a true statement)
            ans = r["answer_text"]
            q = r["question"]
            d.update(type="true_false", question=(re.sub(r"_{2,}", ans, q) if re.search(r"_{2,}", q) else f"{q} {ans}"), answer="TRUE")
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--src", choices=["400", "extra200"], default="400")
    a = ap.parse_args()
    rows = load_rows(a.src)
    if a.limit:
        rows = rows[:a.limit]
    store = default_store()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(lambda r: work(r, store), rows))
    dst = ROOT / "data/2_types_questions" / ({"400": "OTTER_2types_400_aligned.jsonl", "extra200": "OTTER_EXTRA_200_aligned.jsonl"}[a.src] if not a.limit else "OTTER_2types_aligned_test.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    import collections
    print(len(out), dict(collections.Counter(o["alignment_status"] for o in out)))


if __name__ == "__main__":
    main()
