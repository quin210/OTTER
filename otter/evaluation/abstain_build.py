"""OTTER-Abstain construction: unanswerable MCQs (+ answerable controls) over the v1-core textbook corpus.

Unanswerable item types
* out_of_corpus      -- curriculum-plausible question about a lesson listed in the book's table of contents whose
                        pages are NOT part of the sample pages (TOC page > last page present in the corpus). Absence is
                        verified: hybrid (BM25 + bge-m3) top-10 spans + an LLM support check must find no span that
                        determines any option.
* missing_premise    -- an answerable multi-hop OTTER-Hard *candidate* (not an OTTER-Hard final item) with one gold span
                        removed from the retrievable view (`excluded_span_ids`); verified with the remaining gold spans
                        plus the top retrieved spans (excluded span removed) that no option is determined.
* counterfactual_guard -- a conditional rule ("nếu P thì Q") with a scenario where P is unspecified / not met, so Q (and
                        not-Q) cannot be inferred (never infer negation). Its control is the same question with P stated.
Answerable controls: out-of-corpus controls are single-span questions generated with the same style from lessons that are
present; missing-premise controls are the *same* item with its full evidence; guard controls are the same options with the
condition satisfied.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import (VERIFY_SCHEMA, _VERIFY_PROMPT, _bool, _letter, _status, assemble_options,
                                       chat_json, fmt_options, permute, span_block, check_verify)
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.text import char_ngrams, jaccard, norm, sha1
from otter.llm.interface import LLM

VERSION = "otter_abstain_v1"
ABSTAIN_OPTION_TEXT = "Không đủ căn cứ trong sách"
SUBJECTS = ["physics", "chemistry", "biology", "geography", "history", "literature"]
ABBR = {"physics": "PHY", "chemistry": "CHE", "biology": "BIO", "geography": "GEO", "history": "HIS", "literature": "LIT"}

# --------------------------------------------------------------------------------------------- TOC topics
_GENERIC_TITLE = re.compile(r"^(đọc|viết|nói và nghe|củng cố,? mở rộng|thực hành đọc|ôn tập|bảng |lời nói đầu|hướng dẫn|mục lục|"
                            r"nội dung|trang|bài|chương|chủ đề|phần|mục|stt|tt)\b", re.I)


def _cells(row: str) -> list[str]:
    return [re.sub(r"\*+", "", c).strip() for c in row.strip().strip("|").split("|")]


def toc_entries(book_spans: list[Span]) -> list[dict]:
    """Parse the book's printed table of contents (markdown table spans on the first pages)."""
    out, chapter = [], None
    for s in sorted(book_spans, key=lambda s: (s.pdf_page_index or 0, s.span_id)):
        if (s.pdf_page_index or 0) > 10 or s.text.count("|") < 6:
            continue
        for row in s.text.split("\n"):
            if "|" not in row or set(row.replace("|", "").strip()) <= set("-: "):
                continue
            cs = [c for c in _cells(row) if c]
            if len(cs) < 2:
                continue
            page = next((int(c) for c in reversed(cs) if re.fullmatch(r"\d{1,3}", c)), None)
            words = [c for c in cs if not re.fullmatch(r"\d{1,3}", c)]
            if page is None or not words:
                continue
            title = " – ".join(words)
            m = re.match(r"^(CHƯƠNG|CHỦ ĐỀ|PHẦN|CHUYÊN ĐỀ)\b", title, re.I)
            if m or (title.isupper() and len(title) > 12 and len(words) == 1 and not re.match(r"^bài", title, re.I)):
                chapter = title
                continue
            lesson = re.match(r"^(?:bài\s*(\d+)\s*[.:–-]?\s*)(.*)$", title, re.I)
            num = None
            if lesson and lesson.group(2):
                num, title = int(lesson.group(1)), lesson.group(2).strip()
            elif len(words) > 1 and re.fullmatch(r"(?:bài\s*)?\d{1,2}", words[0], re.I):
                num, title = int(re.sub(r"\D", "", words[0])), " – ".join(words[1:])
            if len(title) < 8 or _GENERIC_TITLE.match(title) or re.match(r"^(thực hành|ôn tập)\b", title, re.I) and not num:
                continue
            out.append({"num": num, "title": title.strip(" .–"), "page": page, "chapter": chapter})
    return out


def max_present_page(book_spans: list[Span]) -> int:
    pages = [int(s.printed_page_label) for s in book_spans if s.printed_page_label and str(s.printed_page_label).isdigit()]
    return max(pages) if pages else 0


def absent_topics(corpus: SpanCorpus) -> list[dict]:
    by_book: dict[str, list[Span]] = defaultdict(list)
    for s in corpus.spans:
        by_book[s.book_id].append(s)
    topics = []
    for b, sps in sorted(by_book.items()):
        mx = max_present_page(sps)
        if not mx:
            continue
        for e in toc_entries(sps):
            if e["page"] > mx + 1 and not re.match(r"^(thực hành|ôn tập)", e["title"], re.I):
                s0 = sps[0]
                topics.append({**e, "book_id": b, "domain": s0.domain, "grade": s0.grade, "book_title": s0.book_title,
                               "max_present_page": mx,
                               "topic_id": "top_" + sha1(f"{b}|{e['num']}|{e['title']}")[:10]})
    return topics


# --------------------------------------------------------------------------------------------- prompts
_SYS = "Bạn là giáo viên giàu kinh nghiệm ra đề trắc nghiệm theo sách giáo khoa Việt Nam (GDPT 2018). Chỉ trả về JSON hợp lệ."

_STYLE_RULES = """YÊU CẦU:
1. Câu hỏi tự đứng được như câu hỏi kiểm tra cho học sinh; TUYỆT ĐỐI không nhắc tới "đoạn trích", "đoạn văn", "sách", "bài học", "theo thông tin trên".
2. Một đáp án đúng duy nhất, đáp án đúng ngắn gọn; ba phương án nhiễu hợp lí (nghe có vẻ đúng), cùng loại và độ dài tương đương đáp án đúng, nhưng chắc chắn sai.
3. Không dùng phương án "tất cả đều đúng/sai", "cả A và B"; không hỏi phủ định ("không phải", "sai"); đáp án đúng không xuất hiện nguyên văn trong câu hỏi.
4. Tiếng Việt chuẩn, đúng thuật ngữ."""

OOC_PROMPT = """Môn: {subject}. Lớp: {grade}. Sách giáo khoa: {book}.
Mục lục sách có nội dung: "{title}"{chapter}.
Hãy viết MỘT câu hỏi trắc nghiệm 4 phương án kiểm tra một kiến thức cốt lõi, cụ thể, đặc trưng của nội dung này theo chương trình GDPT 2018 (định nghĩa, tính chất, quy luật, sự kiện, số liệu, tác giả/thể loại...), mức độ vừa phải. Chỉ hỏi điều bạn chắc chắn về đáp án đúng.
{rules}

Trả về JSON: {{"question": "...", "correct_answer": "...", "distractors": ["...", "...", "..."]}}"""

SPAN_PROMPT = """Môn: {subject}. Lớp: {grade}.
Đoạn trích từ sách giáo khoa:

{block}

Hãy viết MỘT câu hỏi trắc nghiệm 4 phương án kiểm tra một kiến thức cốt lõi, cụ thể trong đoạn trích, mức độ vừa phải; đáp án đúng phải được đoạn trích khẳng định rõ ràng.
{rules}

Trả về JSON: {{"question": "...", "correct_answer": "...", "distractors": ["...", "...", "..."]}}"""

GUARD_PROMPT = """Môn: {subject}. Lớp: {grade}.
Đoạn trích từ sách giáo khoa:

{block}

Đoạn trích có thể chứa một quy tắc/quy luật có điều kiện dạng "nếu/khi P thì Q" (hoặc tương đương). Nếu không có quy tắc điều kiện rõ ràng thì trả về {{"feasible": false}}.
Nếu có, dựng cặp câu hỏi trắc nghiệm có CHUNG bốn phương án:
- "question_satisfied": tình huống cụ thể trong đó điều kiện P được nêu rõ là ĐƯỢC THỎA; hỏi kết luận. Đáp án đúng (Q) suy ra chắc chắn nhờ quy tắc trong đoạn trích.
- "question_guard": tình huống tương tự (cùng đối tượng, cùng câu hỏi cuối, cùng bốn phương án) nhưng điều kiện P KHÔNG được nêu (guard_kind = "unspecified") hoặc được nêu là khác/không thỏa (guard_kind = "unmet"), sao cho từ đoạn trích KHÔNG thể kết luận Q, cũng KHÔNG thể kết luận phủ định của Q (không suy ra phủ định).
- "correct_answer": Q ngắn gọn; "distractors": 3 phương án nhiễu hợp lí (có thể có một phương án phủ định Q), cùng độ dài tương đương.
{rules}
Câu hỏi không được chứa chính quy tắc "nếu P thì Q" (học sinh phải tự nhớ quy tắc từ sách).

Trả về JSON: {{"feasible": true, "condition": "P", "conclusion": "Q", "guard_kind": "unspecified|unmet", "question_satisfied": "...", "question_guard": "...", "correct_answer": "...", "distractors": ["...", "...", "..."]}}"""

JUDGE_PROMPT_NOTE = ("\nChú ý: chỉ đánh dấu \"correct\" nếu các đoạn trích NÊU RÕ hoặc suy ra chắc chắn phương án đó; "
                     "kiến thức bên ngoài đoạn trích, dù đúng, KHÔNG được tính.")


def _opts_from(cand_key: str, correct: str, distractors: list[str]) -> tuple[dict[str, str], str]:
    return assemble_options(cand_key, correct.strip(), [d.strip() for d in distractors][:3])


def _basic_ok(question: str, correct: str, distractors: list[str]) -> bool:
    if not question or not correct or len(distractors) < 3:
        return False
    ds = [d.strip() for d in distractors[:3]]
    texts = {norm(correct)} | {norm(d) for d in ds}
    if len(texts) != 4 or "" in texts:
        return False
    if re.search(r"(đoạn trích|đoạn văn|văn bản trên|thông tin trên|theo sách|trong sách)", question, re.I):
        return False
    if re.search(r"(tất cả (các )?(phương án|đáp án|ý)|cả [ABCD] và [ABCD]|cả ba)", " ".join([correct] + ds), re.I):
        return False
    dl = sum(len(d) for d in ds) / 3
    return len(correct) <= 1.8 * max(dl, 1) and len(question) >= 15


def _strlist(x: Any) -> list[str]:
    return [str(v.get("text") if isinstance(v, dict) else v) for v in (x or []) if v]


GEN_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}, "correct_answer": {"type": "string"},
                                               "distractors": {"type": "array", "items": {"type": "string"}}},
              "required": ["question", "correct_answer", "distractors"]}
GUARD_SCHEMA = {"type": "object", "properties": {
    "feasible": {"type": "boolean"}, "condition": {"type": "string"}, "conclusion": {"type": "string"},
    "guard_kind": {"type": "string", "enum": ["unspecified", "unmet"]}, "question_satisfied": {"type": "string"},
    "question_guard": {"type": "string"}, "correct_answer": {"type": "string"},
    "distractors": {"type": "array", "items": {"type": "string"}}},
    "required": ["feasible"]}


# --------------------------------------------------------------------------------------------- generation
def gen_ooc(topic: dict, llm: LLM, variant: int = 0) -> dict:
    p = OOC_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI[topic["domain"]], grade=topic["grade"], book=topic["book_title"],
                          title=topic["title"], chapter=f" (chương/chủ đề: {topic['chapter']})" if topic.get("chapter") else "",
                          rules=_STYLE_RULES)
    if variant:
        p += f"\n(Phương án #{variant + 1}: chọn một khía cạnh kiến thức KHÁC với câu hỏi thông thường nhất về nội dung này.)"
    base = {"cand_id": f"ooc_{topic['topic_id'][4:]}_{variant}", "kind": "ooc", "domain": topic["domain"], "grade": topic["grade"],
            "topic": topic, "generator": {"model": llm.model, "version": VERSION}}
    try:
        o, _ = chat_json(llm, [{"role": "system", "content": _SYS}, {"role": "user", "content": p}], GEN_SCHEMA,
                         max_tokens=700, temperature=0.7, validate=lambda o: "question" in o and "correct_answer" in o)
    except Exception as e:
        return {**base, "error": f"{type(e).__name__}: {e}"}
    ds = _strlist(o.get("distractors"))
    q, c = str(o["question"]).strip(), str(o["correct_answer"]).strip()
    if not _basic_ok(q, c, ds):
        return {**base, "failed": "basic"}
    opts, key = _opts_from(base["cand_id"], c, ds)
    return {**base, "question": q, "options": opts, "reference_key": key}


def gen_from_span(span: Span, llm: LLM, kind: str = "ctrl_ooc") -> dict:
    p = SPAN_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI[span.domain], grade=span.grade,
                           block=span.text.strip()[:1800], rules=_STYLE_RULES)
    base = {"cand_id": "ctl_" + sha1(span.span_id)[:12], "kind": kind, "domain": span.domain, "grade": span.grade,
            "gold_span_ids": [span.span_id], "lesson_keys": [span.lesson_key], "book_id": span.book_id,
            "generator": {"model": llm.model, "version": VERSION}}
    try:
        o, _ = chat_json(llm, [{"role": "system", "content": _SYS}, {"role": "user", "content": p}], GEN_SCHEMA,
                         max_tokens=700, temperature=0.7, validate=lambda o: "question" in o and "correct_answer" in o)
    except Exception as e:
        return {**base, "error": f"{type(e).__name__}: {e}"}
    ds = _strlist(o.get("distractors"))
    q, c = str(o["question"]).strip(), str(o["correct_answer"]).strip()
    if not _basic_ok(q, c, ds) or norm(c) in norm(q):
        return {**base, "failed": "basic"}
    opts, key = _opts_from(base["cand_id"], c, ds)
    return {**base, "question": q, "options": opts, "answer": key, "correct_answer": c}


def gen_guard(span: Span, llm: LLM) -> dict:
    p = GUARD_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI[span.domain], grade=span.grade,
                            block=span.text.strip()[:1800], rules=_STYLE_RULES)
    base = {"cand_id": "grd_" + sha1(span.span_id)[:12], "kind": "guard", "domain": span.domain, "grade": span.grade,
            "gold_span_ids": [span.span_id], "lesson_keys": [span.lesson_key], "book_id": span.book_id,
            "generator": {"model": llm.model, "version": VERSION}}
    try:
        o, _ = chat_json(llm, [{"role": "system", "content": _SYS}, {"role": "user", "content": p}], GUARD_SCHEMA,
                         max_tokens=1200, temperature=0.7, validate=lambda o: "feasible" in o)
    except Exception as e:
        return {**base, "error": f"{type(e).__name__}: {e}"}
    if not _bool(o.get("feasible")):
        return {**base, "failed": "infeasible"}
    ds = _strlist(o.get("distractors"))
    qs, qg, c = (str(o.get(k) or "").strip() for k in ("question_satisfied", "question_guard", "correct_answer"))
    if not (_basic_ok(qs, c, ds) and _basic_ok(qg, c, ds)) or norm(c) in norm(qs) or norm(c) in norm(qg):
        return {**base, "failed": "basic"}
    if jaccard(char_ngrams(qs), char_ngrams(qg)) > 0.97:
        return {**base, "failed": "identical_questions"}
    opts, key = _opts_from(base["cand_id"], c, ds)
    return {**base, "question_satisfied": qs, "question_guard": qg, "options": opts, "answer": key, "correct_answer": c,
            "condition": o.get("condition"), "conclusion": o.get("conclusion"), "guard_kind": o.get("guard_kind")}


# --------------------------------------------------------------------------------------------- checks
def retrieve(corpus: SpanCorpus, query: str, *, exclude: set[str] = frozenset(), top_k: int = 10) -> list[Span]:
    hits = corpus.search(query, top_k=top_k + len(exclude) + 2, mode="hybrid")
    return [s for s, _, _ in hits if s.span_id not in exclude][:top_k]


def judge_support(llm: LLM, question: str, options: dict[str, str], spans: list[Span], salt: str) -> dict[str, Any]:
    """Per-option status given ONLY `spans` (options re-permuted, spans shuffled)."""
    order = sorted(spans, key=lambda s: sha1(salt + s.span_id))
    labels = [f"E{i + 1}" for i in range(len(order))]
    dummy_key = LETTERS[0]
    opts, _, back = permute(options, dummy_key, salt + ":j")
    p = _VERIFY_PROMPT.format(blocks="\n\n".join(span_block(l, s, 1200) for l, s in zip(labels, order)),
                              question=question, options=fmt_options(opts)) + JUDGE_PROMPT_NOTE
    valid = lambda o: isinstance(o.get("option_status"), dict) and all(_status(o["option_status"].get(k)) for k in LETTERS)
    o, _ = chat_json(llm, [{"role": "user", "content": p}], VERIFY_SCHEMA, max_tokens=500, validate=valid)
    st = {back[k]: _status(o["option_status"].get(k)) for k in LETTERS}
    a = _letter(o.get("answer"))
    lab2 = dict(zip(labels, [s.span_id for s in order]))
    used = []
    for x in o.get("used_spans") or []:
        m = re.search(r"E\d+", str(x).upper())
        if m and m.group(0) in lab2:
            used.append(lab2[m.group(0)])
    return {"status": st, "answer": back.get(a) if a else "none", "confidence": o.get("confidence"), "used": used,
            "evidence_ids": [s.span_id for s in order], "model": llm.model}


def undetermined_ok(j: dict, *, key: Optional[str] = None, key_must_be_undetermined: bool = False) -> bool:
    """Unanswerable acceptance: no option is determined correct by the evidence."""
    if any(v == "correct" for v in j["status"].values()) or j["answer"] not in ("none", None):
        return False
    if key_must_be_undetermined and key is not None:
        return j["status"].get(key) == "undetermined"
    return True


def check_absent(c: dict, corpus: SpanCorpus, llm: LLM, *, exclude: frozenset[str] = frozenset(),
                 extra_spans: Optional[list[Span]] = None, key_must_be_undetermined: bool = False,
                 query_extra: str = "") -> dict:
    q = c["question"] + " " + " ".join(c["options"].values()) + " " + query_extra
    top = retrieve(corpus, q, exclude=set(exclude), top_k=10)
    ev = list(extra_spans or [])
    ids = {s.span_id for s in ev}
    ev += [s for s in top if s.span_id not in ids][: max(4, 10 - len(ev))]
    j = judge_support(llm, c["question"], c["options"], ev, c["cand_id"])
    key = c.get("reference_key") or c.get("answer")
    return {**j, "accepted": undetermined_ok(j, key=key, key_must_be_undetermined=key_must_be_undetermined),
            "top_span_ids": [s.span_id for s in top]}


def check_control(c: dict, corpus: SpanCorpus, llm: LLM, question: Optional[str] = None) -> dict:
    cc = dict(c)
    if question:
        cc["question"] = question
    r = check_verify(cc, corpus, llm)
    st = r["option_status"]
    others = [k for k in LETTERS if k != c["answer"]]
    ok = (r["answer"] == c["answer"] and st.get(c["answer"]) == "correct"
          and all(st.get(k) != "correct" for k in others) and sum(st.get(k) == "undetermined" for k in others) <= 3)
    return {**r, "accepted": ok}
