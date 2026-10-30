"""ViEduHop-Compose: teacher-facing, textbook-grounded exam-item authoring (OTTER product mode, spec 0.10).

Input  = a teacher request (subject, grade, book/lesson, optional focus section, skill level, MCQ-4 format, optional constraint).
Output = a ready-to-use item: question, 4 options, key, short explanation, Bloom/difficulty label and a SOURCE POINTER
         (book, lesson, page, verbatim quote) for the key and for the explanation, or an explicit abstention when the
         requested lesson does not hold enough grounded material.

Systems (same generation prompt skeleton, same model; they differ only in textbook access)
  otter_compose  retrieval over the spans of the requested lesson -> ONE generation call (Qwen3-32B-FP8, no guided
                 decoding) -> deterministic checks (schema, option-length balance +-20%, no exercise copying, valid
                 citations: resolver + quote-in-span) -> separate solver call with the gold passages (key unique) ->
                 verifier (Qwen3-8B) support check of the key and of the explanation. A failed check triggers up to two
                 repair generations (the failed checks are fed back); if still failing the system abstains.
  llm_compose    baseline: same prompt, NO textbook access, the model writes from its own knowledge and is asked for a
                 source pointer anyway; the pointer is marked unverified (and audited post hoc against the lesson).

Both systems are additionally audited POST HOC with the same two checks (so the comparison is symmetric): the lesson spans
most related to the finished item are retrieved and (i) a solver call decides whether the key is derivable and unique,
(ii) the 8B verifier decides whether the key is supported.

Everything is resumable per (request, system) through JsonlStore; every LLM call is logged by otter.llm (runs/llm_calls).
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from otter.agents.v2 import quote_in_span
from otter.evaluation.corpus import ROOT, Span, SpanCorpus, core_book_filter
from otter.evaluation.hard_gen import (_BAD_OPTION, _META_REF, _VERIFY_PROMPT, VERIFY_SCHEMA, JsonlStore, _label,
                                       _letter, _status, assemble_options, chat_json, fmt_options, permute)
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.text import bigram_tokens, char_ngrams, jaccard, norm, sha1
from otter.llm.interface import LLM

PROMPT_VERSION = "otter_compose_gen_v1"
LEVELS = {"nhan_biet": "nhận biết", "thong_hieu": "thông hiểu", "van_dung": "vận dụng"}
LEVEL_DEF = {
    "nhan_biet": ("NHẬN BIẾT: học sinh chỉ cần nhớ lại/nhận ra một khái niệm, định nghĩa, tên gọi, sự kiện, mốc thời gian, "
                  "số liệu hay đặc điểm được nêu trực tiếp trong bài. Câu hỏi ngắn, một ý, không cần suy luận."),
    "thong_hieu": ("THÔNG HIỂU: học sinh phải hiểu và diễn đạt lại, giải thích, phân biệt/so sánh, nhận ra ví dụ hoặc mối quan hệ "
                   "nguyên nhân – kết quả đã học; đáp án không trùng nguyên văn một câu trong sách nhưng suy ra chắc chắn từ nội dung bài."),
    "van_dung": ("VẬN DỤNG: nêu một tình huống/số liệu/ví dụ MỚI (không có sẵn trong sách) và yêu cầu học sinh dùng khái niệm, "
                 "quy tắc, công thức hoặc nhận định của bài để suy ra kết quả/kết luận (nếu có tính toán thì chỉ 1-2 phép tính đơn "
                 "giản, kết quả phải chính xác)."),
}
# constraint templates the teacher can pick (optional field of the request)
CONSTRAINTS = {
    "bang_so_lieu": "cần số liệu trong bảng",
    "nhieu_hay_nham": "có phương án nhiễu là những nhầm lẫn học sinh hay mắc",
    "tinh_huong": "gắn với một tình huống thực tế",
    "so_lieu_cu_the": "có số liệu cụ thể trong câu hỏi",
}

ABSTAIN_TEXT = "Không đủ căn cứ trong bài để soạn câu hỏi theo yêu cầu."


# ------------------------------------------------------------------------------------------------ request
@dataclass
class ComposeRequest:
    req_id: str
    subject: str                      # english domain key (physics ...)
    grade: int
    book_id: str
    lesson_id: str                    # SpanCorpus lesson_key
    level: str                        # nhan_biet | thong_hieu | van_dung
    lesson_label: str = ""            # e.g. "Bài 8: Mô tả sóng"
    book_title: str = ""
    focus: Optional[str] = None       # optional sub-section heading inside the lesson
    focus_path: Optional[list[str]] = None
    constraint: Optional[str] = None  # free text (one of CONSTRAINTS values or custom)
    fmt: str = "mcq4"
    teacher: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def subject_vi(self) -> str:
        return DOMAIN_TO_SUBJECT_VI.get(self.subject, self.subject)

    def text(self) -> str:
        t = f"Môn {self.subject_vi} lớp {self.grade}, {self.lesson_label}"
        if self.focus:
            t += f", mục “{self.focus}”"
        t += f". Soạn 1 câu trắc nghiệm 4 phương án, mức độ {LEVELS[self.level]}."
        if self.constraint:
            t += f" Yêu cầu thêm: {self.constraint}."
        return t

    def to_dict(self) -> dict:
        d = asdict(self)
        d["request_text"] = self.text()
        d["subject_vi"] = self.subject_vi
        d["level_vi"] = LEVELS[self.level]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ComposeRequest":
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in keys})


# ------------------------------------------------------------------------------------------------ lessons
_NON_EVIDENCE = {"exercise", "question", "activity", "figure_caption", "objective"}


def span_usable(s: Span, min_chars: int = 120) -> bool:
    """Evidence span: textbook exposition (not an exercise / question / caption); short table spans are kept."""
    if s.pedagogical_type in _NON_EVIDENCE:
        return False
    t = s.text.strip()
    if s.pedagogical_type == "table":
        return len(t) >= 60 and sum(ch.isdigit() for ch in t) >= 6
    if len(t) < min_chars:
        return False
    return not s.is_exercise()


_SMALL = {"và", "của", "trong", "ở", "với", "từ", "đến", "là", "cho", "các", "những", "một", "về", "theo", "do", "bằng", "để", "tới"}


def _cap1(w: str) -> str:
    for i, ch in enumerate(w):
        if ch.isalpha():
            return w[:i] + ch.upper() + w[i + 1:]
    return w


def _sentence_case(x: str) -> str:
    """ALL-CAPS headings -> Title-ish case (keeps proper nouns capitalised); other headings unchanged. Footnote marks removed."""
    x = re.sub(r"\(\*\)|\(\d\)", "", x or "")
    x = re.sub(r"\s+", " ", x).strip()
    if not x.isupper():
        return _cap1(x)
    ws = x.lower().split(" ")
    first = next((i for i, w in enumerate(ws) if any(c.isalpha() for c in w)), 0)
    return " ".join(w if (i != first and w in _SMALL) else _cap1(w) for i, w in enumerate(ws))


_BAI = re.compile(r"^\s*(bài|chủ đề|chương)\s*\d+", re.I)


def page_text(s: Span) -> str:
    if s.printed_page_verified and s.printed_page_label:
        return f"trang {s.printed_page_label}"
    return f"trang PDF {s.pdf_page_index + 1}" if s.pdf_page_index is not None else "trang ?"


class LessonIndex:
    """Lesson-level view of the 18 core textbooks: lesson -> ordered spans, labels, focus groups, retrieval."""

    def __init__(self, corpus: SpanCorpus, embedder: Any = None):
        self.corpus = corpus
        self.embedder = embedder
        self.order = {s.span_id: i for i, s in enumerate(corpus.spans)}
        self.by_lesson: dict[str, list[Span]] = defaultdict(list)
        for s in corpus.spans:
            self.by_lesson[s.lesson_key].append(s)
        self._vecs: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._bm: dict[str, Any] = {}

    # ---- structure
    def spans(self, lesson_id: str, usable_only: bool = True) -> list[Span]:
        xs = self.by_lesson.get(lesson_id, [])
        return [s for s in xs if span_usable(s)] if usable_only else list(xs)

    def label(self, lesson_id: str) -> str:
        xs = self.by_lesson.get(lesson_id, [])
        cnt: Counter = Counter()
        for s in xs:
            for h in s.heading_path[:2]:
                if _BAI.match(h):
                    cnt[h] += 1
        pref = [h for h in cnt if re.match(r"^\s*bài", h, re.I)] or [h for h in cnt if re.match(r"^\s*chủ đề", h, re.I)] or list(cnt)
        if pref:
            h = max(pref, key=lambda x: cnt[x])
        else:
            c2 = Counter((s.heading_path[1] if len(s.heading_path) > 1 else s.heading_path[0]) for s in xs if s.heading_path)
            h = c2.most_common(1)[0][0] if c2 else lesson_id
        m = re.match(r"^\s*(bài|chủ đề|chương)\s*(\d+)\s*[:.\-–]?\s*(.*)$", h, re.I)
        if m:
            kind = {"bài": "Bài", "chủ đề": "Chủ đề"}.get(m.group(1).lower(), "Chương")
            title = _sentence_case(m.group(3))
            if len(title) > 90:
                title = re.split(r"(?<=\w)\.\s", title)[0][:110]
            return f"{kind} {m.group(2)}" + (f": {title}" if title else "")
        h = _sentence_case(h)
        m = re.search(r":l(\d+)$", lesson_id)
        if m and not re.match(r"^\d+\b", h) and not _BAI.match(h):
            h = f"Bài {int(m.group(1))}: {h}"
        elif m and re.match(r"^\d+\b", h):
            h = re.sub(r"^(\d+)\s*", r"Bài \1: ", h)
        return h

    def focus_groups(self, lesson_id: str, depth: int = 3) -> dict[tuple, list[Span]]:
        g: dict[tuple, list[Span]] = defaultdict(list)
        for s in self.spans(lesson_id):
            if len(s.heading_path) >= depth:
                g[tuple(s.heading_path[:depth])].append(s)
        return g

    # ---- retrieval (BM25 + bge-m3 RRF, restricted to one lesson)
    def _dense(self, spans: list[Span]) -> Optional[np.ndarray]:
        emb = self.embedder
        if emb is None:
            return None
        try:
            if not emb.available():
                return None
            with self._lock:
                todo = [s for s in spans if s.span_id not in self._vecs]
            if todo:
                m = emb.embed([s.text[:1800] for s in todo])
                with self._lock:
                    for s, v in zip(todo, m):
                        self._vecs[s.span_id] = np.asarray(v, dtype=np.float32)
            return np.stack([self._vecs[s.span_id] for s in spans])
        except Exception:
            return None

    def retrieve(self, lesson_id: str, query: str, *, k: int = 6, pool: Optional[list[Span]] = None,
                 numeric_boost: bool = False) -> list[Span]:
        """Top-k usable spans of the lesson for `query`, returned in reading order."""
        cand = pool if pool is not None else self.spans(lesson_id)
        if not cand:
            return []
        from rank_bm25 import BM25Okapi
        bm = BM25Okapi([bigram_tokens(s.text) or ["∅"] for s in cand])
        lists = [list(np.argsort(-np.asarray(bm.get_scores(bigram_tokens(query)), dtype=float), kind="stable"))]
        mat = self._dense(cand)
        if mat is not None:
            try:
                q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
                lists.append(list(np.argsort(-(mat @ q), kind="stable")))
            except Exception:
                pass
        if numeric_boost:
            num = [sum(ch.isdigit() for ch in s.text) / max(1, len(s.text)) + (0.5 if s.pedagogical_type == "table" else 0)
                   for s in cand]
            lists.append(list(np.argsort(-np.asarray(num), kind="stable")))
        sc: dict[int, float] = defaultdict(float)
        for lst in lists:
            for r, i in enumerate(lst):
                sc[int(i)] += 1.0 / (60 + r + 1)
        top = sorted(sc, key=lambda i: -sc[i])[:k]
        return sorted((cand[i] for i in top), key=lambda s: self.order.get(s.span_id, 0))

    def retrieve_for_request(self, req: ComposeRequest, *, k: int = 6) -> list[Span]:
        lesson = self.spans(req.lesson_id)
        pool = lesson
        if req.focus_path:
            fp = tuple(req.focus_path)
            grp = [s for s in lesson if tuple(s.heading_path[:len(fp)]) == fp]
            if len(grp) >= 3:
                pool = grp
        q = " ".join(x for x in (req.focus, req.lesson_label, req.constraint or "") if x)
        numeric = bool(req.constraint and re.search(r"bảng|số liệu|tính", req.constraint, re.I)) or req.level == "van_dung"
        return self.retrieve(req.lesson_id, q, k=k, pool=pool, numeric_boost=numeric)


# ------------------------------------------------------------------------------------------------ prompts
GEN_SYS = ("Bạn là giáo viên giàu kinh nghiệm, soạn câu hỏi trắc nghiệm cho đề kiểm tra bám sát sách giáo khoa Việt Nam "
           "(chương trình GDPT 2018). Chỉ trả về JSON hợp lệ.")

_RULES = """YÊU CẦU CHUNG:
1. Câu hỏi tự đứng được như một câu trong đề kiểm tra cho học sinh: không nhắc "đoạn trích", "văn bản trên", "theo sách", mã [S1]...; \
không chép nguyên văn cả câu trong sách; KHÔNG chép lại câu hỏi/bài tập/hoạt động có sẵn trong sách.
2. Đúng MỘT đáp án đúng. Ba phương án nhiễu hợp lí đối với người chưa nắm chắc bài (nghe có vẻ đúng, cùng loại với đáp án đúng), \
nhưng chắc chắn SAI. Không dùng "tất cả đều đúng/sai", "cả A và B"; tránh câu hỏi phủ định ("không phải", "sai").
3. Bốn phương án có độ dài gần bằng nhau (chênh lệch không quá khoảng 20%); đáp án đúng không dài hơn hẳn và không xuất hiện \
nguyên văn trong câu hỏi.
4. "explanation": 1-2 câu giải thích vì sao đáp án đúng (nêu căn cứ; nếu có tính toán thì nêu phép tính). Không nhắc chữ cái phương án.
5. "bloom": mức độ thực tế của câu bạn soạn (nhan_biet | thong_hieu | van_dung); phải đúng mức độ được yêu cầu.
6. Tiếng Việt chuẩn, đúng thuật ngữ của sách giáo khoa."""

_COMPOSE_SRC = """NGUỒN: chỉ dùng thông tin có trong các đoạn trích (cùng phép tính số học đơn giản nếu cần), không dùng kiến thức ngoài.
- "key_source": {"span": "S<i>", "quote": "<một câu hoặc cụm NGUYÊN VĂN (<= 250 ký tự) trong đoạn S<i> chứng minh đáp án đúng>"}.
- "explanation_source": như trên, cho căn cứ của phần giải thích (có thể là cùng đoạn).
- Nếu các đoạn trích KHÔNG đủ để soạn câu hỏi đúng yêu cầu (ví dụ thiếu nội dung/số liệu được yêu cầu), đặt "feasible": false, \
nêu lí do trong "reason" và để các trường còn lại rỗng. Tuyệt đối không bịa nội dung ngoài đoạn trích."""

_BASELINE_SRC = """NGUỒN: bạn không có sách trước mặt; hãy soạn từ hiểu biết của bạn về nội dung bài được yêu cầu của sách giáo khoa. Vẫn cần ghi \
nguồn: "key_source": {"book": "<tên sách>", "lesson": "<bài>", "page": "<số trang>", "quote": "<một câu trong sách chứng minh đáp án \
đúng>"}; "explanation_source": như trên cho phần giải thích. Đặt "feasible": true."""

_COMPOSE_SCHEMA_TAIL = (
    'Trả về JSON: {"feasible": true|false, "reason": "", "question": "...", "correct": "<đáp án đúng>", '
    '"distractors": ["<nhiễu 1>", "<nhiễu 2>", "<nhiễu 3>"], "distractor_notes": ["<vì sao sai>", "<..>", "<..>"], '
    '"explanation": "...", "bloom": "nhan_biet|thong_hieu|van_dung", '
    '"key_source": {...}, "explanation_source": {...}}')

GEN_SCHEMA = {"type": "object", "properties": {
    "feasible": {"type": "boolean"}, "reason": {"type": "string"}, "question": {"type": "string"},
    "correct": {"type": "string"}, "distractors": {"type": "array", "items": {"type": "string"}},
    "distractor_notes": {"type": "array", "items": {"type": "string"}}, "explanation": {"type": "string"},
    "bloom": {"type": "string", "enum": list(LEVELS)},
    "key_source": {"type": "object"}, "explanation_source": {"type": "object"}},
    "required": ["feasible", "question", "correct", "distractors", "explanation", "bloom", "key_source", "explanation_source"]}


def span_block(label: str, s: Span, max_chars: int = 1400) -> str:
    hp = " › ".join(s.heading_path[-2:]) if s.heading_path else ""
    return f"[{label}] ({s.book_title or s.book_id}, {page_text(s)}" + (f"; {hp}" if hp else "") + f")\n{s.text.strip()[:max_chars]}"


def compose_prompt(req: ComposeRequest, spans: Optional[list[Span]]) -> str:
    head = f"YÊU CẦU CỦA GIÁO VIÊN:\n{req.text()}\nSách: {req.book_title}.\n\n"
    if spans is not None:
        body = ("CÁC ĐOẠN TRÍCH SÁCH GIÁO KHOA của bài được yêu cầu (mã [S1], [S2], ...):\n\n"
                + "\n\n".join(span_block(f"S{i + 1}", s) for i, s in enumerate(spans)) + "\n\n")
        src = _COMPOSE_SRC
    else:
        body = ""
        src = _BASELINE_SRC
    task = (f"NHIỆM VỤ: soạn MỘT câu trắc nghiệm 4 phương án đáp ứng yêu cầu trên.\nMức độ yêu cầu: {LEVEL_DEF[req.level]}\n"
            + (f"Yêu cầu thêm của giáo viên (phải đáp ứng): {req.constraint}.\n" if req.constraint else ""))
    return head + body + task + "\n" + _RULES + "\n\n" + src + "\n\n" + _COMPOSE_SCHEMA_TAIL


# ------------------------------------------------------------------------------------------------ deterministic checks
def len_balance(opts: dict[str, str], tol: float = 0.2, slack: int = 6) -> tuple[bool, float]:
    ls = [len(norm(v)) for v in opts.values()] or [1]
    mean = sum(ls) / len(ls)
    dev = max(abs(x - mean) for x in ls)
    return dev <= max(tol * mean, slack), round(dev / max(mean, 1), 3)


_DANGLING = re.compile(r"(?<![\w])((bảng|hình vẽ|hình ảnh|biểu đồ|sơ đồ|đồ thị|bản đồ|lược đồ|tranh)( số liệu)? (sau|dưới đây|trên|bên)"
                       r"|(dựa vào|quan sát|cho|trong|từ) (bảng|hình vẽ|hình ảnh|hình|biểu đồ|sơ đồ|đồ thị|bản đồ|lược đồ) (sau|dưới|trên|bên))", re.I)


class Leakage:
    """Near-duplicate of the question against textbook exercise spans of the same subject and other benchmarks."""

    def __init__(self, corpus: SpanCorpus, other: Optional[list[tuple[str, str]]] = None):
        self.ex = [(s.domain, char_ngrams(s.text)) for s in corpus.spans if s.is_exercise() or s.pedagogical_type in _NON_EVIDENCE]
        self.other = [(i, char_ngrams(q)) for i, q in (other or [])]

    def check(self, question: str, subject: str) -> dict:
        g = char_ngrams(question)
        ex = max((jaccard(g, e) for d, e in self.ex if d == subject), default=0.0)
        ot = max((jaccard(g, e) for _, e in self.other), default=0.0)
        return {"exercise_sim": round(ex, 3), "other_bench_sim": round(ot, 3), "no_copy": ex < 0.5 and ot < 0.7}


def load_other_questions() -> list[tuple[str, str]]:
    out = []
    for p in (ROOT / "tests/textbooks/OTTER_MCQS_300.jsonl",):
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    out.append((str(r.get("id")), str(r.get("question", ""))))
    return out


def basic_checks(item: dict, req: ComposeRequest, leak: Optional[Leakage]) -> dict:
    """Deterministic checks on an assembled item (question/options/key)."""
    o = item["options"]
    f: dict[str, Any] = {}
    texts = [norm(o.get(k, "")) for k in LETTERS]
    f["schema_ok"] = (len(o) == 4 and all(texts) and len(set(texts)) == 4 and bool(item["question"].strip())
                      and len(item["question"]) >= 15 and bool(item.get("explanation", "").strip()))
    f["no_meta_ref"] = not _META_REF.search(item["question"]) and not any(_META_REF.search(v) for v in o.values())
    f["no_bad_option"] = not any(_BAD_OPTION.search(v) for v in o.values())
    f["no_dangling_ref"] = not _DANGLING.search(item["question"])      # "dựa vào bảng sau" with no table in the item
    f["answer_not_in_question"] = not (len(norm(o[item["key"]])) >= 6 and norm(o[item["key"]]) in norm(item["question"]))
    ok, dev = len_balance(o)
    f["length_balance"], f["length_dev"] = ok, dev
    if leak is not None:
        f.update(leak.check(item["question"], req.subject))
    else:
        f["no_copy"] = True
    f["bloom_match"] = item.get("bloom") == req.level
    hard = ("schema_ok", "no_meta_ref", "no_bad_option", "no_dangling_ref", "answer_not_in_question", "length_balance", "no_copy")
    f["basic_ok"] = all(f[k] for k in hard)
    f["basic_fail"] = [k for k in hard if not f[k]]
    return f


# ------------------------------------------------------------------------------------------------ LLM checks
_SOLVE_NOTE = ("\nTrước khi trả lời, nếu câu hỏi có tính toán hãy tính cẩn thận (không ghi quá trình, chỉ trả JSON).")


def solver_check(llm: LLM, item: dict, spans: list[Span], salt: str) -> dict:
    """Separate solver call: given ONLY `spans`, which options are correct / incorrect / undetermined (options re-permuted)."""
    labels = [f"E{i + 1}" for i in range(len(spans))]
    opts, ans, back = permute(item["options"], item["key"], salt + ":solve")
    p = _VERIFY_PROMPT.format(blocks="\n\n".join(span_block(l, s, 1800) for l, s in zip(labels, spans)),
                              question=item["question"], options=fmt_options(opts))
    valid = lambda o: isinstance(o.get("option_status"), dict) and all(_status(o["option_status"].get(k)) for k in LETTERS)
    o, r = chat_json(llm, [{"role": "user", "content": p}], VERIFY_SCHEMA, max_tokens=500, validate=valid)
    st = {back[k]: _status(o["option_status"].get(k)) for k in LETTERS}
    a = _letter(o.get("answer"))
    a = back.get(a) if a else "none"
    others = [k for k in LETTERS if k != item["key"]]
    n_other_correct = sum(st.get(k) == "correct" for k in others)
    return {"answer": a, "option_status": st, "n_other_correct": n_other_correct,
            "n_other_undetermined": sum(st.get(k) == "undetermined" for k in others),
            "key_unique": a == item["key"] and st.get(item["key"]) == "correct" and n_other_correct == 0, "model": llm.model}


VERIFY_SYS = ("Bạn là người kiểm định bằng chứng độc lập. Chỉ dựa vào các đoạn trích được đưa, không dùng kiến thức bên ngoài. "
              "Chỉ trả về một đối tượng JSON.")
VERIFY_SCHEMA2 = {"type": "object", "properties": {
    "claims": {"type": "array", "items": {"type": "object"}}, "answer_supported": {"type": "string"}},
    "required": ["claims", "answer_supported"]}


def verifier_check(llm: LLM, item: dict, claims: list[dict]) -> dict:
    """Second-model support check. claims: [{id, text, spans:[Span]}]; returns verdict per claim + answer_supported."""
    blocks = []
    for c in claims:
        sp = "\n".join(f"[{s.span_id.split(':', 1)[-1]}] {s.text[:1600]}" for s in c["spans"])
        blocks.append(f"MỆNH ĐỀ {c['id']}: {c['text']}\nĐOẠN TRÍCH ĐƯỢC TRÍCH DẪN:\n{sp}")
    ans = f"{item['key']}. {item['options'][item['key']]}"
    p = (f"Câu hỏi: {item['question']}\nĐáp án được đề xuất: {ans}\n\n" + "\n\n".join(blocks)
         + '\n\nVới mỗi mệnh đề: "supported" = đoạn trích nêu trực tiếp hoặc suy ra hiển nhiên mọi ý quan trọng; "partial" = chỉ một phần; '
           '"unsupported" = không hỗ trợ/mâu thuẫn. Thêm "answer_supported": "yes|no" = các đoạn trích có đủ để kết luận đáp án đề xuất '
           'là đáp án đúng của câu hỏi không.\nTrả về JSON: {"claims": [{"id": "<id>", "verdict": "supported|partial|unsupported"}], '
           '"answer_supported": "yes|no"}')
    o, r = chat_json(llm, [{"role": "system", "content": VERIFY_SYS}, {"role": "user", "content": p}], VERIFY_SCHEMA2,
                     max_tokens=300, validate=lambda o: isinstance(o.get("claims"), list))
    vd = {str(x.get("id")): str(x.get("verdict", "")).lower() for x in o["claims"] if isinstance(x, dict)}
    out = {c["id"]: (vd.get(c["id"]) if vd.get(c["id"]) in ("supported", "partial", "unsupported") else "unsupported") for c in claims}
    a = str(o.get("answer_supported", "")).lower()
    return {"verdicts": out, "answer_supported": a.startswith("y"), "model": llm.model}


# ------------------------------------------------------------------------------------------------ item assembly
def parse_item(obj: dict, cand_key: str) -> Optional[dict]:
    """Model JSON -> item with balanced key position, or None when the reply is not an item."""
    if not isinstance(obj, dict):
        return None
    correct = str(obj.get("correct") or "").strip()
    dis = [str(x.get("text") if isinstance(x, dict) else x).strip() for x in (obj.get("distractors") or [])]
    dis = [d for d in dis if d]
    q = str(obj.get("question") or "").strip()
    if not q or not correct or len(dis) != 3:
        return None
    opts, key = assemble_options(cand_key, correct, dis)
    import unicodedata
    raw = unicodedata.normalize("NFD", str(obj.get("bloom") or "").lower().replace("đ", "d"))
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch)).strip().replace(" ", "_")
    bloom = raw if raw in LEVELS else ""
    return {"question": q, "options": opts, "key": key, "correct_text": correct, "explanation": str(obj.get("explanation") or "").strip(),
            "bloom": bloom, "distractor_notes": [str(x) for x in (obj.get("distractor_notes") or [])][:3]}


def best_sentence(text: str, target: str, min_overlap: float = 0.3) -> Optional[str]:
    """Extractive repair of a pointer: the sentence of `text` that covers most tokens of `target` (verbatim by construction)."""
    tt = set(norm(target).split())
    if not tt:
        return None
    best, score = None, 0.0
    for sent in re.split(r"(?<=[.!?;])\s+|\n+", text):
        st = set(norm(sent).split())
        if len(sent.strip()) < 25 or not st:
            continue
        sc = len(tt & st) / len(tt)
        if sc > score:
            best, score = sent.strip(), sc
    return best[:300] if best and score >= min_overlap else None


def parse_robust(text: str) -> dict:
    from otter.llm.interface import parse_json_loose
    try:
        o = parse_json_loose(text)
    except ValueError:
        s = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        i, j = s.find("{"), s.rfind("}")
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s[i:j + 1]) if i != -1 else ""
        try:
            o = json.loads(fixed)
        except Exception:
            raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    if not isinstance(o, dict):
        raise ValueError("not an object")
    return o


REBALANCE_PROMPT = """Câu trắc nghiệm sau có độ dài các phương án chênh lệch quá nhiều.
Câu hỏi: {q}
Đáp án đúng: {correct}
Các phương án nhiễu: {dis}
Độ dài hiện tại (ký tự): đúng={lc}; nhiễu={ld}.

Hãy viết lại ĐÁP ÁN ĐÚNG và BA PHƯƠNG ÁN NHIỄU sao cho: (1) giữ nguyên ý/nghĩa của đáp án đúng và mỗi phương án nhiễu vẫn SAI; \
(2) cả bốn phương án có cùng cấu trúc câu và mỗi phương án dài từ {lo} đến {hi} ký tự (bốn phương án khoảng {tg} ký tự); \
(3) không thêm chi tiết mới không có trong nội dung bài. Không đổi câu hỏi.
Trả về JSON: {{"correct": "...", "distractors": ["...", "...", "..."]}}"""


def rebalance_options(llm: LLM, item: dict, tries: int = 2) -> tuple[Optional[dict], int]:
    """Targeted repair of option-length imbalance (one cheap call, no re-generation of the question)."""
    calls = 0
    cur = item
    for _ in range(tries):
        ls = [len(v) for v in cur["options"].values()]
        tg = int(round(sum(ls) / 4))
        dis = [v for k, v in cur["options"].items() if k != cur["key"]]
        p = REBALANCE_PROMPT.format(q=cur["question"], correct=cur["correct_text"], dis=json.dumps(dis, ensure_ascii=False),
                                    lc=len(cur["correct_text"]), ld=[len(d) for d in dis], lo=int(tg * 0.85), hi=int(tg * 1.15) + 1, tg=tg)
        try:
            r = llm.chat([{"role": "system", "content": GEN_SYS}, {"role": "user", "content": p}], temperature=0.3, max_tokens=500)
            calls += 1
            o = parse_robust(r.text)
        except Exception:                                     # noqa: BLE001
            continue
        new = parse_item({**{"question": cur["question"], "explanation": cur["explanation"], "bloom": cur["bloom"],
                             "distractor_notes": cur.get("distractor_notes")}, "correct": o.get("correct"),
                          "distractors": o.get("distractors")}, sha1(cur["question"] + str(len(cur["options"]))) + "rb")
        if new is None:
            continue
        if len_balance(new["options"])[0]:
            return new, calls
        cur = new
    return None, calls


def _src_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


# ------------------------------------------------------------------------------------------------ systems
class _Common:
    def __init__(self, main: LLM, verifier: LLM, index: LessonIndex, store: Any, leak: Optional[Leakage] = None,
                 temperature: float = 0.4):
        self.main, self.verifier, self.index, self.store, self.leak = main, verifier, index, store, leak
        self.temperature = temperature
        from otter.reasoning.citation_resolver import CitationResolver
        self.resolver = CitationResolver(store)

    def _chat_obj(self, msgs, temperature):
        """One generation call; tolerant JSON parse (LaTeX backslashes in formulas break strict JSON); one retry."""
        last = None
        for k in range(2):
            r = self.main.chat(msgs, temperature=temperature if k == 0 else max(temperature, 0.5), max_tokens=1500, json_schema=GEN_SCHEMA)
            try:
                return parse_robust(r.text), r
            except ValueError as e:
                last = e
        raise last  # type: ignore[misc]

    def _gen(self, req: ComposeRequest, spans: Optional[list[Span]], temperature: float, feedback: Optional[tuple[str, str]] = None):
        msgs = [{"role": "system", "content": GEN_SYS}, {"role": "user", "content": compose_prompt(req, spans)}]
        if feedback:
            msgs += [{"role": "assistant", "content": feedback[0]}, {"role": "user", "content": feedback[1]}]
        return self._chat_obj(msgs, temperature)

    def post_hoc(self, req: ComposeRequest, item: dict, cited: list[Span]) -> dict:
        """Symmetric audit for both systems: lesson spans most related to the finished item + same solver/verifier."""
        q = item["question"] + " " + item["correct_text"] + " " + item["explanation"]
        ref = self.index.retrieve(req.lesson_id, q, k=5)
        ids = {s.span_id for s in ref}
        ref_all = sorted(ref + [s for s in cited if s.span_id not in ids], key=lambda s: self.index.order.get(s.span_id, 0))
        out: dict[str, Any] = {"ref_span_ids": [s.span_id for s in ref_all]}
        if not ref_all:
            return {**out, "post_solver_key_unique": False, "post_supported": False}
        try:
            sv = solver_check(self.main, item, ref_all, req.req_id + ":post")
            out.update(post_solver_answer=sv["answer"], post_solver_key_unique=sv["key_unique"],
                       post_solver_n_other_correct=sv["n_other_correct"])
        except Exception as e:                       # noqa: BLE001
            out.update(post_solver_error=f"{type(e).__name__}: {e}", post_solver_key_unique=None)
        try:
            vf = verifier_check(self.verifier, item, [{"id": "c1", "text": f"Đáp án đúng của câu hỏi là: {item['correct_text']}",
                                                       "spans": ref_all}])
            out.update(post_verdict=vf["verdicts"]["c1"], post_supported=vf["verdicts"]["c1"] == "supported" and vf["answer_supported"])
        except Exception as e:                       # noqa: BLE001
            out.update(post_verifier_error=f"{type(e).__name__}: {e}", post_supported=None)
        return out

    def _base_record(self, req: ComposeRequest, system: str) -> dict:
        return {"req_id": req.req_id, "system": system, "request": req.to_dict(), "status": "failed", "attempts": [],
                "gen_model": self.main.model, "verifier_model": self.verifier.model, "prompt_version": PROMPT_VERSION}


class OtterCompose(_Common):
    name = "otter_compose"

    def __init__(self, *a, max_attempts: int = 3, min_spans: int = 3, k: int = 6, **kw):
        super().__init__(*a, **kw)
        self.max_attempts, self.min_spans, self.k = max_attempts, min_spans, k

    def compose(self, req: ComposeRequest) -> dict:
        t0 = time.perf_counter()
        rec = self._base_record(req, self.name)
        calls = 0
        try:
            lesson_spans = self.index.spans(req.lesson_id)
            spans = self.index.retrieve_for_request(req, k=self.k)
            rec["shown_span_ids"] = [s.span_id for s in spans]
            rec["n_lesson_spans"] = len(lesson_spans)
            if len(lesson_spans) < self.min_spans or len(spans) < self.min_spans or sum(len(s.text) for s in spans) < 500:
                rec.update(status="abstain", abstain_reason="insufficient_material", abstain_stage="retrieval")
                return rec
            shown = {f"S{i + 1}": s for i, s in enumerate(spans)}
            feedback = None
            for att in range(self.max_attempts):
                a: dict[str, Any] = {"attempt": att}
                obj, r = self._gen(req, spans, self.temperature if att == 0 else 0.7, feedback)
                calls += 1
                a["raw_reply"] = r.text
                if str(obj.get("feasible")).lower() in ("false", "0", "no"):
                    a["feasible"] = False
                    rec["attempts"].append(a)
                    rec.update(status="abstain", abstain_reason="model_infeasible", abstain_stage="generation",
                               abstain_note=str(obj.get("reason") or "")[:400])
                    return rec
                item = parse_item(obj, f"{req.req_id}:{self.name}:{att}")
                if item is None:
                    a["fail"] = ["unparsable_item"]
                    rec["attempts"].append(a)
                    feedback = (r.text[:1500], "Kết quả không đúng định dạng (cần question, correct, 3 distractors). Hãy trả lại JSON đầy đủ.")
                    continue
                chk = basic_checks(item, req, self.leak)
                chk["rebalanced"] = False
                if chk["basic_fail"] == ["length_balance"]:      # targeted option rewrite instead of a full re-generation
                    new, nc = rebalance_options(self.main, item)
                    calls += nc
                    if new is not None:
                        item, chk = new, basic_checks(new, req, self.leak)
                        chk["rebalanced"] = True
                # citations
                ks, es = _src_dict(obj.get("key_source")), _src_dict(obj.get("explanation_source"))
                kl, el = _label(ks.get("span"), list(shown)), _label(es.get("span"), list(shown))
                kq, eq = str(ks.get("quote") or ""), str(es.get("quote") or "")
                kspan, espan = shown.get(kl), shown.get(el)
                chk["citation_resolves"] = bool(kspan and espan and self.resolver.resolve(kspan.span_id) and self.resolver.resolve(espan.span_id))
                chk["key_quote_in_span"] = bool(kspan and quote_in_span(kq, kspan.text))
                chk["key_quote_repaired"] = False
                if kspan and not chk["key_quote_in_span"]:         # extractive repair: best verbatim sentence for the key
                    rq = best_sentence(kspan.text, item["correct_text"] + " " + item["question"] + " " + kq, 0.35)
                    if rq:
                        kq, chk["key_quote_in_span"], chk["key_quote_repaired"] = rq, True, True
                chk["expl_quote_in_span"] = bool(espan and quote_in_span(eq, espan.text))
                chk["expl_quote_repaired"] = False
                if espan and not chk["expl_quote_in_span"]:       # explanation pointer: fall back to the best verbatim sentence
                    rq = best_sentence(espan.text, item["explanation"] + " " + eq)
                    if rq:
                        eq, chk["expl_quote_in_span"], chk["expl_quote_repaired"] = rq, True, True
                chk["citations_ok"] = chk["citation_resolves"] and chk["key_quote_in_span"] and chk["expl_quote_in_span"]
                chk["constraint_table_ok"] = True
                if req.constraint and re.search(r"bảng", req.constraint, re.I):      # "cần số liệu trong bảng" must be honoured or refused
                    chk["constraint_table_ok"] = any(s is not None and (s.pedagogical_type == "table" or s.text.count("|") >= 4)
                                                     for s in (kspan, espan))
                fails = list(chk["basic_fail"]) + ([] if chk["constraint_table_ok"] else ["constraint_table"]) + ([] if chk["citations_ok"] else
                                                   [k for k in ("citation_resolves", "key_quote_in_span", "expl_quote_in_span") if not chk[k]])
                a.update(item=item, key_source={"span": kl, "span_id": kspan.span_id if kspan else None, "quote": kq},
                         explanation_source={"span": el, "span_id": espan.span_id if espan else None, "quote": eq})
                gold = list({s.span_id: s for s in (kspan, espan) if s is not None}.values())
                if not fails and gold:
                    try:
                        sv = solver_check(self.main, item, gold, f"{req.req_id}:{att}")
                        calls += 1
                        chk.update(solver_answer=sv["answer"], solver_n_other_correct=sv["n_other_correct"],
                                   solver_key_unique=sv["key_unique"])
                        if not sv["key_unique"]:
                            fails.append("solver_key_unique")
                    except Exception as e:                   # noqa: BLE001
                        chk["solver_error"] = f"{type(e).__name__}: {e}"
                        fails.append("solver_error")
                    try:
                        claims = [{"id": "c1", "text": f"Đáp án đúng của câu hỏi là: {item['correct_text']}", "spans": [kspan]},
                                  {"id": "c2", "text": item["explanation"], "spans": [espan]}]
                        vf = verifier_check(self.verifier, item, claims)
                        calls += 1
                        chk.update(verifier_key=vf["verdicts"]["c1"], verifier_expl=vf["verdicts"]["c2"],
                                   verifier_answer_supported=vf["answer_supported"])
                        chk["verifier_ok"] = vf["verdicts"]["c1"] == "supported" and vf["answer_supported"] and vf["verdicts"]["c2"] != "unsupported"
                        if not chk["verifier_ok"]:
                            fails.append("verifier_support")
                    except Exception as e:                   # noqa: BLE001
                        chk["verifier_error"] = f"{type(e).__name__}: {e}"
                        fails.append("verifier_error")
                a.update(checks=chk, fail=fails)
                rec["attempts"].append(a)
                if not fails:
                    rec.update(status="ok", item=item)
                    rec["source"] = {"key": self._cite(kspan, kq), "explanation": self._cite(espan, eq)}
                    rec["cited_span_ids"] = [s.span_id for s in gold]
                    rec["checks"] = chk
                    rec.update(self.post_hoc(req, item, gold))
                    return rec
                extra = ""
                if "length_balance" in fails:
                    ls = {k: len(v) for k, v in item["options"].items()}
                    extra = (f" Độ dài (số ký tự) hiện tại của các phương án: {ls}; trung bình {sum(ls.values()) / 4:.0f}. "
                             "Hãy viết lại sao cho MỖI phương án dài trong khoảng 80%-120% độ dài trung bình đó (rút gọn phương án dài hoặc "
                             "bổ sung chi tiết hợp lí, vẫn sai, cho phương án ngắn).")
                feedback = (r.text[:1800], "Câu trên CHƯA đạt kiểm tra tự động: " + ", ".join(_FAIL_VI.get(f, f) for f in fails) + "." + extra
                            + " Hãy soạn lại (có thể đổi ý hỏi) để khắc phục, vẫn chỉ dùng các đoạn trích, trả lại JSON đầy đủ.")
            last = rec["attempts"][-1] if rec["attempts"] else {}
            rec.update(status="rejected", abstain_reason="checks_failed", abstain_stage="checks", item_rejected=last.get("item"),
                       last_fail=last.get("fail"))
        except Exception as e:                               # noqa: BLE001
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            rec["llm_calls"] = calls
            rec["latency_s"] = round(time.perf_counter() - t0, 1)
        return rec

    def _cite(self, s: Span, quote: str) -> dict:
        c = self.resolver.resolve(s.span_id)
        return {"span_id": s.span_id, "book": c.book_title, "grade": c.grade, "lesson": c.lesson_title,
                "page": (c.page_number and f"trang {c.page_number}") or (c.pdf_page_number and f"trang PDF {c.pdf_page_number}") or "trang ?",
                "page_verified": bool(c.page_number), "quote": quote, "quote_in_span": quote_in_span(quote, s.text)}


_FAIL_VI = {"length_balance": "độ dài bốn phương án chênh lệch quá 20%", "no_copy": "câu hỏi quá giống một bài tập/câu hỏi có sẵn",
            "schema_ok": "thiếu/trùng phương án hoặc thiếu giải thích", "no_meta_ref": "nhắc tới 'đoạn trích/văn bản trên'",
            "no_bad_option": "dùng phương án kiểu 'tất cả đều đúng'",
            "constraint_table": "yêu cầu 'số liệu trong bảng' nhưng nguồn của đáp án không phải một bảng trong sách; hãy dùng số liệu của bảng trong các đoạn trích (nếu không có bảng thì đặt feasible=false)",
            "no_dangling_ref": "câu hỏi viện dẫn 'bảng/hình sau' nhưng câu hỏi không kèm bảng/hình (hãy nêu số liệu ngay trong câu hỏi)", "answer_not_in_question": "đáp án xuất hiện trong câu hỏi",
            "citation_resolves": "mã đoạn nguồn không hợp lệ", "key_quote_in_span": "quote của đáp án không có nguyên văn trong đoạn",
            "expl_quote_in_span": "quote của giải thích không có nguyên văn trong đoạn",
            "solver_key_unique": "người giải độc lập chỉ dựa vào đoạn trích không xác định được đáp án duy nhất",
            "verifier_support": "bộ kiểm định thấy đoạn trích không đủ chứng minh đáp án/giải thích", "unparsable_item": "định dạng sai"}


class LLMCompose(_Common):
    name = "llm_compose"

    def compose(self, req: ComposeRequest) -> dict:
        t0 = time.perf_counter()
        rec = self._base_record(req, self.name)
        calls = 0
        try:
            for att in range(2):          # second attempt only when the reply is not an item
                obj, r = self._gen(req, None, self.temperature if att == 0 else 0.7)
                calls += 1
                rec["attempts"].append({"attempt": att, "raw_reply": r.text})
                item = parse_item(obj, f"{req.req_id}:{self.name}:{att}")
                if item is not None:
                    break
            else:
                rec.update(status="failed", error="unparsable_item")
                return rec
            chk = basic_checks(item, req, self.leak)
            ks, es = _src_dict(obj.get("key_source")), _src_dict(obj.get("explanation_source"))
            claimed = {"key": {k: str(ks.get(k) or "") for k in ("book", "lesson", "page", "quote")},
                       "explanation": {k: str(es.get(k) or "") for k in ("book", "lesson", "page", "quote")}}
            chk.update(self._audit_pointer(req, claimed["key"]))
            rec.update(status="ok", item=item, source={"key": {**claimed["key"], "verified": False},
                                                       "explanation": {**claimed["explanation"], "verified": False}},
                       checks=chk)
            rec.update(self.post_hoc(req, item, []))
        except Exception as e:                               # noqa: BLE001
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            rec["llm_calls"] = calls
            rec["latency_s"] = round(time.perf_counter() - t0, 1)
        return rec

    def _audit_pointer(self, req: ComposeRequest, src: dict) -> dict:
        """Post-hoc audit of the (unverified) pointer the model wrote: does the quote exist in the requested lesson and on that page?"""
        lesson = self.index.spans(req.lesson_id, usable_only=False)
        hit = [s for s in lesson if quote_in_span(src.get("quote", ""), s.text)]
        pg = re.search(r"\d+", src.get("page", ""))
        pages = {str(s.printed_page_label) for s in hit if s.printed_page_label}
        return {"pointer_quote_in_lesson": bool(hit), "pointer_page_ok": bool(hit and pg and pg.group(0) in pages),
                "pointer_book_ok": norm(req.book_title) in norm(src.get("book", "")) if src.get("book") else False}


# ------------------------------------------------------------------------------------------------ runner
def build_runtime(*, mock: bool = False):
    """(index, store, leak, main, verifier) with the default config (:8004 main, :8001 small, bge-m3 :8002)."""
    from otter.agents.systems import default_store
    from otter.llm.interface import get_llm
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    emb = None
    if not mock:
        try:
            from otter.storage.embeddings import Bgem3Embedder
            emb = Bgem3Embedder.from_config()
        except Exception:
            emb = None
    index = LessonIndex(corpus, emb)
    store = default_store()
    leak = Leakage(corpus, load_other_questions())
    return index, store, leak, get_llm("main"), get_llm("small")


def make_systems(index, store, leak, main, verifier, **kw) -> dict[str, Any]:
    return {"otter_compose": OtterCompose(main, verifier, index, store, leak, **kw),
            "llm_compose": LLMCompose(main, verifier, index, store, leak)}


def run_all(requests: list[ComposeRequest], systems: dict[str, Any], out_dir: Path, *, workers: int = 8,
            only: Optional[list[str]] = None, tag: str = "") -> dict[str, JsonlStore]:
    """Resumable: one JSONL per system under out_dir; rows with status failed/error are retried."""
    stores = {n: JsonlStore(out_dir / n / f"items{tag}.jsonl", lambda r: r["req_id"]) for n in systems}
    jobs = [(n, r) for r in requests for n in systems if (not only or n in only)
            and not (stores[n].rows.get(r.req_id) and stores[n].rows[r.req_id].get("status") != "failed")]

    def one(job):
        n, r = job
        stores[n].add(systems[n].compose(r))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as ex:
        list(ex.map(one, jobs))
    return stores
