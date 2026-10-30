"""OTTER-Hard item generation and automatic checks.

Generation (one structured call per seed): the LLM sees 2-3 labelled textbook spans and a target hop type
and returns {feasible, hop_type, link, reasoning_steps, question, correct_answer, distractors, gold_spans}.
We assemble the four options ourselves and place the key at a balanced position (seed-hash cycling), so the
answer-position distribution is uniform by construction.

Automatic checks (each stored as its own resumable record):
* `verify`      — main model, *gold spans only*, options re-permuted: per-option correct/incorrect/
                  undetermined + chosen answer (answerability + unique-answer + distractor validity);
* `single:<sid>` — main model with ONE gold span: is it sufficient? (multi-hop necessity);
* `closed:<role>:<perm>` — closed-book answer + confidence (difficulty label; the small model is the
                  difficulty probe, the main model is recorded for reference);
* deterministic: structure, meta-references to "the excerpt", answer leaked in the question, option
  length cue, near-duplicate options, verbatim copying, exercise/MCQ300 leakage, dedup.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_seeds import HOP_TYPES, Seed
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.text import char_ngrams, jaccard, norm, sha1, tokens
from otter.llm.interface import LLM, parse_json_loose

GEN_PROMPT_VERSION = "otter_hard_gen_v2"
VERIFY_PROMPT_VERSION = "otter_hard_verify_v1"
SINGLE_PROMPT_VERSION = "otter_hard_single_v1"
CLOSED_PROMPT_VERSION = "otter_hard_closed_v1"
QUALITY_PROMPT_VERSION = "otter_hard_quality_v1"

HOP_DEF_VI = {
    "bridge": "BẮC CẦU: một đoạn giúp xác định một khái niệm/thực thể trung gian (KHÔNG được nêu tên trong câu hỏi), "
              "đoạn còn lại cho biết đặc điểm/thuộc tính của thực thể đó; phải đi qua thực thể trung gian mới trả lời được.",
    "comparison": "SO SÁNH: so sánh/đối chiếu hai đối tượng, quá trình hoặc khái niệm mà thông tin về mỗi bên nằm ở "
                  "một đoạn khác nhau (giống/khác nhau, lớn/nhỏ hơn, điểm chung...).",
    "conditional_rule": "ÁP DỤNG QUY TẮC CÓ ĐIỀU KIỆN: một đoạn nêu quy tắc dạng 'nếu/khi … thì …'; câu hỏi đưa ra một "
                        "tình huống mà muốn biết điều kiện có thỏa mãn hay không phải dùng thông tin ở đoạn khác; "
                        "đáp án là kết luận của quy tắc (hoặc kết luận rằng quy tắc không áp dụng).",
    "causal_chain": "CHUỖI NHÂN QUẢ: chuỗi nguyên nhân → kết quả có ít nhất hai mắt xích nằm ở các đoạn khác nhau; "
                    "câu hỏi hỏi về mắt xích đầu và cuối, bỏ qua mắt xích ở giữa.",
    "calculation": "TÍNH TOÁN: cần lấy công thức/định nghĩa/định luật từ một đoạn và một đại lượng, hằng số, điều kiện "
                   "hoặc quan hệ từ đoạn khác rồi tính ra kết quả số (có đơn vị). Có thể cho số liệu trong tình huống "
                   "câu hỏi, nhưng công thức và quan hệ phải lấy từ các đoạn.",
    "temporal": "THỜI GIAN: sắp xếp, đối chiếu thứ tự hoặc khoảng cách thời gian giữa các sự kiện/giai đoạn được nêu "
                "ở các đoạn khác nhau.",
}

_GEN_SYS = ("Bạn là giáo viên giàu kinh nghiệm, đang ra đề trắc nghiệm KHÓ (suy luận nhiều bước) bám sát sách giáo "
            "khoa Việt Nam (chương trình GDPT 2018). Chỉ trả về JSON hợp lệ.")

HOP_PATTERN_VI = {
    "bridge": "S1: “X là nơi diễn ra quá trình Y.” S2: “Quá trình Y cần chất Z.” → Hỏi: “Hoạt động diễn ra ở X cần chất nào?” "
              "(không nêu Y).",
    "comparison": "S1 mô tả đặc điểm của X; S2 mô tả đặc điểm của Y → Hỏi: “Điểm khác nhau giữa X và Y là gì?”, mỗi phương án "
                  "là một cặp (đặc điểm X – đặc điểm Y), phương án nhiễu tráo đặc điểm hoặc lấy đặc điểm không thuộc đối tượng.",
    "conditional_rule": "S1: “Nếu P thì Q.” S2: “Đối tượng O có đặc điểm P.” → Hỏi: “Điều gì xảy ra với O khi …?” (phải dùng S2 "
                        "để biết O thỏa P, dùng S1 để suy ra Q).",
    "causal_chain": "S1: “A dẫn đến B.” S2: “B làm cho C.” → Hỏi: “A cuối cùng kéo theo hệ quả nào?” (đáp án C; nhiễu: B "
                    "(dừng giữa chừng), hệ quả ngược chiều, nguyên nhân khác).",
    "calculation": "S1: công thức/định nghĩa đại lượng; S2: hằng số, quan hệ hoặc điều kiện → tình huống có số cụ thể, hỏi kết "
                   "quả; nhiễu là kết quả khi dùng sai công thức/đơn vị/bỏ một bước.",
    "temporal": "S1: sự kiện E1 (thời điểm t1); S2: sự kiện E2 (thời điểm t2) → Hỏi thứ tự, khoảng cách thời gian hoặc sự kiện "
                "nào diễn ra trong cùng giai đoạn; không nêu thời điểm trong câu hỏi nếu đó là thông tin cần tìm.",
}

_GEN_PROMPT = """Môn: {subject}. Lớp: {grades}.
Dưới đây là {k} đoạn trích từ sách giáo khoa (mã [S1], [S2], ...).

{blocks}

NHIỆM VỤ: Viết MỘT câu hỏi trắc nghiệm 4 phương án, KHÓ, kiểu suy luận ưu tiên: {hop_def}
Mẫu trừu tượng: {hop_pattern}
(Nếu kiểu này không phù hợp với các đoạn trích, được chọn kiểu khác trong danh sách: {compat}.)

CÁCH LÀM:
(a) "facts": rút ra từ MỖI đoạn một vài sự kiện/quy tắc/số liệu nguyên tử, ghi rõ đoạn nguồn.
(b) "link": xác định mắt xích nối các đoạn (thực thể trung gian, quy tắc, chuỗi nhân quả, công thức, mốc thời gian).
(c) Viết câu hỏi sao cho: chỉ đọc một đoạn thì KHÔNG biết đáp án (thiếu mắt xích), phải ghép thông tin từ ít nhất hai đoạn.
    Câu hỏi phải cụ thể (nêu chi tiết/tình huống lấy từ một đoạn, hỏi điều chỉ suy ra được khi dùng đoạn kia), không hỏi
    chung chung kiểu "Yếu tố nào ảnh hưởng đến …?".

YÊU CẦU BẮT BUỘC:
1. Câu hỏi tự đứng được như câu hỏi kiểm tra cho học sinh: TUYỆT ĐỐI không nhắc tới "đoạn trích", "đoạn văn", "văn \
bản trên", "S1", "theo thông tin trên"...; không chép nguyên văn cả câu trong đoạn trích; với kiểu bắc cầu không nêu \
tên thực thể trung gian.
2. Đáp án đúng duy nhất, được suy ra chắc chắn từ các đoạn trích (chỉ dùng kiến thức trong đoạn trích, cùng phép tính \
số học nếu cần). Không dùng kiến thức ngoài sách.
3. Ba phương án nhiễu hợp lí đối với người chưa đọc kĩ sách (nghe có vẻ đúng theo hiểu biết thông thường), cùng loại và \
độ dài tương đương đáp án đúng, ưu tiên dùng khái niệm/số liệu có trong đoạn trích (kết quả khi chỉ dùng một đoạn, khi bỏ \
qua điều kiện, khi đảo nguyên nhân–kết quả, khi tính sai một bước), nhưng CHẮC CHẮN SAI theo các đoạn trích.
4. Không dùng phương án "tất cả đều đúng/sai", "cả A và B"; tránh câu hỏi phủ định ("không phải", "sai").
5. Đáp án đúng không xuất hiện nguyên văn trong câu hỏi; không dài hơn hẳn các phương án khác.
6. Tiếng Việt chuẩn, đúng thuật ngữ của sách.
7. Nếu các đoạn trích không cho phép một câu hỏi nhiều bước tốt (ví dụ các đoạn không thực sự liên quan) thì đặt
   "feasible": false (các trường khác để rỗng).

Trả về JSON theo đúng lược đồ: feasible, hop_type, facts, link, reasoning_steps (mỗi bước ghi đoạn được dùng và suy luận),
question, correct_answer, distractors (đúng 3, mỗi cái kèm lí do sai), gold_spans (các đoạn cần thiết)."""

_VERIFY_PROMPT = """Chỉ dựa vào các ĐOẠN TRÍCH sách giáo khoa dưới đây (không dùng kiến thức bên ngoài), hãy trả lời câu hỏi \
trắc nghiệm và đánh giá từng phương án.
- "correct": đoạn trích cho phép khẳng định phương án đúng.
- "incorrect": đoạn trích cho phép khẳng định phương án sai.
- "undetermined": đoạn trích không đủ để khẳng định đúng hay sai.
"answer" là phương án đúng duy nhất nếu xác định được, ngược lại "none".

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
{options}

Trả về JSON: {{"option_status": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A|B|C|D|none", \
"used_spans": ["E1", ...], "confidence": 0-100}}"""

_SINGLE_PROMPT = """Chỉ dựa vào MỘT đoạn trích sách giáo khoa dưới đây (không dùng kiến thức bên ngoài): đoạn trích này có \
ĐỦ thông tin để xác định chắc chắn đáp án đúng của câu hỏi không? Nếu đủ, cho biết đáp án; nếu thiếu thông tin cần \
thiết (dù có thể đoán), trả lời "sufficient": false và "answer": "none".

ĐOẠN TRÍCH:
{block}

Câu hỏi: {question}
{options}

Trả về JSON: {{"sufficient": true|false, "answer": "A|B|C|D|none", "missing": "<thông tin còn thiếu, nếu có>"}}"""

_QUALITY_PROMPT = """Bạn là chuyên gia khảo thí. Đánh giá câu hỏi trắc nghiệm nhiều bước dưới đây, được ra từ các ĐOẠN TRÍCH \
sách giáo khoa. Chấm điểm 1-5 cho:
- "natural": câu hỏi tự nhiên, rõ ràng, một học sinh có thể hiểu mà không cần thấy đoạn trích; không gượng ép ghép hai \
chủ đề không liên quan.
- "needs_both": muốn trả lời đúng phải kết hợp thông tin từ cả hai đoạn (không phải kiến thức chung).
- "distractors": ba phương án nhiễu hợp lí, cùng dạng với đáp án, nhưng chắc chắn sai theo đoạn trích.
Chỉ dựa vào đoạn trích.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
{options}
Đáp án đúng: {answer}

Trả về JSON: {{"natural": 1-5, "needs_both": 1-5, "distractors": 1-5}}"""

_CLOSED_SYS = ("Bạn là trợ lý trả lời câu hỏi trắc nghiệm theo chương trình sách giáo khoa Việt Nam. "
               "Chỉ trả về JSON hợp lệ.")

_META_REF = re.compile(r"(đoạn trích|đoạn văn|văn bản trên|thông tin trên|các đoạn|đoạn \d|\bS\d\b|\bE\d\b|"
                       r"theo đoạn|dựa vào đoạn|dữ liệu trên|ngữ liệu trên)", re.I)
_BAD_OPTION = re.compile(r"(tất cả (các )?(phương án|đáp án|ý)|cả [ABCD] và [ABCD]|cả ba|không có (phương án|đáp án))", re.I)


def gen_schema(labels: list[str], hop_types: list[str]) -> dict:
    s = {"type": "string"}
    return {"type": "object", "properties": {
        "feasible": {"type": "boolean"},
        "hop_type": {"type": "string", "enum": hop_types},
        "facts": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "properties": {"span": {"type": "string", "enum": labels}, "fact": s},
            "required": ["span", "fact"]}},
        "link": s,
        "reasoning_steps": {"type": "array", "minItems": 0, "maxItems": 5, "items": {
            "type": "object", "properties": {"span": {"type": "string", "enum": labels}, "step": s},
            "required": ["span", "step"]}},
        "question": s,
        "correct_answer": s,
        "distractors": {"type": "array", "minItems": 0, "maxItems": 3, "items": {
            "type": "object", "properties": {"text": s, "why_wrong": s}, "required": ["text", "why_wrong"]}},
        "gold_spans": {"type": "array", "items": {"type": "string", "enum": labels}, "maxItems": len(labels)},
    }, "required": ["feasible", "hop_type", "facts", "link", "reasoning_steps", "question", "correct_answer", "distractors",
                    "gold_spans"]}


VERIFY_SCHEMA = {"type": "object", "properties": {
    "option_status": {"type": "object", "properties": {k: {"type": "string", "enum": ["correct", "incorrect", "undetermined"]}
                                                        for k in LETTERS}, "required": list(LETTERS)},
    "answer": {"type": "string", "enum": list(LETTERS) + ["none"]},
    "used_spans": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["option_status", "answer", "used_spans", "confidence"]}
SINGLE_SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"},
                                                  "answer": {"type": "string", "enum": list(LETTERS) + ["none"]},
                                                  "missing": {"type": "string"}},
                 "required": ["sufficient", "answer", "missing"]}
CLOSED_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": list(LETTERS)},
                                                  "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
                 "required": ["answer", "confidence"]}


def chat_json(llm: LLM, messages: list[dict], schema: dict, *, max_tokens: int, temperature: float = 0.0,
              retries: int = 1, validate: Optional[Callable[[Any], bool]] = None) -> tuple[Any, Any]:
    """Chat + loose JSON parse; retries once on transport errors, unparsable or invalid replies
    (no server-side guided decoding by default, see otter/llm/interface.py)."""
    last: Optional[Exception] = None
    for k in range(retries + 1):
        try:
            r = llm.chat(messages, temperature=temperature if k == 0 else max(temperature, 0.3),
                         max_tokens=max_tokens, json_schema=schema)
            obj = parse_json_loose(r.text)
            if not isinstance(obj, dict) or (validate is not None and not validate(obj)):
                raise ValueError(f"invalid JSON reply: {r.text[:200]!r}")
            return obj, r
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            last = e
    raise last  # type: ignore[misc]


def _letter(x: Any) -> Optional[str]:
    m = re.match(r"\s*\(?([ABCD])\b", str(x or "").upper())
    return m.group(1) if m else None


_STATUS_SYN = {"correct": "correct", "đúng": "correct", "true": "correct", "incorrect": "incorrect", "sai": "incorrect",
               "false": "incorrect", "wrong": "incorrect", "undetermined": "undetermined", "unknown": "undetermined",
               "không xác định": "undetermined", "không xác định được": "undetermined"}


def _status(x: Any) -> Optional[str]:
    return _STATUS_SYN.get(str(x or "").strip().lower())


def _bool(x: Any) -> bool:
    return x is True or str(x).strip().lower() in ("true", "yes", "có", "đủ", "1")


def _label(x: Any, labels: list[str]) -> Optional[str]:
    m = re.search(r"([SE]\d+)", str(x or "").upper())
    return m.group(1) if m and m.group(1) in labels else None


def span_block(label: str, s: Span, max_chars: int = 1800) -> str:
    hp = " › ".join(s.heading_path[-2:]) if s.heading_path else ""
    head = f"[{label}] ({s.book_title or s.book_id}, lớp {s.grade}" + (f"; {hp}" if hp else "") + ")"
    return f"{head}\n{s.text.strip()[:max_chars]}"


def _hash_int(x: str) -> int:
    return int(sha1(x)[:8], 16)


def assemble_options(cand_key: str, correct: str, distractors: list[str]) -> tuple[dict[str, str], str]:
    """Key letter cycles with a hash of the candidate key -> uniform answer positions."""
    pos = _hash_int(cand_key) % 4
    ds = sorted(distractors, key=lambda d: sha1(cand_key + d))
    opts = ds[:pos] + [correct] + ds[pos:]
    return dict(zip(LETTERS, opts)), LETTERS[pos]


def permute(options: dict[str, str], answer: str, salt: str) -> tuple[dict[str, str], str, dict[str, str]]:
    """Deterministic re-permutation; returns (new_options, new_answer, new_letter->old_letter)."""
    order = sorted(LETTERS, key=lambda k: sha1(salt + k))
    new = {nl: options[ol] for nl, ol in zip(LETTERS, order)}
    back = dict(zip(LETTERS, order))
    return new, next(nl for nl, ol in back.items() if ol == answer), back


def fmt_options(o: dict[str, str]) -> str:
    return "\n".join(f"{k}. {o[k]}" for k in LETTERS)


# ------------------------------------------------------------------ generation
def generate_candidate(seed: Seed, corpus: SpanCorpus, llm: LLM, *, temperature: float = 0.7,
                       max_tokens: int = 3000, extra_prompt: str = "",
                       prompt_version: str = GEN_PROMPT_VERSION) -> dict[str, Any]:
    spans = [corpus.get(i) for i in seed.span_ids]
    labels = [f"S{i + 1}" for i in range(len(spans))]
    compat = [h for h in dict.fromkeys([seed.target_hop_type] + list(seed.compatible_hop_types)) if h in HOP_TYPES]
    grades = sorted({s.grade for s in spans if s.grade is not None})
    prompt = _GEN_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI.get(seed.domain, seed.domain),
                                grades=", ".join(map(str, grades)) or "?", k=len(spans),
                                blocks="\n\n".join(span_block(l, s) for l, s in zip(labels, spans)),
                                hop_def=HOP_DEF_VI[seed.target_hop_type],
                                hop_pattern=HOP_PATTERN_VI[seed.target_hop_type], compat=", ".join(compat))
    if extra_prompt:
        prompt = prompt.replace("NHIỆM VỤ:", extra_prompt.strip() + "\n\nNHIỆM VỤ:", 1)
    cand: dict[str, Any] = {"cand_id": "cand_" + sha1(seed.seed_id + prompt_version + llm.model)[:12],
                            "seed_id": seed.seed_id, "domain": seed.domain, "grade": seed.grade,
                            "seed_span_ids": seed.span_ids, "strategy": seed.strategy,
                            "target_hop_type": seed.target_hop_type,
                            "generator": {"model": llm.model, "prompt_version": prompt_version,
                                          "temperature": temperature}}
    def ok(o):
        if not _bool(o.get("feasible")):
            return True
        return bool(str(o.get("question", "")).strip()) and isinstance(o.get("distractors"), list)
    try:
        obj, r = chat_json(llm, [{"role": "system", "content": _GEN_SYS}, {"role": "user", "content": prompt}],
                           gen_schema(labels, compat), temperature=temperature, max_tokens=max_tokens, validate=ok)
        cand["generator"].update(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens)
    except Exception as e:
        cand.update(feasible=False, error=f"{type(e).__name__}: {e}")
        return cand
    obj["feasible"] = _bool(obj.get("feasible"))
    obj["distractors"] = [d if isinstance(d, dict) else {"text": str(d), "why_wrong": ""} for d in obj.get("distractors") or []]
    obj["gold_spans"] = [l for l in (_label(x, labels) for x in obj.get("gold_spans") or []) if l]
    for key, fld in (("reasoning_steps", "span"), ("facts", "span")):
        obj[key] = [dict(x, span=_label(x.get(fld), labels)) for x in obj.get(key) or [] if isinstance(x, dict)]
    lab2id = dict(zip(labels, seed.span_ids))
    feasible = bool(obj.get("feasible")) and bool(str(obj.get("question", "")).strip())
    dis = [str(d.get("text", "")).strip() for d in obj.get("distractors", []) if isinstance(d, dict)]
    cand.update(feasible=feasible, hop_type=obj.get("hop_type") if obj.get("hop_type") in HOP_TYPES else seed.target_hop_type,
                link=obj.get("link", ""), calc=obj.get("calc") if isinstance(obj.get("calc"), dict) else None, facts=[{"span_id": lab2id.get(f.get("span")), "fact": f.get("fact", "")}
                                                 for f in obj.get("facts", []) if isinstance(f, dict)],
                question=str(obj.get("question", "")).strip(),
                correct_answer=str(obj.get("correct_answer", "")).strip(), distractors=dis,
                distractor_notes=[str(d.get("why_wrong", "")) for d in obj.get("distractors", []) if isinstance(d, dict)],
                gold_span_ids=list(dict.fromkeys(lab2id[x] for x in obj.get("gold_spans", []) if x in lab2id)),
                reasoning_steps=[{"span_id": lab2id.get(st.get("span")), "step": st.get("step", "")}
                                 for st in obj.get("reasoning_steps", []) if isinstance(st, dict)])
    if feasible and len(dis) == 3 and cand["correct_answer"]:
        cand["options"], cand["answer"] = assemble_options(cand["cand_id"], cand["correct_answer"], dis)
    return cand


# ------------------------------------------------------------------ deterministic checks
def structural_checks(c: dict[str, Any], corpus: SpanCorpus) -> dict[str, Any]:
    f: dict[str, Any] = {}
    one = c.get("min_spans") == 1                   # 1-hop (single-span) items
    f["feasible"] = bool(c.get("feasible"))
    opts = c.get("options") or {}
    texts = [norm(opts.get(k, "")) for k in LETTERS]
    f["four_distinct_options"] = len(opts) == 4 and all(texts) and len(set(texts)) == 4
    f["min_two_gold_spans"] = len(c.get("gold_span_ids") or []) >= (1 if one else 2)
    step_spans = {st["span_id"] for st in c.get("reasoning_steps", []) if st.get("span_id")}
    f["steps_use_two_spans"] = True if one else len(step_spans & set(c.get("gold_span_ids") or [])) >= 2
    q = c.get("question", "")
    f["no_meta_reference"] = not _META_REF.search(q) and not any(_META_REF.search(o or "") for o in opts.values())
    f["no_bad_option"] = not any(_BAD_OPTION.search(o or "") for o in opts.values())
    at = norm(c.get("correct_answer", ""))
    f["answer_not_in_question"] = not (len(at) >= 6 and at in norm(q))
    dl = [len(d) for d in c.get("distractors", [])] or [1]
    ratio = len(c.get("correct_answer", "")) / max(1.0, sum(dl) / len(dl))
    f["answer_len_ratio"] = round(ratio, 3)
    f["no_length_cue"] = ratio <= 1.8
    sims = [jaccard(char_ngrams(c.get("correct_answer", "")), char_ngrams(d)) for d in c.get("distractors", [])]
    f["distractors_distinct_from_answer"] = all(s < 0.9 for s in sims)
    f["gold_spans_not_exercise"] = all(not corpus.get(s).is_exercise() for s in c.get("gold_span_ids") or []
                                       if corpus.get(s) is not None)
    # verbatim copy: share of question syllable 6-grams found in gold span text
    qt = tokens(q)
    sh = {" ".join(qt[i:i + 6]) for i in range(max(0, len(qt) - 5))}
    gold_txt = " ".join(" ".join(tokens(corpus.get(s).text)) for s in c.get("gold_span_ids") or [] if corpus.get(s))
    f["copy_rate"] = round(sum(1 for x in sh if x in gold_txt) / len(sh), 3) if sh else 0.0
    f["q_len_ok"] = 25 <= len(q) <= 900
    hard = ("feasible", "four_distinct_options", "min_two_gold_spans", "steps_use_two_spans", "no_meta_reference",
            "no_bad_option", "answer_not_in_question", "no_length_cue", "distractors_distinct_from_answer",
            "gold_spans_not_exercise", "q_len_ok")
    f["structural_ok"] = all(f[k] for k in hard)
    f["structural_fail"] = [k for k in hard if not f[k]]
    return f


class LeakageIndex:
    """Near-duplicate detection against textbook exercise spans and another benchmark (MCQ300)."""

    def __init__(self, corpus: SpanCorpus, other_questions: Optional[list[tuple[str, str]]] = None):
        self.ex = [(s.span_id, s.domain, char_ngrams(s.text)) for s in corpus.spans if s.is_exercise()]
        self.other = [(i, char_ngrams(q)) for i, q in (other_questions or [])]

    def check(self, c: dict[str, Any], *, ex_thr: float = 0.5, other_thr: float = 0.7) -> dict[str, Any]:
        g = char_ngrams(c.get("question", ""))
        best_ex = max(((jaccard(g, e), sid) for sid, d, e in self.ex if d == c.get("domain")), default=(0.0, None))
        best_o = max(((jaccard(g, o), i) for i, o in self.other), default=(0.0, None))
        return {"exercise_sim": round(best_ex[0], 3), "exercise_span": best_ex[1], "other_bench_sim": round(best_o[0], 3),
                "other_bench_id": best_o[1], "leak_ok": best_ex[0] < ex_thr and best_o[0] < other_thr}


# ------------------------------------------------------------------ LLM checks
def check_verify(c: dict[str, Any], corpus: SpanCorpus, llm: LLM) -> dict[str, Any]:
    gold = sorted(c["gold_span_ids"], key=lambda s: sha1(c["cand_id"] + s))     # shuffled span order
    labels = [f"E{i + 1}" for i in range(len(gold))]
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + ":verify")
    p = _VERIFY_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)),
                              question=c["question"], options=fmt_options(opts))
    valid = lambda o: isinstance(o.get("option_status"), dict) and all(_status(o["option_status"].get(k)) for k in LETTERS)
    o, r = chat_json(llm, [{"role": "user", "content": p}], VERIFY_SCHEMA, max_tokens=400, validate=valid)
    st = {back[k]: _status(o["option_status"].get(k)) for k in LETTERS}         # back to presentation letters
    a = _letter(o.get("answer"))
    a = back.get(a) if a else "none"
    lab2 = dict(zip(labels, gold))
    used = [_label(x, labels) for x in o.get("used_spans") or []]
    return {"answer": a, "option_status": st, "confidence": o.get("confidence"),
            "used_span_ids": [lab2[x] for x in used if x], "model": llm.model,
            "prompt_version": VERIFY_PROMPT_VERSION}


QUALITY_SCHEMA = {"type": "object", "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5}
                                                   for k in ("natural", "needs_both", "distractors")},
                  "required": ["natural", "needs_both", "distractors"]}


def check_quality(c: dict[str, Any], corpus: SpanCorpus, llm: LLM) -> dict[str, Any]:
    gold = list(c["gold_span_ids"])
    labels = [f"E{i + 1}" for i in range(len(gold))]
    p = _QUALITY_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)),
                               question=c["question"], options=fmt_options(c["options"]), answer=c["answer"])
    def num(o, k):
        try:
            return int(float(o.get(k)))
        except (TypeError, ValueError):
            return None
    o, r = chat_json(llm, [{"role": "user", "content": p}], QUALITY_SCHEMA, max_tokens=120,
                     validate=lambda o: all(num(o, k) is not None for k in QUALITY_SCHEMA["required"]))
    return {**{k: num(o, k) for k in QUALITY_SCHEMA["required"]}, "model": llm.model,
            "prompt_version": QUALITY_PROMPT_VERSION}


def check_single(c: dict[str, Any], span_id: str, corpus: SpanCorpus, llm: LLM) -> dict[str, Any]:
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + ":single:" + span_id)
    p = _SINGLE_PROMPT.format(block=span_block("E1", corpus.get(span_id)), question=c["question"], options=fmt_options(opts))
    o, r = chat_json(llm, [{"role": "user", "content": p}], SINGLE_SCHEMA, max_tokens=300,
                     validate=lambda o: "sufficient" in o)
    a = _letter(o.get("answer"))
    return {"sufficient": _bool(o.get("sufficient")), "answer": back.get(a) if a else "none",
            "missing": o.get("missing", ""), "model": llm.model, "prompt_version": SINGLE_PROMPT_VERSION}


def check_closed(c: dict[str, Any], llm: LLM, perm: int) -> dict[str, Any]:
    opts, ans, back = (c["options"], c["answer"], {k: k for k in LETTERS}) if perm == 0 else \
        permute(c["options"], c["answer"], c["cand_id"] + f":closed:{perm}")
    head = f"Môn: {DOMAIN_TO_SUBJECT_VI.get(c['domain'], c['domain'])}; lớp {c.get('grade') or '?'}\n"
    msgs = [{"role": "system", "content": _CLOSED_SYS},
            {"role": "user", "content": f"{head}Câu hỏi: {c['question']}\n{fmt_options(opts)}\n\nChọn một đáp án đúng "
                                        'nhất. Trả về JSON: {"answer": "<A|B|C|D>", "confidence": <0-100>}'}]
    o, r = chat_json(llm, msgs, CLOSED_SCHEMA, max_tokens=64, validate=lambda o: _letter(o.get("answer")) is not None)
    a = _letter(o.get("answer"))
    conf = o.get("confidence")
    try:
        conf = int(float(conf))
    except (TypeError, ValueError):
        conf = None
    return {"answer": back.get(a) if a else None, "confidence": conf, "model": llm.model,
            "prompt_version": CLOSED_PROMPT_VERSION}


# ------------------------------------------------------------------ resumable stores
class JsonlStore:
    """Append-only JSONL keyed by `key_fn(row)`; later rows win; rows with `error` are retried."""

    def __init__(self, path: Path, key_fn: Callable[[dict], str]):
        self.path = Path(path)
        self.key_fn = key_fn
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[key_fn(r)] = r

    def done(self, key: str) -> bool:
        r = self.rows.get(key)
        return r is not None and not r.get("error")

    def add(self, row: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows[self.key_fn(row)] = row


def run_generation(seeds: list[Seed], corpus: SpanCorpus, llm: LLM, store: JsonlStore, *, workers: int = 8,
                   temperature: float = 0.7, extra_fn: Optional[Callable[[Seed], str]] = None,
                   prompt_version: str = GEN_PROMPT_VERSION) -> None:
    def one(sd: Seed):
        store.add(generate_candidate(sd, corpus, llm, temperature=temperature,
                                     extra_prompt=extra_fn(sd) if extra_fn else "", prompt_version=prompt_version))
    todo = [s for s in seeds if not store.done("cand_" + sha1(s.seed_id + prompt_version + llm.model)[:12])]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, todo))


def run_checks(cands: list[dict], corpus: SpanCorpus, store: JsonlStore, *, main: Optional[LLM] = None,
               small: Optional[LLM] = None, workers: int = 8, closed_perms: int = 2,
               closed_main: bool = True, light: bool = False) -> None:
    """Runs every LLM check not yet in `store` (key = cand_id|check). `store` rows: {cand_id, check, result}."""
    jobs: list[tuple[dict, str, Callable[[], dict]]] = []
    for c in cands:
        cid = c["cand_id"]
        if main is not None:
            jobs.append((c, "verify", lambda c=c: check_verify(c, corpus, main)))
            if not light:
                jobs.append((c, "quality", lambda c=c: check_quality(c, corpus, main)))
                for s in c["gold_span_ids"]:
                    jobs.append((c, f"single:{s}", lambda c=c, s=s: check_single(c, s, corpus, main)))
            if closed_main:
                for p in range(closed_perms):
                    jobs.append((c, f"closed:main:{p}", lambda c=c, p=p: check_closed(c, main, p)))
        if small is not None:
            for p in range(closed_perms):
                jobs.append((c, f"closed:small:{p}", lambda c=c, p=p: check_closed(c, small, p)))
    jobs = [j for j in jobs if not store.done(f"{j[0]['cand_id']}|{j[1]}")]

    def one(job):
        c, name, fn = job
        try:
            row = {"cand_id": c["cand_id"], "check": name, "result": fn()}
        except Exception as e:
            row = {"cand_id": c["cand_id"], "check": name, "error": f"{type(e).__name__}: {e}"}
        store.add(row)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, jobs))


def check_key(row: dict) -> str:
    return f"{row['cand_id']}|{row['check']}"


def summarize_checks(c: dict[str, Any], checks: dict[str, dict], *, max_undetermined: int = 0,
                     easy_conf: int = 80) -> dict[str, Any]:
    """Fold check records of one candidate into filter decisions + difficulty."""
    cid = c["cand_id"]
    get = lambda name: (checks.get(f"{cid}|{name}") or {}).get("result")
    out: dict[str, Any] = {}
    v = get("verify")
    if v is None:
        out["verify_ok"] = None
    else:
        st = v.get("option_status") or {}
        dis = [k for k in LETTERS if k != c["answer"]]
        n_wrong_correct = sum(1 for k in dis if st.get(k) == "correct")
        n_undet = sum(1 for k in dis if st.get(k) == "undetermined")
        out.update(verify_answer=v.get("answer"), verify_conf=v.get("confidence"),
                   verify_key_correct=st.get(c["answer"]) == "correct", verify_n_other_correct=n_wrong_correct,
                   verify_n_undetermined=n_undet,
                   verify_ok=v.get("answer") == c["answer"] and st.get(c["answer"]) == "correct"
                   and n_wrong_correct == 0 and n_undet <= max_undetermined)
    q = get("quality")
    if q is not None:
        out.update(q_natural=q["natural"], q_needs_both=q["needs_both"], q_distractors=q["distractors"],
                   quality_ok=q["natural"] >= 4 and q["needs_both"] >= 4 and q["distractors"] >= 3)
    singles = {s: get(f"single:{s}") for s in c["gold_span_ids"]}
    if c.get("min_spans") == 1:
        out["multi_hop_ok"] = True
    elif all(r is not None for r in singles.values()) and singles:
        suff = [s for s, r in singles.items() if r["sufficient"] and r["answer"] == c["answer"]]
        out["single_span_sufficient"] = suff
        out["multi_hop_ok"] = not suff
    else:
        out["multi_hop_ok"] = None
    for role in ("small", "main"):
        rs = [get(f"closed:{role}:{p}") for p in range(4)]
        rs = [r for r in rs if r is not None]
        if rs:
            ok = [r["answer"] == c["answer"] for r in rs]
            conf = [r.get("confidence") or 0 for r in rs]
            out[f"closed_{role}_correct"] = sum(ok)
            out[f"closed_{role}_n"] = len(rs)
            out[f"closed_{role}_conf"] = round(sum(conf) / len(conf), 1)
    if "closed_small_n" in out:
        k, n, cf = out["closed_small_correct"], out["closed_small_n"], out["closed_small_conf"]
        out["difficulty"] = "hard" if k == 0 else ("easy" if k == n and cf >= easy_conf else "medium")
    return out
