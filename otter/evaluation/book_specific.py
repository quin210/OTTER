"""ViEduHop `book_specific` pool: HARD / TRICKY items that test what THIS textbook says (table values, the book's own
examples and experiments, its classification wording and numbering, conditions/exceptions, negation stems,
statement-combination stems, literature excerpt details, history dates and roles) and not general school knowledge.

Pipeline (see scripts/eval/build_otter_hard_bs.py; every stage is resumable):
  seeds -> generate (<=2 items per call) -> cheap structural/leak/arithmetic/number-grounding/length filters ->
  [8B prescreen on raw options for items whose distractors must be rewritten] -> length-matched copy `_F` ->
  CLOSED-BOOK PROBE (small model, then main model, no retrieval, 4 option permutations; keep both accuracies <= 0.5 =
  the `closed_book_hard` rule) -> verification on survivors only (32B with gold spans over 2 permutations + independent
  8B check, single-span sufficiency for >=2 hops, per-statement check for statement-combination stems, quality) ->
  dedup (lexical + bge-m3 question embedding) -> enrichment -> annotated_pool.jsonl.

Nothing here relaxes the correctness filters of the other pools: key entailed by and unique in the gold spans, gold
spans never exercise/question spans, no multi-hop item answerable from a single gold span.
"""
from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from typing import Any, Optional

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import (_GEN_SYS, LETTERS, _bool, _label, assemble_options, chat_json, fmt_options,
                                       permute, span_block)
from otter.evaluation.hard_seeds import HOP_TYPES, Seed
from otter.evaluation.hard_v2 import BLOOM, eligible_rule_span, length_ok, parse_numbers, unique_longest
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI
from otter.evaluation.text import norm, sha1, short_hash, tokens

BS_GEN_VERSION = "otter_bs_gen_v4"
BS_SEEDS_VERSION = "otter_bs_seeds_v1"
BS_STATEMENT_VERSION = "otter_bs_statement_check_v1"
BS_INDEP_VERSION = "otter_bs_indep_verify_v1"

TRICK_TYPES = ["table_lookup", "table_calc", "example_case", "classification_wording", "condition_exception",
               "negation", "statement_combo", "cross_lesson", "lit_excerpt", "hist_fact"]

TRICK_GUIDE = {
    "table_lookup": "TRA CỨU/SO SÁNH SỐ LIỆU RIÊNG CỦA SÁCH: hỏi một con số, đơn vị, thứ hạng, giá trị lớn nhất/nhỏ nhất, hiệu/tỉ lệ giữa các "
                    "dòng hoặc mốc (bảng, số liệu, mốc thời gian) CHỈ có trong sách này; nhiễu là số liệu của dòng/cột/mốc kề bên hoặc con số "
                    "gần đúng hợp lí. Chép CHÍNH XÁC chữ số từ đoạn trích.",
    "table_calc": "TÍNH TOÁN VỚI GIÁ TRỊ CỦA SÁCH: câu hỏi cần dùng một hằng số/số liệu/công thức/hệ số cụ thể mà sách nêu (đưa số liệu cụ "
                  "thể vào câu hỏi; giá trị riêng của sách thì hỏi dựa vào sách). Đáp án là kết quả số có đơn vị; nhiễu là kết quả khi dùng "
                  "sai số liệu/sai bước/sai đơn vị. BẮT BUỘC có trường \"calc\": {\"expression\": \"<biểu thức số học thuần túy gồm số và + - * / ** ( )>\", "
                  "\"unit\": \"...\"} cho ĐÚNG giá trị số của đáp án đúng.",
    "example_case": "VÍ DỤ / THÍ NGHIỆM / TÌNH HUỐNG CỦA CHÍNH SÁCH: hỏi chi tiết (đối tượng, cách bố trí, điều kiện, quan sát, kết quả, kết luận) của "
                    "ví dụ, thí nghiệm hay tình huống mà sách trình bày, KHÔNG phải phiên bản chung chung ai cũng biết; nhiễu là kết quả/chi tiết "
                    "của phiên bản quen thuộc khác hoặc của ví dụ khác.",
    "classification_wording": "CÁCH PHÂN LOẠI / ĐỊNH NGHĨA / SẮP XẾP RIÊNG CỦA SÁCH: hỏi sách chia thành mấy nhóm/loại/giai đoạn, gọi tên là gì, sắp theo thứ "
                              "tự nào, hoặc câu định nghĩa dùng những ý nào; nhiễu là cách chia/cách gọi chuẩn khác thường gặp hoặc của bài lân cận "
                              "(xem NGỮ CẢNH LÂN CẬN) nhưng KHÁC với sách này.",
    "condition_exception": "ĐIỀU KIỆN / NGOẠI LỆ / GIỚI HẠN: hỏi trường hợp NÀO không áp dụng, điều kiện cần/đủ nào, giới hạn nào của định luật/quy tắc/kết luận "
                           "theo sách; nhiễu là điều kiện sai, điều kiện của quy tắc khác hoặc bỏ sót điều kiện.",
    "negation": "CÂU PHỦ ĐỊNH: \"Phát biểu nào sau đây KHÔNG đúng theo sách giáo khoa …?\" hoặc \"… NGOẠI TRỪ …\". Đáp án đúng là MỘT phát biểu SAI "
                "(sai một chi tiết nhỏ, có thể đối chiếu với đoạn trích); ba phương án còn lại là phát biểu ĐÚNG theo đoạn trích, mỗi phát biểu một "
                "chi tiết khác nhau. In đậm từ KHÔNG bằng chữ hoa trong câu hỏi.",
    "statement_combo": "TỔ HỢP PHÁT BIỂU kiểu đề thi THPT: viết đúng 4 phát biểu (I, II, III, IV) về cùng chủ đề, mỗi phát biểu kiểm tra một chi tiết khác "
                       "nhau trong đoạn; phát biểu SAI phải sai ở một chi tiết cụ thể trái với đoạn (đổi số liệu, đổi điều kiện, đảo quan hệ), không sai mơ hồ. "
                       "Có từ 1 đến 4 phát biểu đúng. KHÔNG tự viết câu hỏi/phương án: chỉ trả về chủ đề (\"stem\") và danh sách \"statements\" "
                       "[{\"text\": \"...\", \"true\": true|false, \"span\": \"S1\", \"quote\": \"<câu nguyên văn trong đoạn làm căn cứ>\"}] .",
    "cross_lesson": "LIÊN KẾT HAI/BA BÀI: câu hỏi chỉ trả lời được khi kết hợp thông tin riêng của sách ở các đoạn khác nhau (đoạn nêu quy tắc + đoạn nêu số liệu/ví dụ; "
                    "hoặc so sánh hai nội dung ở hai bài); mỗi đoạn trong gold_spans đều PHẢI cần thiết.",
    "lit_excerpt": "NGỮ VĂN — CHI TIẾT TRONG NGỮ LIỆU IN TRONG SÁCH: hỏi nhân vật, sự việc, chi tiết, từ ngữ, hình ảnh, hoặc thông tin về tác giả/hoàn cảnh sáng tác "
                   "NHƯ ĐƯỢC NÊU trong đoạn (không phải kiến thức văn học chung); nhiễu là chi tiết hợp lí nhưng sách không nói hoặc nói khác.",
    "hist_fact": "LỊCH SỬ — NIÊN ĐẠI/SỰ KIỆN/VAI TRÒ NHƯ SÁCH TRÌNH BÀY: hỏi mốc thời gian, địa điểm, con số, vai trò hoặc thứ tự cụ thể được sách nêu; nhiễu là mốc/số "
                 "liệu/nhân vật gần đúng từ sự kiện kề bên cũng có trong lịch sử.",
}

_COND = re.compile(r"(điều kiện|chỉ khi|chỉ (?:xảy ra|có thể|được)|trừ |ngoại trừ|không áp dụng|giới hạn|tuy nhiên|nếu không|khi và chỉ khi|"
                   r"miễn là|với điều kiện|không phải|ngoại lệ|trường hợp|chỉ có)", re.I)
_CLASS = re.compile(r"(gồm|bao gồm|chia thành|phân (?:chia|loại|thành)|được chia|(?:hai|ba|bốn|năm|sáu|bảy|\d) (?:loại|dạng|kiểu|nhóm|giai đoạn|"
                    r"bước|thành phần|pha|kì|giai|yếu tố|nguyên tắc|đặc điểm|đặc trưng|biện pháp|nhiệm vụ))", re.I)
_EXPR = re.compile(r"(thí nghiệm|quan sát|kết quả|ví dụ|nghiên cứu|thực nghiệm|câu chuyện|người ta|năm \d{4}|nhà (?:khoa học|bác học))", re.I)
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_UNIT = re.compile(r"\d\s*(?:m/s|km|kg|mol|°C|K\b|Pa|atm|J\b|N\b|W\b|V\b|A\b|Hz|cm|mm|nm|%|g/|lít|L\b|mL|kJ|eV|MeV|tấn|triệu|nghìn|tỉ|ha\b)", re.I)
_OCR_SUSPECT = re.compile(r"\d[OolI]\d|(?<![A-Za-zÀ-ỹ])[OolI]\d{2,}\b|\d{1,3}[OIl](?![A-Za-zÀ-ỹ])")
_LISTLINE = re.compile(r"^\s*(?:[-–•+*]|\d+[.)]|[a-zđ][.)]|[IVX]+\.)\s+\S", re.M)


def span_nums(text: str) -> int:
    t = _LISTLINE.sub("", text)
    return len(re.findall(r"(?<![\w])\d+(?:[.,]\d+)?", t))


def is_table(s: Span) -> bool:
    t = s.text
    return "|" in t and "---" in t and t.count("?") < 2 and t.count("\n") >= 3 and span_nums(t) >= 2


def trick_scores(s: Span) -> dict[str, float]:
    """Heuristic archetype affinity of one expository span (0 = unsuitable)."""
    t = s.text
    n = len(t)
    nn = span_nums(t)
    sc: dict[str, float] = {}
    if is_table(s):
        sc["table_lookup"] = 10
    elif nn >= 5 and n <= 1400:
        sc["table_lookup"] = 3 + min(nn, 12) / 4
    if s.domain in ("physics", "chemistry") and nn >= 3 and (_UNIT.search(t) or "=" in t):
        sc["table_calc"] = 4 + min(nn, 10) / 5
    if s.pedagogical_type in ("example", "worked_example", "process") or len(_EXPR.findall(t)) >= 2:
        sc["example_case"] = 3 + len(_EXPR.findall(t)) + (3 if s.pedagogical_type in ("example", "worked_example") else 0)
    if _CLASS.search(t) or len(_LISTLINE.findall(t)) >= 3:
        sc["classification_wording"] = 4 + len(_CLASS.findall(t)) + len(_LISTLINE.findall(t)) / 2
    if _COND.search(t) or s.pedagogical_type in ("law", "condition", "principle"):
        sc["condition_exception"] = 3 + len(_COND.findall(t)) + (3 if s.pedagogical_type in ("law", "condition", "principle") else 0)
    if n >= 220 and (_CLASS.search(t) or t.count(",") >= 3 or len(_LISTLINE.findall(t)) >= 3):
        sc["negation"] = 3 + t.count(",") / 4
    if n >= 280:
        sc["statement_combo"] = 3 + n / 400
    if s.domain == "literature" and n >= 150:
        sc["lit_excerpt"] = 5 + n / 500
    if s.domain == "history" and (len(_YEAR.findall(t)) >= 1 or nn >= 2):
        sc["hist_fact"] = 5 + len(_YEAR.findall(t))
    return sc


# ------------------------------------------------------------------ seeds
def span_ok(s: Span, pageq: dict, *, min_chars: int = 110, max_chars: int = 2000, min_q: float = 0.9) -> bool:
    if not eligible_rule_span(s, min_chars) or len(s.text) > max_chars or s.pedagogical_type in ("summary", "unknown"):
        return False
    q = pageq.get((s.book_id, s.pdf_page_index))
    if q is not None and q < min_q:
        return False
    if _OCR_SUSPECT.search(s.text) or re.search(r"(\.{5,}|…{3,})", s.text):
        return False
    return not s.text.lstrip().startswith(("[", "!["))


def _mk(span_ids: list[str], corpus: SpanCorpus, trick_types: list[str], ctx: list[str], tag: str) -> Seed:
    ss = [corpus.get(i) for i in span_ids]
    grades = [s.grade for s in ss]
    return Seed(seed_id="seed_" + short_hash(span_ids + trick_types + [tag]), strategy="book_specific", span_ids=span_ids,
                domain=ss[0].domain, grade=max((g for g in grades if g is not None), default=None), grades=grades,
                book_ids=list(dict.fromkeys(s.book_id for s in ss)), lesson_keys=list(dict.fromkeys(s.lesson_key for s in ss)),
                target_hop_type="single" if len(ss) == 1 else "bridge", compatible_hop_types=[h for h in HOP_TYPES],
                link={"trick_types": trick_types, "ctx_span_ids": ctx, "hops": len(ss)}, version=BS_SEEDS_VERSION)


def build_seeds(corpus: SpanCorpus, pageq: dict, span_concepts: dict[str, set[str]], *, round: int = 0, domains=None,
                max_per_domain: Optional[int] = None, hop_mix: tuple[float, float, float] = (0.58, 0.30, 0.12), seed: int = 7,
                min_chars: int = 90, term_index=None) -> list[Seed]:
    """One seed per eligible expository span and round: round r uses the r-th best-matching archetype of the span (so successive
    rounds give DIFFERENT questions on the same text); the hop count is drawn from `hop_mix` (partners: same lesson, or concept-linked
    spans of another lesson of the same book, ranked by shared high-IDF terms)."""
    import random
    rng = random.Random(seed * 1000 + round)
    from otter.evaluation.hard_seeds import TermIndex
    elig = [s for s in corpus.spans if s.domain and span_ok(s, pageq, min_chars=min_chars) and (not domains or s.domain in domains)]
    by_dom: dict[str, list[Span]] = defaultdict(list)
    for s in elig:
        by_dom[s.domain].append(s)
    ti = term_index or TermIndex(elig)
    by_lesson: dict[str, list[Span]] = defaultdict(list)
    for s in elig:
        by_lesson[s.lesson_key].append(s)
    all_in_lesson: dict[str, list[Span]] = defaultdict(list)
    for s in corpus.spans:
        if s.domain and not s.is_exercise() and len(s.text) >= 60:
            all_in_lesson[s.lesson_key].append(s)
    concept_idx: dict[str, list[Span]] = defaultdict(list)
    for s in elig:
        for c in span_concepts.get(s.span_id, ()):
            concept_idx[c].append(s)
    out: list[Seed] = []
    for dom, spans in sorted(by_dom.items()):
        order = list(spans)
        rng.shuffle(order)
        n_dom = 0
        for s in order:
            if max_per_domain and n_dom >= max_per_domain:
                break
            sc = trick_scores(s)
            if not sc:
                continue
            ranked = [tt for _, tt in sorted(((v + rng.random() * 2, tt) for tt, v in sc.items()), reverse=True)]
            tts = [ranked[round % len(ranked)]] + ([ranked[(round + 1) % len(ranked)]] if len(ranked) > 1 else [])
            tts = list(dict.fromkeys(tts))
            u = rng.random()
            hop = 1 if u < hop_mix[0] else (2 if u < hop_mix[0] + hop_mix[1] else 3)
            ids = [s.span_id]
            if hop >= 2:
                same = [x for x in by_lesson.get(s.lesson_key, []) if x.span_id != s.span_id]
                cross = []
                for c in span_concepts.get(s.span_id, ()):
                    cross += [x for x in concept_idx.get(c, []) if x.lesson_key != s.lesson_key and x.book_id == s.book_id]
                pool = []
                for x in same + cross:
                    sh = ti.shared(s, x)
                    if sh:
                        pool.append((sum(w for _, w in sh[:3]) + rng.random(), x))
                pool.sort(key=lambda z: -z[0])
                chosen: list[Span] = []
                for _, x in pool[:8]:
                    if len(chosen) < hop - 1 and all(x.span_id != y.span_id for y in chosen):
                        chosen.append(x)
                ids += [x.span_id for x in chosen]
                if len(ids) > 1:
                    tts = ["cross_lesson"] + [tt for tt in tts if tt not in ("statement_combo", "cross_lesson")][:1]
            ctx = [x.span_id for x in all_in_lesson.get(s.lesson_key, []) if x.section_key != s.section_key and x.span_id not in ids]
            rng.shuffle(ctx)
            out.append(_mk(ids, corpus, tts, ctx[:2], f"r{round}"))
            n_dom += 1
    return out


# ------------------------------------------------------------------ prompts
_GEN_PROMPT = """Môn: {subject}. Lớp: {grades}.
Dưới đây là {k} đoạn trích từ sách giáo khoa ({where}).

{blocks}
{ctx}
MỤC TIÊU: viết các câu hỏi trắc nghiệm KHÓ, "lắt léo" nhưng CÔNG BẰNG, kiểm tra điều mà CHÍNH QUYỂN SÁCH NÀY nói, để một học sinh (hoặc một mô hình) chỉ dựa vào kiến thức phổ thông bên ngoài sách KHÔNG chắc chắn trả lời đúng. Câu hỏi phải tự đứng được; có thể ghi "theo sách giáo khoa {subject} lớp {grade_one}" để xác định nguồn; KHÔNG nhắc "đoạn trích", "S1", "văn bản trên".

KIỂU CÂU HỎI ĐƯỢC PHÉP (chọn kiểu phù hợp nhất với đoạn, mỗi câu một kiểu; có thể viết 1 hoặc 2 câu hỏi KHÁC NHAU):
{guides}

ĐIỀU LÀM CHO CÂU HỎI "LẮT LÉO" (bắt buộc áp dụng): đáp án đúng là một CHI TIẾT CỤ THỂ của sách này (con số, tên gọi, thứ tự, mốc, điều kiện, chi tiết trong ví dụ/thí nghiệm) mà người chỉ biết kiến thức phổ thông sẽ dễ chọn nhầm; phương án nhiễu nên chứa chính cách hiểu/con số PHỔ BIẾN hoặc cách trình bày chuẩn khác mà sách này KHÔNG dùng, hoặc chi tiết ở dòng/ví dụ/bài kề bên. Tránh câu hỏi mà phương án đúng là điều hiển nhiên hoặc chung chung (\"quan trọng\", \"góp phần phát triển\").

YÊU CẦU BẮT BUỘC:
0. Câu hỏi không được hỏi "đoạn trích/văn bản trên nói gì": với Ngữ văn hãy nêu TÊN văn bản/tác phẩm (lấy từ tiêu đề trong ngoặc ở đầu mỗi đoạn) và tên nhân vật/tác giả cụ thể.
1. Đáp án đúng DUY NHẤT và có căn cứ rõ ràng trong các đoạn trích; không dùng kiến thức ngoài sách; không hỏi chi tiết vụn vặt (số hình, số trang, chú thích ảnh).
2. NĂM phương án nhiễu (hệ thống sẽ chọn ra 3 cái có độ dài sát đáp án nhất) "gần đúng" (near-miss) hợp lí, CHẮC CHẮN SAI theo đoạn trích, cùng dạng, cùng cấu trúc câu và cùng mức chi tiết với đáp án đúng; độ dài (số ký tự) mỗi nhiễu trong khoảng ±15% so với đáp án đúng; không dùng "tất cả đều đúng/sai", "cả A và B".
3. Các con số/đơn vị chép CHÍNH XÁC từ đoạn trích. Với mỗi đoạn dùng làm căn cứ, trích nguyên văn (đúng từng chữ) một cụm 8-30 âm tiết vào "evidence".
4. Câu hỏi nhiều đoạn: mỗi đoạn trong "gold_spans" phải thực sự cần thiết; nếu một đoạn là đủ thì chỉ liệt kê đoạn đó.
5. Nếu đoạn không đủ nội dung đặc thù của sách cho kiểu nào, trả "items": [] .

Trả về JSON:
{{"items": [{{"trick_type": "<{tt_enum}>", "hop_type": "<{hop_enum}>", "question": "...", "correct_answer": "...",
 "distractors": ["...", "...", "...", "...", "..."],
 "gold_spans": ["S1"], "evidence": {{"S1": "<cụm nguyên văn>"}}, "bloom": "nhận biết|thông hiểu|vận dụng", "calc": null}}]}}
Với kiểu statement_combo thay các trường question/correct_answer/distractors bằng "stem" và "statements" như mô tả (vẫn giữ trick_type, gold_spans, evidence)."""

_LIT_NOTE = ("LƯU Ý NGỮ VĂN: chỉ hỏi NỘI DUNG được nêu rõ trong đoạn (nhân vật, sự việc, chi tiết, từ ngữ, tác giả, hoàn cảnh sáng tác); không yêu cầu cảm thụ "
             "hay diễn giải riêng, không bịa ý nghĩa ngoài đoạn.")


def trick_hop_type(tt: str, n_gold: int, given: str) -> str:
    if n_gold == 1:
        return "single"
    if tt == "table_calc":
        return "calculation"
    return given if given in HOP_TYPES and given != "single" else "bridge"


def _stmt_roman(i: int) -> str:
    return ["I", "II", "III", "IV"][i]


COMBOS = {1: [(0,), (1,), (2,), (3,)], 2: list(itertools.combinations(range(4), 2)), 3: list(itertools.combinations(range(4), 3))}


def _combo_text(c) -> str:
    r = [_stmt_roman(i) for i in c]
    return r[0] if len(r) == 1 else ", ".join(r[:-1]) + " và " + r[-1]


def assemble_statement_item(stem: str, statements: list[dict], key: str) -> Optional[dict]:
    """Deterministic assembly of an I-IV statement-combination item. Returns question/options/answer/correct/distractors."""
    if len(statements) != 4:
        return None
    truth = [bool(s.get("true")) for s in statements]
    n = sum(truth)
    if n == 0:
        return None
    form = "count" if int(sha1(key + "form")[:4], 16) % 2 == 0 or n == 4 else "combo"
    body = "\n".join(f"{_stmt_roman(i)}. {str(s['text']).strip()}" for i, s in enumerate(statements))
    stem = stem.strip().rstrip(":") + ":"
    if form == "count":
        q = f"{stem}\n{body}\nCó bao nhiêu phát biểu đúng?"
        correct = str(n)
        dis = [str(x) for x in (1, 2, 3, 4) if x != n]
    else:
        q = f"{stem}\n{body}\nNhững phát biểu đúng là"
        true_c = tuple(i for i, t in enumerate(truth) if t)
        correct = _combo_text(true_c)
        others = [c for c in COMBOS[n] if c != true_c]
        others.sort(key=lambda c: sha1(key + str(c)))
        dis = [_combo_text(c) for c in others[:3]]
        if len(dis) < 3:
            return None
    return {"question": q, "correct_answer": correct, "distractors": dis, "statement_form": form, "n_true": n}


def _norm_quote(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"“”‘’'`*_]", "", t)).strip().lower()


def quote_in_span(q: str, text: str, min_cov: float = 0.7) -> bool:
    """Verbatim (modulo whitespace/quote marks/case) or near-verbatim: >= min_cov of the quote's syllable bigrams occur in the span."""
    qq = _norm_quote(q)
    if len(qq) < 10:
        return False
    if qq in _norm_quote(text):
        return True
    qt, tt = tokens(q), tokens(text)
    if len(qt) < 5:
        return False
    tb = {(a, b) for a, b in zip(tt, tt[1:])}
    qb = [(a, b) for a, b in zip(qt, qt[1:])]
    return sum(1 for x in qb if x in tb) / len(qb) >= min_cov


def pick_distractors(key: str, ds: list[str], salt: str) -> list[str]:
    """Choose 3 of the offered distractors whose lengths are within +-20% of the key's and not all shorter than it (the key must not be
    the unique longest option); falls back to the 3 closest in length (the length rewrite repairs it later)."""
    if len(ds) <= 3:
        return ds
    n = max(1, len(key))
    best = None
    for combo in itertools.combinations(range(len(ds)), 3):
        sel = [ds[i] for i in combo]
        spread = max(abs(len(d) / n - 1) for d in sel)
        ok = length_ok(key, sel) and max(len(d) for d in sel) >= n
        score = (0 if ok else 1, round(spread, 2), sha1(salt + "".join(sel)))
        if best is None or score < best[0]:
            best = (score, sel)
    return best[1]


def generate_bs(seed: Seed, corpus: SpanCorpus, llm, *, temperature: float = 0.8, max_tokens: int = 2200) -> list[dict]:
    """One LLM call -> up to 2 candidate records (cand_id = hash(seed, prompt version, model, index))."""
    spans = [corpus.get(i) for i in seed.span_ids]
    labels = [f"S{i + 1}" for i in range(len(spans))]
    tts = [t for t in (seed.link or {}).get("trick_types", []) if t in TRICK_GUIDE]
    if len(spans) > 1 and "cross_lesson" not in tts:
        tts.insert(0, "cross_lesson")
    allowed = list(dict.fromkeys(tts))
    guides = "\n".join(f"- {t}: {TRICK_GUIDE[t]}" for t in allowed)
    ctx_spans = [corpus.get(i) for i in (seed.link or {}).get("ctx_span_ids", []) if corpus.get(i)]
    ctx = ""
    if ctx_spans:
        ctx = ("\nNGỮ CẢNH LÂN CẬN (các phần khác của cùng bài; KHÔNG dùng làm căn cứ đáp án đúng, chỉ dùng để viết phương án nhiễu gần đúng, có thể lấy ý "
               "từ đây nhưng phải sai theo S1…):\n" + "\n".join(f"- {c.text.strip()[:350]}" for c in ctx_spans) + "\n")
    grades = sorted({s.grade for s in spans if s.grade is not None})
    s0 = spans[0]
    where = f"{s0.book_title or s0.book_id}; " + " › ".join(s0.heading_path[-2:])
    note = ("\n" + _LIT_NOTE) if seed.domain == "literature" else ""
    prompt = _GEN_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI.get(seed.domain, seed.domain), grades=", ".join(map(str, grades)) or "?",
                                grade_one=grades[-1] if grades else "", k=len(spans), where=where,
                                blocks="\n\n".join(span_block(l, s, 1900) for l, s in zip(labels, spans)), ctx=ctx,
                                guides=guides, tt_enum="|".join(allowed), hop_enum="|".join(h for h in HOP_TYPES if h != "single")) + note
    base = {"seed_id": seed.seed_id, "domain": seed.domain, "grade": seed.grade, "seed_span_ids": seed.span_ids,
            "strategy": "book_specific", "book_specific": True,
            "generator": {"model": llm.model, "prompt_version": BS_GEN_VERSION, "temperature": temperature}}
    try:
        o, r = chat_json(llm, [{"role": "system", "content": _GEN_SYS}, {"role": "user", "content": prompt}], {}, max_tokens=max_tokens,
                         temperature=temperature, validate=lambda o: isinstance(o.get("items"), list))
    except Exception as e:  # noqa: BLE001
        return [dict(base, cand_id="cand_" + sha1(seed.seed_id + BS_GEN_VERSION + llm.model + "0")[:12], feasible=False,
                     error=f"{type(e).__name__}: {e}")]
    lab2id = dict(zip(labels, seed.span_ids))
    out = []
    for idx, it in enumerate([x for x in o["items"] if isinstance(x, dict)][:2]):
        cid = "cand_" + sha1(seed.seed_id + BS_GEN_VERSION + llm.model + str(idx))[:12]
        c = dict(base, cand_id=cid, item_index=idx, generator=dict(base["generator"], prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens))
        tt = it.get("trick_type") if it.get("trick_type") in TRICK_GUIDE else (allowed[0] if allowed else "cross_lesson")
        gold = [lab2id[x] for x in dict.fromkeys(_label(g, labels) for g in it.get("gold_spans") or []) if x in lab2id]
        ev = {lab2id[l]: str(v) for l, v in ((_label(k, labels), v) for k, v in (it.get("evidence") or {}).items()) if l in lab2id and isinstance(v, str)}
        c.update(trick_type=tt, gold_span_ids=gold, evidence=ev, link=str(it.get("link", "")), bloom_target=it.get("bloom") if it.get("bloom") in BLOOM else None,
                 calc=it.get("calc") if isinstance(it.get("calc"), dict) else None, min_spans=1 if len(gold) == 1 else 2,
                 hop_type=trick_hop_type(tt, len(gold), str(it.get("hop_type", ""))), facts=[],
                 reasoning_steps=[{"span_id": g, "step": ev.get(g, "")} for g in gold])
        if tt == "statement_combo":
            sts = [x for x in it.get("statements") or [] if isinstance(x, dict) and str(x.get("text", "")).strip()]
            for x in sts:
                x["true"] = _bool(x.get("true"))
                x["span_id"] = lab2id.get(_label(x.get("span"), labels))
            c["statements"] = sts
            asm = assemble_statement_item(str(it.get("stem", "")), sts, cid) if str(it.get("stem", "")).strip() else None
            if asm:
                c.update(question=asm["question"], correct_answer=asm["correct_answer"], distractors=asm["distractors"],
                         statement_form=asm["statement_form"], feasible=True)
            else:
                c.update(feasible=False, question="", correct_answer="", distractors=[])
        else:
            raw = [(str((d.get("text") if isinstance(d, dict) else d) or "").strip(), str(d.get("why_wrong", "")) if isinstance(d, dict) else "")
                   for d in it.get("distractors") or []]
            key_txt = str(it.get("correct_answer", "")).strip()
            raw = [x for i, x in enumerate(raw) if x[0] and x[0].lower() != key_txt.lower() and x[0].lower() not in {y[0].lower() for y in raw[:i]}]
            pick = pick_distractors(key_txt, [x[0] for x in raw], cid)
            dis = pick
            notes = [dict(raw).get(d, "") for d in pick]
            c.update(question=str(it.get("question", "")).strip(), correct_answer=key_txt, distractors=dis[:3], distractors_offered=[x[0] for x in raw],
                     distractor_notes=notes,
                     feasible=bool(str(it.get("question", "")).strip()) and len(dis) >= 3 and bool(str(it.get("correct_answer", "")).strip()))
        if c["feasible"] and len(c["distractors"]) == 3:
            c["options"], c["answer"] = assemble_options(cid, c["correct_answer"], c["distractors"])
        out.append(c)
    if not out:
        out.append(dict(base, cand_id="cand_" + sha1(seed.seed_id + BS_GEN_VERSION + llm.model + "0")[:12], feasible=False, empty=True))
    return out


# ------------------------------------------------------------------ deterministic checks
_SMALL = re.compile(r"^\d$")


def number_tokens(text: str) -> list[str]:
    t = text.replace("−", "-")
    return [m.group(0).replace(",", ".") for m in re.finditer(r"(?<![\w])\d+(?:[.,]\d+)?", t)]


def nontrivial_numbers(text: str) -> list[str]:
    return [x for x in number_tokens(text) if not _SMALL.match(x)]


_QUOTED = re.compile(r"[‘’“”'\"«]([^‘’“”'\"»]{3,80})[’”'\"»]")
_GENERIC_NAMES = {"việt nam", "trái đất", "ngữ văn", "lịch sử", "địa lí", "vật lí", "hóa học", "hoá học", "sinh học", "thế giới", "chiến tranh thế giới",
                  "bắc bộ", "nam bộ", "trung bộ", "đông nam á", "liên xô", "hoa kì", "hợp chủng quốc hoa kì", "tây nguyên", "biển đông"}


def _capital_sequences(q: str) -> list[str]:
    """Runs of >= 2 consecutive capitalised (not ALL-CAPS) words that do not start a sentence."""
    toks = q.split()
    out, cur = [], []
    for i, t in enumerate(toks):
        w = t.strip("(),;:?!.“”'\"")
        starts_sentence = i == 0 or toks[i - 1][-1:] in ".!?:"
        if w and w[0].isupper() and not (len(w) > 1 and w.isupper()) and not starts_sentence:
            cur.append(w)
        else:
            if len(cur) >= 2:
                out.append(" ".join(cur))
            cur = [w] if (w and w[0].isupper() and not (len(w) > 1 and w.isupper()) and starts_sentence and False) else []
    if len(cur) >= 2:
        out.append(" ".join(cur))
    return out


def names_grounded(question: str, texts: list[str]) -> bool:
    """Quoted titles and multi-word proper names in the question must occur in the gold spans / their headings / the book title
    (guards against hallucinated work, character or place names)."""
    hay = norm(" ".join(texts))
    q = re.sub(r"^\s*(theo|trong|ở|tại)\s+(sách giáo khoa|sgk)[^,]*,", "", question, flags=re.I)
    cands = [m.group(1) for m in _QUOTED.finditer(q)] + _capital_sequences(q)
    for n in cands:
        nn = norm(n)
        if not nn or nn in _GENERIC_NAMES or re.fullmatch(r"(sách giáo khoa|lớp \d+|bài \d+)", nn):
            continue
        if nn not in hay:
            return False
    return True


def bs_structural(c: dict, corpus: SpanCorpus) -> dict[str, Any]:
    """Extra deterministic checks on top of hard_gen.structural_checks (which the caller also applies)."""
    f: dict[str, Any] = {}
    gold = c.get("gold_span_ids") or []
    seed_ids = set(c.get("seed_span_ids") or [])
    f["gold_in_seed"] = bool(gold) and set(gold) <= seed_ids
    gtxt = {g: corpus.get(g).text for g in gold if corpus.get(g)}
    ev = c.get("evidence") or {}
    f["evidence_verbatim"] = bool(gold) and all(g in ev and quote_in_span(ev[g], gtxt.get(g, "")) for g in gold)
    # numbers of the key must be grounded in the gold spans unless the item is a deterministic calculation
    alltxt = " ".join(gtxt.values()).replace(",", ".")
    calc_item = bool(c.get("calc"))
    key_nums = nontrivial_numbers(c.get("correct_answer", ""))
    f["key_numbers_grounded"] = calc_item or all(n in alltxt for n in key_nums)
    qn = nontrivial_numbers(re.sub(r"(lớp|bài|chương|hình|bảng|trang|tập)\s*\d+", "", c.get("question", ""), flags=re.I))
    f["question_numbers_grounded"] = calc_item or c.get("trick_type") == "statement_combo" or all(n in alltxt for n in qn)
    f["length_ok"] = length_ok(c.get("correct_answer", ""), c.get("distractors", [])) and not unique_longest(c.get("options") or {}, c.get("answer"))
    f["trick_type_valid"] = c.get("trick_type") in TRICK_GUIDE
    f["deictic_free"] = deictic_free(c.get("question", ""))
    hay = [corpus.get(g).text + " " + " ".join(corpus.get(g).heading_path) + " " + (corpus.get(g).book_title or "") for g in gold if corpus.get(g)]
    f["names_grounded"] = names_grounded(c.get("question", ""), hay) if c.get("trick_type") != "statement_combo" else True
    if c.get("trick_type") == "table_calc":
        f["calc_declared"] = bool(c.get("calc"))
    if c.get("trick_type") == "statement_combo":
        sts = c.get("statements") or []
        f["statements_grounded"] = len(sts) == 4 and all(s.get("span_id") in gold and quote_in_span(str(s.get("quote", "")), gtxt.get(s.get("span_id"), ""), 0.6)
                                                       for s in sts)
        f["statements_both_values"] = 1 <= sum(1 for s in sts if s.get("true")) <= 4
    soft = ["gold_in_seed", "deictic_free", "names_grounded", "evidence_verbatim", "key_numbers_grounded", "question_numbers_grounded", "trick_type_valid"]
    soft += [k for k in ("calc_declared", "statements_grounded", "statements_both_values") if k in f]
    f["bs_structural_ok"] = all(f[k] for k in soft)
    f["bs_structural_fail"] = [k for k in soft if not f[k]]
    return f


def multi_hop_min_spans(c: dict) -> int:
    return 1 if len(c.get("gold_span_ids") or []) == 1 else 2


def needs_numeric_check(c: dict, corpus: SpanCorpus) -> bool:
    """OCR risk flag for the teacher pack: numbers in key/question/options, or a table among the gold spans."""
    txt = " ".join([c.get("question", ""), c.get("correct_answer", "")] + list((c.get("options") or {}).values()))
    if c.get("trick_type") in ("table_lookup", "table_calc") or len(nontrivial_numbers(txt)) > 0:
        return True
    return any(corpus.get(g) is not None and is_table(corpus.get(g)) for g in c.get("gold_span_ids") or [])


# ------------------------------------------------------------------ extra LLM checks
STATEMENT_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in ("I", "II", "III", "IV")}, "required": ["I", "II", "III", "IV"]}

_STATEMENT_PROMPT = """Chỉ dựa vào các ĐOẠN TRÍCH sách giáo khoa dưới đây (không dùng kiến thức bên ngoài), đánh giá từng phát biểu:
- "đúng": đoạn trích khẳng định phát biểu đúng.
- "sai": đoạn trích mâu thuẫn trực tiếp với phát biểu.
- "không xác định": đoạn trích không đủ để kết luận.

ĐOẠN TRÍCH:
{blocks}

{statements}

Trả về JSON: {{"I": "đúng|sai|không xác định", "II": "...", "III": "...", "IV": "..."}}"""


def check_statements(c: dict, corpus: SpanCorpus, llm) -> dict[str, Any]:
    gold = sorted(c["gold_span_ids"], key=lambda s: sha1(c["cand_id"] + s))
    labels = [f"E{i + 1}" for i in range(len(gold))]
    body = "\n".join(f"{_stmt_roman(i)}. {s['text']}" for i, s in enumerate(c["statements"]))
    p = _STATEMENT_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), statements=body)
    o, _ = chat_json(llm, [{"role": "user", "content": p}], STATEMENT_SCHEMA, max_tokens=120,
                     validate=lambda o: all(k in o for k in ("I", "II", "III", "IV")))
    m = {"đúng": True, "sai": False}
    got = [m.get(str(o[k]).strip().lower()) for k in ("I", "II", "III", "IV")]
    want = [bool(s["true"]) for s in c["statements"]]
    return {"labels": got, "want": want, "match": got == want, "model": llm.model, "prompt_version": BS_STATEMENT_VERSION}


_INDEP_PROMPT = """Bạn là học sinh giỏi. Chỉ dựa vào các ĐOẠN TRÍCH sách giáo khoa dưới đây, chọn đáp án đúng nhất của câu hỏi trắc nghiệm.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
{options}

Trả về JSON: {{"answer": "A|B|C|D"}}"""


def check_independent(c: dict, corpus: SpanCorpus, llm, perm: int) -> dict[str, Any]:
    """Independent (different model) answer with the gold spans only, one option permutation."""
    gold = sorted(c["gold_span_ids"], key=lambda s: sha1(c["cand_id"] + s + "i"))
    labels = [f"E{i + 1}" for i in range(len(gold))]
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + f":indep:{perm}")
    p = _INDEP_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), question=c["question"],
                             options=fmt_options(opts))
    from otter.evaluation.hard_gen import _letter
    o, _ = chat_json(llm, [{"role": "user", "content": p}], {}, max_tokens=40, validate=lambda o: _letter(o.get("answer")) is not None)
    a = _letter(o.get("answer"))
    return {"answer": back.get(a) if a else None, "correct": back.get(a) == c["answer"], "model": llm.model, "prompt_version": BS_INDEP_VERSION}


# ------------------------------------------------------------------ negation: per-option truth check
OPT_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in LETTERS}, "required": list(LETTERS)}
BS_OPTTRUTH_VERSION = "otter_bs_optruth_v1"
_OPTTRUTH_PROMPT = """Chỉ dựa vào các ĐOẠN TRÍCH sách giáo khoa dưới đây (không dùng kiến thức bên ngoài), đánh giá TỪNG phát biểu A, B, C, D:
- "đúng": đoạn trích khẳng định phát biểu đúng.
- "sai": đoạn trích mâu thuẫn trực tiếp với phát biểu.
- "không xác định": đoạn trích không đủ để kết luận.

ĐOẠN TRÍCH:
{blocks}

Chủ đề: {question}
{options}

Trả về JSON: {{"A": "đúng|sai|không xác định", "B": "...", "C": "...", "D": "..."}}"""


def check_option_truth(c: dict, corpus: SpanCorpus, llm, perm: int) -> dict[str, Any]:
    """For negation ('KHÔNG đúng') items: the key must be the only statement the gold spans contradict, the three others must be supported."""
    gold = sorted(c["gold_span_ids"], key=lambda s: sha1(c["cand_id"] + s + "t"))
    labels = [f"E{i + 1}" for i in range(len(gold))]
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + f":optruth:{perm}")
    p = _OPTTRUTH_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), question=c["question"], options=fmt_options(opts))
    o, _ = chat_json(llm, [{"role": "user", "content": p}], OPT_SCHEMA, max_tokens=120, validate=lambda o: all(k in o for k in LETTERS))
    m = {"đúng": True, "sai": False}
    t = {back[k]: m.get(str(o[k]).strip().lower()) for k in LETTERS}          # truth of the statement, by original letter
    ok = t.get(c["answer"]) is False and all(t.get(k) is True for k in LETTERS if k != c["answer"])
    return {"truth": t, "ok": ok, "model": llm.model, "prompt_version": BS_OPTTRUTH_VERSION}


# ------------------------------------------------------------------ audit without the book: self-containedness + other defensible options
AUDIT_SCHEMA = {"type": "object", "properties": {"self_contained": {"type": "integer"}, "also_correct": {"type": "array", "items": {"type": "string"}},
                                                 "subjective": {"type": "boolean"}}, "required": ["self_contained", "also_correct", "subjective"]}
BS_AUDIT_VERSION = "otter_bs_audit_v1"
_AUDIT_PROMPT = """Bạn là chuyên gia khảo thí. Câu hỏi trắc nghiệm dưới đây hỏi theo một sách giáo khoa Việt Nam cụ thể ({subject}, lớp {grade}). Đáp án được cho là {answer}.

Câu hỏi: {question}
{options}

Đánh giá:
1. "self_contained" (1-5): câu hỏi tự đứng được, đủ thông tin; 1 = tham chiếu tới hình/đoạn/văn bản/nhân vật/ngữ cảnh KHÔNG được nêu trong câu hỏi (ví dụ "người thứ nhất", "văn bản này", "trong hình"); 5 = hoàn toàn rõ ràng.
2. "also_correct": danh sách chữ cái các phương án KHÁC đáp án đã cho mà vẫn có thể được xem là đúng theo kiến thức khoa học/lịch sử/văn học thực tế hoặc theo cách trình bày phổ biến (kể cả khi sách này không nhắc tới). Nếu không có thì [].
3. "subjective": true nếu việc chọn đáp án phụ thuộc cảm nhận/quan điểm cá nhân.
Trả về JSON: {{"self_contained": 1-5, "also_correct": [], "subjective": false}}"""


def check_audit(c: dict, llm) -> dict[str, Any]:
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + ":audit")
    p = _AUDIT_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI.get(c["domain"], c["domain"]), grade=c.get("grade") or "?", answer=ans, question=c["question"],
                             options=fmt_options(opts))
    o, _ = chat_json(llm, [{"role": "user", "content": p}], AUDIT_SCHEMA, max_tokens=120,
                     validate=lambda o: isinstance(o.get("also_correct"), list) and str(o.get("self_contained", "")).strip().isdigit() or isinstance(o.get("self_contained"), int))
    also = [back[x] for x in (str(y).strip().upper()[:1] for y in o.get("also_correct") or []) if x in back and x != ans]
    sc = int(float(o.get("self_contained")))
    return {"self_contained": sc, "also_correct": also, "subjective": _bool(o.get("subjective")),
            "ok": sc >= 4 and not also and not _bool(o.get("subjective")), "model": llm.model, "prompt_version": BS_AUDIT_VERSION}


_DEICTIC = re.compile(r"\b(văn bản|bài viết|bài học|bài thơ|đoạn|phần|tác phẩm|thí nghiệm|hình|bảng|sơ đồ|biểu đồ|ví dụ|nhân vật|trường hợp|câu chuyện|bài) (này|trên|đó|kia)\b|"
                      r"\b(ở đây|như hình|như bảng|trong hình|trong bảng|hình bên|bảng bên)\b|người thứ (nhất|hai)\b", re.I)


def deictic_free(q: str) -> bool:
    return not _DEICTIC.search(q)


# ------------------------------------------------------------------ strict final review (32B, thinking mode)
BS_STRICT_VERSION = "otter_bs_strict_review_v1"
STRICT_SCHEMA = {"type": "object", "properties": {k: {"type": "boolean"} for k in
                                                  ("key_supported", "unique_key", "question_clear", "facts_accurate", "options_distinct")} | {"issues": {"type": "string"}},
                 "required": ["key_supported", "unique_key", "question_clear", "facts_accurate", "options_distinct", "issues"]}
_STRICT_PROMPT = """Bạn là giám khảo khó tính kiểm định câu hỏi trắc nghiệm do máy tạo ra từ sách giáo khoa. Hãy suy nghĩ kĩ rồi kiểm tra từng tiêu chí.

ĐOẠN TRÍCH SÁCH GIÁO KHOA (căn cứ):
{blocks}

Câu hỏi: {question}
{options}
Đáp án được cho là: {answer}

Tiêu chí (true = đạt):
- "key_supported": đoạn trích nói rõ đáp án được cho là đúng (không phải suy diễn lỏng lẻo hay ghép ý sai).
- "unique_key": CHỈ MỘT phương án đúng. Trả false nếu có phương án khác cũng đúng theo đoạn trích HOẶC theo kiến thức khoa học/lịch sử/văn học thực tế (kể cả khi sách không nhắc), hoặc nếu hai phương án là cách nói khác của cùng một ý.
- "question_clear": câu hỏi rõ ràng, tự đứng được, mọi tên tác phẩm/tác giả/nhân vật/địa danh nêu trong câu hỏi phải xuất hiện trong đoạn trích hoặc tiêu đề của nó (không bịa, không nhầm), không mơ hồ về phạm vi, không phụ thuộc hình hay ngữ cảnh không được nêu.
- "facts_accurate": các phát biểu trong câu hỏi và đáp án đúng không sai về khoa học/lịch sử (không phi lí, không tự mâu thuẫn).
- "options_distinct": bốn phương án khác nhau về nghĩa và cùng loại.
Trả về JSON: {{"key_supported": true, "unique_key": true, "question_clear": true, "facts_accurate": true, "options_distinct": true, "issues": "<mô tả ngắn vấn đề nếu có>"}}"""


def check_strict_review(c: dict, corpus: SpanCorpus, llm_thinking) -> dict[str, Any]:
    gold = list(c["gold_span_ids"])
    labels = [f"E{i + 1}" for i in range(len(gold))]
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + ":strict")
    p = _STRICT_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), question=c["question"],
                              options=fmt_options(opts), answer=f"{ans}. {opts[ans]}")
    keys = ("key_supported", "unique_key", "question_clear", "facts_accurate", "options_distinct")
    o, _ = chat_json(llm_thinking, [{"role": "user", "content": p}], STRICT_SCHEMA, max_tokens=4000, temperature=0.0,
                     validate=lambda o: all(k in o for k in keys))
    res = {k: _bool(o[k]) for k in keys}
    res.update(issues=str(o.get("issues", ""))[:400], ok=all(res.values()), model=llm_thinking.model, thinking=True, prompt_version=BS_STRICT_VERSION)
    return res
