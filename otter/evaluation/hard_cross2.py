"""ViEduHop-Cross v2 (internal: OTTER-Hard cross): seed building for cross-subject items, improved generator prompt.

Diagnosis of the v1 cross funnel (1,120 seeds -> 47 verified; see docs in the run report):
* seeds were bge-m3 nearest neighbours with only lexical overlap ("trung bình năm", "cốc thủy tinh", "carbon"), i.e. the two spans
  shared a word but not an idea; the generator rarely declined (28/1,120 infeasible) and bridged them with hallucinated links;
* 292/980 candidates failed the gold-span verification (the key is not entailed by the spans) and 447/561 of the verified,
  coherent ones were answerable from ONE span (the second subject only decorated the stem);
* 45/114 survivors failed the distractor-length rewrite.

Fixes implemented here: (1) pair seeds come from three sources (shared ontology concepts, bge-m3 neighbours, LLM topic bridges from
lesson titles) and every pair is screened by an LLM judge that must name a bridging entity/process/quantity plus the specific
fact each span contributes; (2) the generator prompt receives that bridge, must state the integrating idea, must use a different
subject fact in each step, must put the first fact in the stem and make the key depend on the second, and must name the sentence
that refutes every distractor; (3) improved distractor-length rewrite.

Nothing here relaxes the correctness filters of the shared funnel (hard_gen / hard_build): key entailed by the gold spans and unique,
not answerable from a single gold span, no exercise leakage, +-20% option length.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import (HOP_DEF_VI, HOP_PATTERN_VI, _META_REF, _GEN_SYS, _bool, _label, assemble_options,
                                       chat_json, fmt_options, span_block)
from otter.evaluation.hard_seeds import HOP_TYPES, Seed
from otter.evaluation.hard_v2 import LEN_HI, LEN_LO, length_ok, unique_longest  # noqa: F401
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI
from otter.evaluation.text import sha1, short_hash, tokens

SEEDS_CROSS2_VERSION = "otter_hard_seeds_cross_v2"
GEN_CROSS2_VERSION = "otter_hard_gen_cross_v4"
JUDGE_VERSION = "otter_cross_pair_judge_v1"
BRIDGE_VERSION = "otter_cross_topic_bridge_v1"
REWRITE2_VERSION = "otter_hard_distractor_rewrite_v2"

# unordered subject pairs (internal domain names); order = (first span's subject, second span's subject)
PAIRS = [("physics", "chemistry"), ("chemistry", "biology"), ("physics", "biology"), ("history", "geography"),
         ("geography", "biology"), ("geography", "chemistry"), ("geography", "physics"), ("literature", "history"),
         ("geography", "literature")]

_BAD_HEAD = re.compile(r"(thực hành|luyện tập|vận dụng|bài tập|ôn tập|khởi động|hoạt động|dự án|chuẩn bị|dụng cụ)", re.I)
_LAB = re.compile(r"(DỤNG CỤ|CHUẨN BỊ|TIẾN HÀNH|BÁO CÁO|MẪU BÁO CÁO|THIẾT BỊ THÍ NGHIỆM)")
_TOC = re.compile(r"(\.{5,}|…{3,})")
_IMP = re.compile(r"^\W*(\d+[.)]\s*)?(hãy|em hãy|sưu tầm|quan sát|tìm hiểu|thảo luận|nêu|trình bày|cho biết|vẽ|đọc|thực hiện|"
                  r"chuẩn bị|lập|so sánh|giải thích|phân tích|viết|tính|xác định|dựa vào)\b", re.I)
_META_HEAD = re.compile(r"(mở đầu|làm quen|lời nói đầu|mục lục|hướng dẫn sử dụng|giải thích thuật ngữ|bảng tra cứu|ngành nghề|"
                        r"học tập môn|phương pháp học|an toàn trong phòng|thực hành|dự án|ôn tập|tổng kết|em có biết)", re.I)
_FIG_REF = re.compile(r"(quan sát|từ|ở|trong|theo|dựa vào|xem|nhìn)\s+(đồ thị|hình|bảng|sơ đồ|bản đồ)\s*(vẽ\s*)?\d", re.I)
_ITEM_START = re.compile(r"^\s*(\d+[.)]|[a-dA-D]\))\s")
_IMP_SENT = re.compile(r"(^|[.:;?!]\s+|\n\s*[-•]?\s*)(Xác định|Tính|Viết|Hãy|Cho biết|Giải thích|Nêu|Trình bày|Chứng minh|Tìm|Vẽ|Lập|"
                       r"Thực hiện|Quan sát|Đọc|Em hãy|Lắc|Gõ|Nhỏ|Đổ|Rót|Lấy|Đun|Hơ|Cho vào|Thêm|Đặt)\b")
_PROC = re.compile(r"^\s*(\*\*)?(Tiến hành|Chuẩn bị|Dụng cụ|Hóa chất|Hoá chất|Bước)", re.I)
_OBJ = re.compile(r"(học xong bài này|sau bài học này|yêu cầu cần đạt|mục tiêu|em sẽ|em có thể)", re.I)


def cross_eligible(s: Span, *, min_chars: int = 90, max_chars: int = 2200) -> bool:
    """Expository text only: no exercise/activity/question/objective/figure/lab-list spans, no TOC, no OCR garbage."""
    t = s.text.strip()
    if not s.domain or not (min_chars <= len(t) <= max_chars) or s.is_exercise():
        return False
    if not (s.section_id or s.heading_path):
        return False
    if s.pedagogical_type in ("objective", "figure_caption", "question", "exercise", "activity", "table"):
        return False
    if t.startswith("[") or t.startswith("![") or t.startswith("#") or _TOC.search(t) or _OBJ.search(t[:160]):
        return False
    if _IMP.match(t) or re.match(r"^\s*[-–]?\s*Bước \d", t):
        return False
    hp = " / ".join(s.heading_path[-2:])
    if _LAB.search(hp) or _BAD_HEAD.search(" ".join(s.heading_path[-1:])):
        return False
    if sum(ch.isalpha() for ch in t) / max(1, len(t)) < 0.5:
        return False
    full_head = " / ".join(s.heading_path)
    if _META_HEAD.search(full_head) or _FIG_REF.search(t) or _ITEM_START.match(t) or _IMP_SENT.search(t) or _PROC.match(t):
        return False
    if len(re.findall(r"(?m)^\s*[-•]\s", t)) >= 3 and sum(len(x) for x in t.split("\n")) / max(1, len(t.split("\n"))) < 70:
        return False                                  # equipment / procedure lists
    toks = tokens(t)
    return len(toks) >= 20 and len(set(toks)) / len(toks) > 0.3


def lesson_title(s: Span) -> str:
    for h in s.heading_path:
        if re.match(r"^\s*(BÀI|Bài|Chủ đề|CHỦ ĐỀ|Chuyên đề|CHUYÊN ĐỀ)\b", h) and not re.match(r"^\s*(CHƯƠNG|Chương|PHẦN)", h):
            return re.sub(r"\s+", " ", h).strip()
    return re.sub(r"\s+", " ", s.heading_path[1] if len(s.heading_path) > 1 else (s.heading_path[0] if s.heading_path else "")).strip()


def load_span_vectors(db: Path) -> dict[str, np.ndarray]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        raw = dict(c.execute("SELECT item_id, vec FROM embeddings WHERE kind='span' AND model='BAAI/bge-m3'"))
    out = {}
    for k, b in raw.items():
        v = np.frombuffer(b, dtype=np.float32)
        out[k] = v / (np.linalg.norm(v) or 1.0)
    return out


# ------------------------------------------------------------------ pair judge
_JUDGE_SYS = "Bạn là giáo viên giàu kinh nghiệm ra đề đánh giá năng lực tích hợp liên môn. Chỉ trả về JSON hợp lệ."
_JUDGE_PROMPT = """Hai đoạn trích từ sách giáo khoa của HAI MÔN KHÁC NHAU.

[S1] (môn {s1}, lớp {g1}; {h1})
{t1}

[S2] (môn {s2}, lớp {g2}; {h2})
{t2}

Đánh giá xem có thể ra MỘT câu hỏi tích hợp thật sự cần CẢ HAI đoạn không. Câu hỏi tích hợp tốt có dạng: tình huống/hiện tượng thực tế → \
dùng một kiến thức CỤ THỂ của S1 để xác định một đối tượng/quá trình/đại lượng trung gian → dùng một kiến thức CỤ THỂ KHÁC của S2 về đối \
tượng đó để ra đáp án. Chỉ trùng từ khóa hoặc cùng chủ đề chung chung (ví dụ cùng nói về "nước", "năng lượng", "phương pháp") KHÔNG phải \
là liên hệ thật sự.

Thang điểm "score": 0 = không liên quan; 1 = chỉ trùng từ; 2 = cùng chủ đề chung, không có quan hệ kiến thức cụ thể; 3 = có quan hệ kiến \
thức nhưng một đoạn đã đủ trả lời hoặc đoạn kia chỉ minh họa; 4 = hai kiến thức cụ thể bổ sung nhau, ra được câu hỏi cần cả hai; 5 = rất \
tự nhiên, hai kiến thức nối nhau thành chuỗi suy luận (nguyên nhân–kết quả, nguyên lí–ứng dụng, đại lượng–điều kiện...).

Trả về JSON: {{"score": 0-5, "bridge": "<thực thể/quá trình/đại lượng nối hai đoạn, rỗng nếu không có>", \
"fact_s1": "<kiến thức cụ thể của S1>", "fact_s2": "<kiến thức cụ thể khác của S2>", "idea": "<ý tưởng tích hợp, một câu>"}}"""


def judge_pair(a: Span, b: Span, llm, *, max_chars: int = 900) -> dict[str, Any]:
    p = _JUDGE_PROMPT.format(s1=DOMAIN_TO_SUBJECT_VI.get(a.domain, a.domain), g1=a.grade, h1=" › ".join(a.heading_path[-2:]),
                             t1=a.text.strip()[:max_chars], s2=DOMAIN_TO_SUBJECT_VI.get(b.domain, b.domain), g2=b.grade,
                             h2=" › ".join(b.heading_path[-2:]), t2=b.text.strip()[:max_chars])

    def num(o):
        try:
            return int(float(o.get("score")))
        except (TypeError, ValueError):
            return None
    o, _ = chat_json(llm, [{"role": "system", "content": _JUDGE_SYS}, {"role": "user", "content": p}], {}, max_tokens=420,
                     validate=lambda o: num(o) is not None)
    return {"score": max(0, min(5, num(o))), "bridge": str(o.get("bridge") or "")[:200], "fact_s1": str(o.get("fact_s1") or "")[:300],
            "fact_s2": str(o.get("fact_s2") or "")[:300], "idea": str(o.get("idea") or "")[:300], "model": llm.model,
            "prompt_version": JUDGE_VERSION}


# ------------------------------------------------------------------ topic bridges from lesson titles
_BRIDGE_SYS = "Bạn là giáo viên tích hợp liên môn của chương trình GDPT 2018. Chỉ trả về JSON hợp lệ."
_BRIDGE_PROMPT = """Bài học môn {sx}: «{tx}» (lớp {gx}). Các mục chính: {secx}.

Danh sách bài học môn {sy} (số thứ tự – lớp – tên bài):
{ylist}

Chọn tối đa 3 bài môn {sy} mà cùng với bài môn {sx} trên có thể ra câu hỏi TÍCH HỢP thật sự (một hiện tượng/vấn đề thực tế cần kiến thức \
cụ thể của cả hai bài; không chỉ trùng từ hay cùng chủ đề chung chung). Nếu không có bài nào thật sự phù hợp thì trả về danh sách rỗng.
Ví dụ kiểu liên hệ: quang hợp ↔ năng lượng; khí hậu ↔ nông nghiệp; phản ứng hóa học ↔ tốc độ/nhiệt; thủy quyển ↔ tính chất của nước.

Trả về JSON: {{"bridges": [{{"y": <số thứ tự>, "idea": "<ý tưởng tích hợp, một câu>", "x_query": "<cụm từ khóa 5-12 từ mô tả kiến thức \
cụ thể cần lấy ở bài {sx}>", "y_query": "<cụm từ khóa 5-12 từ mô tả kiến thức cụ thể cần lấy ở bài {sy}>"}}]}}"""


def topic_bridges(x_card: dict, y_cards: list[dict], llm) -> list[dict]:
    ylist = "\n".join(f"{i} – lớp {c['grade']} – {c['title']}" for i, c in enumerate(y_cards))
    p = _BRIDGE_PROMPT.format(sx=DOMAIN_TO_SUBJECT_VI.get(x_card["domain"]), tx=x_card["title"], gx=x_card["grade"],
                              secx="; ".join(x_card["sections"][:10]) or "—", sy=DOMAIN_TO_SUBJECT_VI.get(y_cards[0]["domain"]), ylist=ylist)
    o, _ = chat_json(llm, [{"role": "system", "content": _BRIDGE_SYS}, {"role": "user", "content": p}], {}, max_tokens=500,
                     validate=lambda o: isinstance(o.get("bridges"), list))
    out = []
    for b in o["bridges"][:3]:
        try:
            y = int(b.get("y"))
        except (TypeError, ValueError):
            continue
        if 0 <= y < len(y_cards):
            out.append({"x_lesson": x_card["key"], "y_lesson": y_cards[y]["key"], "idea": str(b.get("idea") or ""),
                        "x_query": str(b.get("x_query") or ""), "y_query": str(b.get("y_query") or "")})
    return out


# ------------------------------------------------------------------ generation prompt (v4)
CROSS_HOPS = ("bridge", "causal_chain", "conditional_rule", "comparison")

_GEN4 = """Môn: {s1} (S1) và {s2} (S2). Lớp: {grades}.
Dưới đây là hai đoạn trích từ sách giáo khoa của HAI MÔN KHÁC NHAU (mã [S1], [S2]).

{blocks}

{hint}NHIỆM VỤ: viết MỘT câu hỏi trắc nghiệm 4 phương án theo phong cách bài đánh giá năng lực tích hợp liên môn (tình huống/hiện tượng thực tế \
cần kiến thức của cả hai môn), kiểu suy luận ưu tiên: {hop_def}
Mẫu trừu tượng: {hop_pattern}
(Nếu kiểu này không hợp, được chọn kiểu khác trong: {compat}.)

CÁCH LÀM (điền vào JSON theo đúng thứ tự):
1. "feasible": chỉ true nếu có một ý tưởng tích hợp THẬT SỰ — một kiến thức cụ thể của S1 và một kiến thức cụ thể KHÁC của S2 phải được dùng nối \
tiếp nhau. Nếu hai đoạn chỉ trùng từ khóa hoặc cùng chủ đề chung chung thì đặt false (các trường khác để rỗng). Đừng bịa mối liên hệ.
2. "integrating_idea": ý tưởng tích hợp (1–2 câu) nêu rõ thực thể/quá trình/đại lượng nối hai kiến thức.
3. "fact_s1" và "fact_s2": mỗi bên một kiến thức cụ thể, chép gần nguyên văn từ đoạn tương ứng (không được là kiến thức nằm ở đoạn kia).
4. "reasoning_steps": 2–4 bước; MỖI bước ghi đoạn dùng ("S1" hoặc "S2") và phải dùng kiến thức khác nhau; ít nhất một bước dùng S1 và một bước dùng S2; \
bước dùng S2 phải dựa vào kết quả của bước dùng S1 (hoặc ngược lại).
5. "stem_uses" và "key_needs": câu hỏi nêu tình huống và dữ kiện lấy từ MỘT đoạn, KHÔNG nêu tên thực thể trung gian và KHÔNG nêu điều đoạn kia khẳng định; \
đáp án đúng phải là điều CHỈ suy ra được nhờ đoạn kia. Ghi "stem_uses" = đoạn cung cấp dữ kiện của câu hỏi, "key_needs" = đoạn cung cấp kiến thức quyết định đáp án; \
hai giá trị phải khác nhau. "missing_if_no_s1" / "missing_if_no_s2": nếu bỏ đoạn đó thì thiếu thông tin gì (một câu).
6. Viết câu hỏi và 4 phương án.

YÊU CẦU BẮT BUỘC:
a. Câu hỏi tự đứng được như câu hỏi kiểm tra cho học sinh: tuyệt đối không nhắc "đoạn trích", "S1", "theo thông tin trên", tên môn học hay tên sách; không chép \
nguyên câu của đoạn trích. Không mô phỏng hay sao chép đề thi có sẵn.
b. Đáp án đúng duy nhất, suy ra chắc chắn từ hai đoạn trích (chỉ dùng kiến thức trong hai đoạn, cộng phép tính số học nếu cần). Không dùng kiến thức ngoài sách.
c. Ba phương án nhiễu: cùng loại, cùng cấu trúc câu và CÙNG ĐỘ DÀI (±15% số ký tự) với đáp án đúng, nghe hợp lí với người chưa đọc kĩ sách, nhưng mỗi cái bị \
BÁC BỎ rõ ràng bởi một câu/ý cụ thể trong S1 hoặc S2 — ghi câu/ý đó (≤ 25 từ, chép từ đoạn trích) vào "refuted_by"; không được chỉ "sai theo hiểu biết chung". \
Nhiễu nên là kết quả khi chỉ dùng một đoạn, bỏ qua điều kiện, đảo nguyên nhân–kết quả hoặc nhầm đối tượng giữa hai môn; ít nhất một nhiễu là điều ĐÚNG theo S1 \
nhưng không trả lời câu hỏi và ít nhất một nhiễu là điều ĐÚNG theo S2 nhưng không trả lời câu hỏi (để người chỉ đọc một đoạn không loại được chúng).
d. Không dùng "tất cả đều đúng/sai", "cả A và B"; không câu hỏi phủ định ("không phải", "sai"). Đáp án đúng không xuất hiện nguyên văn trong câu hỏi.
e. Tiếng Việt chuẩn, đúng thuật ngữ của sách.

Trả về JSON: {{"feasible": bool, "hop_type": "...", "integrating_idea": "...", "fact_s1": "...", "fact_s2": "...", "reasoning_steps": [{{"span": "S1|S2", \
"step": "..."}}], "stem_uses": "S1|S2", "key_needs": "S1|S2", "missing_if_no_s1": "...", "missing_if_no_s2": "...", "question": "...", \
"correct_answer": "...", "distractors": [{{"text": "...", "why_wrong": "...", "refuted_by": "..."}}], "gold_spans": ["S1", "S2"]}}"""


def _hint(seed: Seed) -> str:
    j = (seed.link or {}).get("onto", {}).get("judge") or {}
    if not j:
        return ""
    lines = []
    if j.get("bridge"):
        lines.append(f"- thực thể/quá trình nối: {j['bridge']}")
    if j.get("fact_s1"):
        lines.append(f"- kiến thức gợi ý từ S1: {j['fact_s1']}")
    if j.get("fact_s2"):
        lines.append(f"- kiến thức gợi ý từ S2: {j['fact_s2']}")
    if j.get("idea"):
        lines.append(f"- ý tưởng tích hợp gợi ý: {j['idea']}")
    return "GỢI Ý (do người thẩm định đề xuất, chỉ để định hướng; đáp án vẫn phải suy ra được từ hai đoạn):\n" + "\n".join(lines) + "\n\n" if lines else ""


def gen_cross_schema(hop_types: list[str]) -> dict:
    return {}


_REPAIR = """LẦN VIẾT TRƯỚC BỊ LOẠI: câu hỏi «{q}» (đáp án: {a}) có thể trả lời chỉ với {which} — đoạn còn lại không thật sự cần thiết{missing}.
Hãy viết một câu hỏi MỚI từ cùng hai đoạn, trong đó: (1) dữ kiện của câu hỏi chỉ dẫn tới đối tượng/điều kiện cần tìm khi kết hợp với đoạn kia (ví dụ tình huống \
có số liệu hoặc điều kiện "nếu … thì …" mà muốn biết điều kiện có thỏa mãn phải dùng một đoạn, rồi dùng đoạn còn lại để suy ra kết quả); (2) đáp án đúng KHÔNG thể \
chọn được khi chỉ đọc một đoạn; (3) không lặp lại ý của câu hỏi cũ. Nếu không thể làm được thì đặt "feasible": false.

"""


def generate_cross(seed: Seed, corpus: SpanCorpus, llm, *, temperature: float = 0.6, max_tokens: int = 1800,
                   prompt_version: str = GEN_CROSS2_VERSION, repair: Optional[dict] = None) -> dict[str, Any]:
    spans = [corpus.get(i) for i in seed.span_ids]
    labels = ["S1", "S2"]
    hop = seed.target_hop_type if seed.target_hop_type in CROSS_HOPS else "bridge"
    compat = [h for h in dict.fromkeys([hop] + list(seed.compatible_hop_types)) if h in CROSS_HOPS and (h != "comparison" or hop == "comparison")]
    grades = sorted({s.grade for s in spans if s.grade is not None})
    prompt = _GEN4.format(s1=DOMAIN_TO_SUBJECT_VI.get(spans[0].domain, spans[0].domain), s2=DOMAIN_TO_SUBJECT_VI.get(spans[1].domain, spans[1].domain),
                          grades=", ".join(map(str, grades)) or "?", blocks="\n\n".join(span_block(l, s) for l, s in zip(labels, spans)),
                          hint=_hint(seed), hop_def=HOP_DEF_VI[hop], hop_pattern=HOP_PATTERN_VI[hop], compat=", ".join(compat))
    if repair:
        prompt = prompt.replace("NHIỆM VỤ:", _REPAIR.format(q=repair["question"], a=repair["answer_text"], which=repair["which"],
                                                             missing=("; thiếu: " + repair["missing"]) if repair.get("missing") else "") + "NHIỆM VỤ:", 1)
        prompt_version = prompt_version + ("_repair" if repair.get("round", 1) == 1 else "_repair2")
    cand: dict[str, Any] = {"cand_id": "cand_" + sha1(seed.seed_id + prompt_version + llm.model)[:12], "seed_id": seed.seed_id,
                            "domain": seed.domain, "grade": seed.grade, "seed_span_ids": seed.span_ids, "strategy": seed.strategy,
                            "target_hop_type": hop, "generator": {"model": llm.model, "prompt_version": prompt_version, "temperature": temperature}}

    def ok(o):
        if not _bool(o.get("feasible")):
            return True
        return bool(str(o.get("question", "")).strip()) and isinstance(o.get("distractors"), list)
    try:
        obj, r = chat_json(llm, [{"role": "system", "content": _GEN_SYS}, {"role": "user", "content": prompt}], {},
                           temperature=temperature, max_tokens=max_tokens, validate=ok)
        cand["generator"].update(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens)
    except Exception as e:  # noqa: BLE001
        cand.update(feasible=False, error=f"{type(e).__name__}: {e}")
        return cand
    lab2id = dict(zip(labels, seed.span_ids))
    feasible = _bool(obj.get("feasible")) and bool(str(obj.get("question", "")).strip())
    dobj = [d if isinstance(d, dict) else {"text": str(d), "why_wrong": ""} for d in obj.get("distractors") or []]
    dis = [str(d.get("text", "")).strip() for d in dobj]
    steps = [{"span_id": lab2id.get(_label(st.get("span"), labels)), "step": str(st.get("step", ""))}
             for st in obj.get("reasoning_steps") or [] if isinstance(st, dict)]
    gold = [l for l in (_label(x, labels) for x in obj.get("gold_spans") or []) if l]
    cand.update(feasible=feasible, hop_type=obj.get("hop_type") if obj.get("hop_type") in HOP_TYPES else hop,
                link=str(obj.get("integrating_idea", "")), integrating_idea=str(obj.get("integrating_idea", "")),
                cross_meta={"fact_s1": obj.get("fact_s1"), "fact_s2": obj.get("fact_s2"), "stem_uses": obj.get("stem_uses"),
                            "key_needs": obj.get("key_needs"), "missing_if_no_s1": obj.get("missing_if_no_s1"),
                            "missing_if_no_s2": obj.get("missing_if_no_s2"), "judge": (seed.link or {}).get("onto", {}).get("judge")},
                facts=[], calc=None, question=str(obj.get("question", "")).strip(), correct_answer=str(obj.get("correct_answer", "")).strip(),
                distractors=dis, distractor_notes=[str(d.get("why_wrong", "")) for d in dobj],
                distractor_refuted_by=[str(d.get("refuted_by", "")) for d in dobj],
                gold_span_ids=list(dict.fromkeys(lab2id[x] for x in gold)), reasoning_steps=steps)
    if repair:
        cand["repair_of"] = repair["parent"]
        cand["repair_round"] = repair.get("round", 1)
    if feasible and len(dis) == 3 and cand["correct_answer"]:
        cand["options"], cand["answer"] = assemble_options(cand["cand_id"], cand["correct_answer"], dis)
    return cand


def cross_pre_ok(c: dict[str, Any]) -> bool:
    """Cheap deterministic gate before any LLM check: both subjects used, stem/key need different spans."""
    m = c.get("cross_meta") or {}
    if m.get("stem_uses") and m.get("key_needs") and str(m["stem_uses"]).strip() == str(m["key_needs"]).strip():
        return False
    return True


# ------------------------------------------------------------------ distractor-length rewrite v2
_REWRITE2 = """Dưới đây là các ĐOẠN TRÍCH sách giáo khoa và một câu hỏi trắc nghiệm có đáp án đúng đã được kiểm chứng.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
Đáp án đúng ({n} ký tự): {correct}
Phương án nhiễu hiện tại:
{current}

NHIỆM VỤ: viết lại đúng 3 phương án nhiễu mới sao cho:
1. CÙNG KIỂU, CÙNG CẤU TRÚC câu và mức độ chi tiết với đáp án đúng.
2. ĐỘ DÀI: mỗi phương án nhiễu dài từ {lo} đến {hi} ký tự (đáp án đúng dài {n} ký tự); nhiễu 1 khoảng {t1}, nhiễu 2 khoảng {t2}, nhiễu 3 khoảng {t3} ký tự. \
Đừng rút gọn bằng cách bỏ chi tiết, cũng đừng nhồi từ vô nghĩa; thêm/bớt vế phụ có nghĩa để chỉnh độ dài.
3. Hợp lí với người chưa đọc kĩ sách nhưng CHẮC CHẮN SAI: mỗi nhiễu phải bị một câu/ý cụ thể trong các đoạn trích bác bỏ (ghi vào "refuted_by", ≤ 25 từ). Khác nhau rõ rệt, \
không chứa ý của đáp án đúng.
4. Không dùng "tất cả đều đúng/sai", "cả A và B"; không nhắc "đoạn trích".
{extra}
Trả về JSON: {{"distractors": [{{"text": "...", "why_wrong": "...", "refuted_by": "..."}}, {{"text": "...", "why_wrong": "...", "refuted_by": "..."}}, \
{{"text": "...", "why_wrong": "...", "refuted_by": "..."}}]}}"""


def rewrite_distractors2(c: dict[str, Any], corpus: SpanCorpus, llm, *, attempts: int = 5) -> Optional[dict[str, Any]]:
    import random
    rng = random.Random(sha1(c["cand_id"] + "len2")[:8])
    n = len(c["correct_answer"])
    lo, hi = max(1, int(n * 0.85)), max(2, int(n * 1.15))
    blocks = "\n\n".join(span_block(f"E{i + 1}", corpus.get(s)) for i, s in enumerate(c["gold_span_ids"]))
    feedback = ""
    for k in range(attempts):
        ts = [max(4, int(round(n * rng.uniform(*r)))) for r in ((1.0, 1.12), (0.92, 1.08), (0.88, 1.05))]
        cur = "\n".join(f"- ({len(d)} ký tự) {d}" for d in c["distractors"])
        p = _REWRITE2.format(blocks=blocks, question=c["question"], correct=c["correct_answer"], current=cur, n=n, lo=lo, hi=hi,
                             t1=ts[0], t2=ts[1], t3=ts[2], extra=feedback)
        try:
            o, _ = chat_json(llm, [{"role": "user", "content": p}], {}, max_tokens=800, temperature=0.3 if k == 0 else 0.6,
                             validate=lambda o: isinstance(o.get("distractors"), list) and len(o["distractors"]) >= 3)
        except Exception:  # noqa: BLE001
            continue
        dd = [d if isinstance(d, dict) else {"text": str(d)} for d in o["distractors"][:3]]
        ds = [str(d.get("text") or "").strip() for d in dd]
        if not all(ds) or len({x.lower() for x in ds + [c["correct_answer"].lower()]}) < 4:
            feedback = "Các phương án phải khác nhau và khác đáp án đúng.\n"
            continue
        if _BAD_OPT(ds):
            feedback = "Không dùng 'tất cả đều đúng/sai' hay 'cả A và B'.\n"
            continue
        if length_ok(c["correct_answer"], ds) and (k >= attempts - 2 or max(len(d) for d in ds) >= n):
            new = dict(c)
            new["parent_cand_id"] = c["cand_id"]
            new["cand_id"] = c["cand_id"] + "_F"
            new["distractors"] = ds
            new["distractor_notes"] = [str(d.get("why_wrong", "")) for d in dd]
            new["distractor_refuted_by"] = [str(d.get("refuted_by", "")) for d in dd]
            new["options"], new["answer"] = assemble_options(new["cand_id"], c["correct_answer"], ds)
            new["distractor_rewrite"] = {"version": REWRITE2_VERSION, "model": llm.model, "attempts": k + 1, "old_distractors": c["distractors"]}
            return new
        feedback = ("Lần trước độ dài chưa đạt: " + ", ".join(f"{len(d)} ký tự" for d in ds) + f" (cần {lo}–{hi}). Hãy chỉnh lại đúng khoảng độ dài.\n")
    return None


_BAD_RX = re.compile(r"(tất cả (các )?(phương án|đáp án|ý)|cả [ABCD] và [ABCD]|cả ba|không có (phương án|đáp án))", re.I)


def _BAD_OPT(ds: list[str]) -> bool:
    return any(_BAD_RX.search(d or "") or _META_REF.search(d or "") for d in ds)
