"""ViEduHop-Deep: cross-subject questions that need 4-7 distinct textbook spans.

Seeds   -- an existing 2-span cross-subject bridge (work_cross pairs sharing an ontology concept / topic bridge / judged >= 3) is
           extended to a pool of k spans from a fixed subject template (3-4 subjects for >= 60% of seeds, else 2): every added span is
           linked to a member by an anchor-type edge (ontology concept / topic bridge / judged; bge-m3 sim >= 0.5), has mean similarity >= 0.45 and min >= 0.35 to all members, belongs to a
           different subject than the previously added span, and comes from a different lesson than every member. Exercise spans are
           never eligible (hard_cross2.cross_eligible); span pairs already used together by a benchmark item are excluded.
Gen     -- one structured call per seed (prompt `otter_hard_deep_gen_v1`): a single chain question; the generator may drop spans
           (gold_spans) but needs >= 4; hop_levels = number of gold spans.
Gates   -- structural, leakage, key entailed by ALL gold spans and unique (verify), quality, no single span sufficient, and
           leave-one-out necessity: for every gold span s the other spans must NOT suffice to derive the key (`check_loo`, two
           option permutations; one "sufficient and keyed" verdict rejects the item).
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Optional

import numpy as np

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import (SINGLE_SCHEMA, _bool, _letter, _label, chat_json, fmt_options, permute, span_block,
                                       assemble_options, structural_checks)
from otter.evaluation.hard_seeds import Seed
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.text import sha1, short_hash

DEEP_SEEDS_VERSION = "otter_hard_deep_seeds_v1"
DEEP_GEN_VERSION = "otter_hard_deep_gen_v2"
LOO_VERSION = "otter_hard_loo_v1"
DEEP_HOPS = ("bridge", "causal_chain", "conditional_rule", "calculation", "temporal", "comparison")


# ------------------------------------------------------------------ seeds
SUBJECT_SETS_3 = [("physics", "chemistry", "biology"), ("biology", "geography", "history"), ("geography", "history", "literature"),
                  ("chemistry", "geography", "biology"), ("physics", "geography", "chemistry"),
                  ("physics", "chemistry", "biology", "geography"), ("biology", "geography", "history", "literature"),
                  ("physics", "biology", "geography")]
SUBJECT_SETS_2 = [("physics", "chemistry"), ("chemistry", "biology"), ("physics", "biology"), ("history", "geography"),
                  ("geography", "biology"), ("literature", "history"), ("geography", "chemistry"), ("geography", "physics")]


def _norm_pair(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def cross_deep_seeds(corpus: SpanCorpus, elig: list[Span], dense: np.ndarray, anchors: list[dict], *, ks: dict[int, int],
                     frac3: float = 0.7, used_pairs: Optional[set] = None, span_concepts: Optional[dict[str, set[str]]] = None,
                     seed: int = 61, max_span_uses: int = 3, tau: float = 0.5, nn: int = 40, min_member_sim: float = 0.45,
                     min_link_sim: float = 0.5, min_any_sim: float = 0.35, concept_label: Optional[dict[str, str]] = None, preload: Optional[list[Seed]] = None,
                     path_only: bool = False, require_terms: bool = True) -> list[Seed]:
    """Extend 2-span cross-subject bridges to connected k-span pools drawn from a fixed subject set.

    anchors: [{"a","b","sources","sim"}] existing cross-subject pairs (concept / topic-bridge / judged); `used_pairs`: span pairs already
    used together by an existing benchmark item (never used as anchor). Growth: the next span is a dense neighbour (bge-m3) of a
    current member, belongs to the template's subjects, differs in subject from the previously added span, keeps the average
    similarity to all members above `min_member_sim`, and brings a subject not yet present while the template is not covered.
    `ks`: {pool size k: number of seeds}; share `frac3` of them use 3-4 subject templates, the rest 2-subject templates."""
    rng = random.Random(seed)
    idx = {s.span_id: i for i, s in enumerate(elig)}
    from otter.evaluation.hard_seeds import _STOP
    from otter.evaluation.text import tokens
    bg = []
    for s in elig:
        tk = tokens(s.text)
        bg.append({f"{a} {b}" for a, b in zip(tk, tk[1:]) if a not in _STOP and b not in _STOP and not a.isdigit() and not b.isdigit()
                   and len(a) + len(b) > 4})
    dfc: Counter = Counter(x for b in bg for x in b)
    terms = [{x for x in b if 2 <= dfc[x] <= max(6, int(0.03 * len(elig)))} for b in bg]      # shared content bigrams across the pool
    sims = dense @ dense.T                                  # `dense` rows are aligned with `elig`
    dom = [s.domain for s in elig]
    used_pairs = used_pairs or set()
    span_concepts = span_concepts or {}
    by_pair: dict[frozenset, list[dict]] = defaultdict(list)
    for r in anchors:
        if r["a"] in idx and r["b"] in idx and dom[idx[r["a"]]] != dom[idx[r["b"]]] and _norm_pair(r["a"], r["b"]) not in used_pairs:
            by_pair[frozenset((dom[idx[r["a"]]], dom[idx[r["b"]]]))].append(r)
    # growth edges = the anchor pairs themselves (ontology concept / topic bridge / judged), so every link of the pool is such an edge
    adj: dict[int, list[int]] = defaultdict(list)
    edge_labels: dict[frozenset, list[str]] = {}
    for rs in by_pair.values():
        for r in rs:
            i, j = idx[r["a"]], idx[r["b"]]
            if r.get("labels"):
                edge_labels[frozenset((i, j))] = r["labels"]
            adj[i].append(j)
            adj[j].append(i)
    nbr = {i: np.array(sorted(set(v), key=lambda j: -sims[i, j])[:nn], dtype=int) for i, v in adj.items()}
    for i in range(len(elig)):
        nbr.setdefault(i, np.array([], dtype=int))
    uses: Counter = Counter()
    taken: list[frozenset] = []
    out: list[Seed] = []
    for sd in preload or []:                     # earlier rounds: their spans count towards reuse, their span sets are never repeated
        uses.update(sd.span_ids)
        taken.append(frozenset(sd.span_ids))
    for k in sorted(ks, reverse=True):
        want3 = int(round(ks[k] * frac3))
        plan = [("3", want3), ("2", ks[k] - want3)]
        for kind, n_want in plan:
            sets = SUBJECT_SETS_3 if kind == "3" else SUBJECT_SETS_2
            made, fails = 0, 0
            while made < n_want and fails < 400 * max(1, n_want):
                T = rng.choice(sets)
                cap = max(2, -(-k // len(T)))
                pairs = [r for pr, rs in by_pair.items() if pr <= set(T) for r in rs]
                if not pairs:
                    fails += 1
                    continue
                r = rng.choice(pairs)
                if uses[r["a"]] >= max_span_uses or uses[r["b"]] >= max_span_uses:
                    fails += 1
                    continue
                mem = [idx[r["a"]], idx[r["b"]]]
                links = [{"from": r["a"], "to": r["b"], "sim": round(float(sims[mem[0], mem[1]]), 3), "source": "anchor:" + "+".join(r["sources"]), "labels": r.get("labels", [])}]
                while len(mem) < k:
                    have = Counter(dom[m] for m in mem)
                    missing = [d for d in T if have[d] == 0]
                    slots_left = k - len(mem)
                    cands = []
                    for m in (mem[-1:] if path_only else mem[-2:] + mem[:-2]):
                        for j in nbr[m]:
                            j = int(j)
                            if j in mem or any(_norm_pair(elig[j].span_id, elig[x].span_id) in used_pairs for x in mem) or dom[j] not in T or have[dom[j]] >= cap or dom[j] == dom[mem[-1]]:
                                continue
                            if uses[elig[j].span_id] >= max_span_uses or sims[m, j] < min_link_sim:
                                continue
                            if len(missing) >= slots_left and dom[j] not in missing:
                                continue                                  # the remaining slots must cover the missing subjects
                            ms = float(np.mean([sims[j, x] for x in mem]))
                            if ms < min_member_sim or min(float(sims[j, x]) for x in mem) < min_any_sim or any(elig[j].lesson_key == elig[x].lesson_key for x in mem):
                                continue
                            cov = sum(1 for x in mem if terms[j] & terms[x])
                            if require_terms and cov < -(-len(mem) // 2):
                                continue                                  # must share content terms with at least half of the members
                            sh = len(span_concepts.get(elig[j].span_id, set()) & set().union(*[span_concepts.get(elig[x].span_id, set()) for x in mem]))
                            cands.append((ms + 0.05 * min(sh, 2) + 0.05 * cov / len(mem) + (0.04 if dom[j] in missing else 0) + rng.random() * 0.03, j, m, sh))
                    if not cands:
                        break
                    cands.sort(reverse=True)
                    _, j, m, sh = cands[rng.randrange(min(3, len(cands)))]
                    links.append({"from": elig[m].span_id, "to": elig[j].span_id, "sim": round(float(sims[m, j]), 3),
                                  "shared_terms": sorted(terms[j] & terms[m])[:4],
                                  "source": "label" if edge_labels.get(frozenset((m, j))) else ("concept" if sh else "dense"),
                                  "labels": edge_labels.get(frozenset((m, j)), []),
                                  "shared_concepts": [(concept_label or {}).get(c, c) for c in sorted(
                                      span_concepts.get(elig[j].span_id, set()) & span_concepts.get(elig[m].span_id, set()))[:3]]})
                    mem.append(j)
                if len(mem) < k or (kind == "3" and len({dom[m] for m in mem}) < 3) or (kind == "2" and len({dom[m] for m in mem}) != 2):
                    fails += 1
                    continue
                ids = [elig[m].span_id for m in mem]
                f = frozenset(ids)
                if any(uses[i] >= max_span_uses for i in ids) or any(len(f & t) / len(f) > tau for t in taken):
                    fails += 1
                    continue
                ss = [corpus.get(i) for i in ids]
                grades = [s.grade for s in ss]
                out.append(Seed(seed_id="seed_" + short_hash(ids + [DEEP_SEEDS_VERSION]), strategy=f"deepx{k}", span_ids=ids, domain=ss[0].domain,
                                grade=max((g for g in grades if g is not None), default=None), grades=grades,
                                book_ids=list(dict.fromkeys(s.book_id for s in ss)),
                                lesson_keys=list(dict.fromkeys(s.lesson_key for s in ss)), target_hop_type="bridge",
                                compatible_hop_types=list(DEEP_HOPS),
                                link={"deep": {"k": k, "template": list(T), "subjects": sorted({s.domain for s in ss}), "chain": links}},
                                version=DEEP_SEEDS_VERSION))
                uses.update(ids)
                taken.append(f)
                made += 1
    return out


# ------------------------------------------------------------------ generation
_GEN_SYS = ("Bạn là giáo viên giàu kinh nghiệm, đang ra đề trắc nghiệm rất KHÓ (suy luận nhiều bước) bám sát sách giáo "
            "khoa Việt Nam (chương trình GDPT 2018). Chỉ trả về JSON hợp lệ.")

_GEN_PROMPT = """CÂU HỎI LIÊN MÔN. Các môn: {subject}. Lớp: {grades}.\nDưới đây là {k} đoạn trích từ sách giáo khoa của các môn khác nhau (mã [S1], ..., [S{k}]; môn ghi trong ngoặc), có thể liên hệ với nhau.

{blocks}

GỢI Ý LIÊN KẾT (tự động, chỉ để tham khảo, có thể không chính xác):
{links}

NHIỆM VỤ: Viết MỘT câu hỏi trắc nghiệm 4 phương án, rất KHÓ, mà muốn trả lời đúng phải dùng TẤT CẢ các đoạn được liệt kê trong \
"gold_spans" (từ {kmin} đến {k} đoạn); mỗi đoạn cung cấp MỘT mắt xích thông tin KHÁC NHAU và BẮT BUỘC, và hai bước liên tiếp nên dùng kiến thức của hai \
môn khác nhau (ví dụ: nguyên nhân → cơ chế → hệ quả; định nghĩa → điều kiện → tính chất số liệu → ý nghĩa; hoặc một tình huống thực tế cần \
một khái niệm của mỗi môn). Nếu bỏ đi bất kỳ một đoạn nào trong gold_spans thì KHÔNG thể xác định được đáp án.

CÁCH LÀM (suy luận ngược, kiểu câu hỏi lồng nhau):
(a) "chain": sắp xếp thành chuỗi các bước, mỗi bước dùng đúng MỘT đoạn, các bước dùng các đoạn KHÁC NHAU; kết quả của bước i (một \
thực thể, điều kiện, đại lượng hay kết luận) là đầu vào của bước i+1: "input" của bước i+1 PHẢI nhắc lại đúng "output" của bước i. Mỗi bước ghi: "span", "input" (điều đã biết), "fact" (thông tin lấy \
từ đoạn đó, bám sát nguyên văn), "output" (điều suy ra).
(b) Viết câu hỏi: chỉ nêu thông tin khởi đầu (hoặc một tình huống cụ thể), rồi hỏi về kết quả của bước CUỐI. Các thực thể/kết luận trung \
gian KHÔNG được nêu tên trong câu hỏi, chỉ được mô tả gián tiếp để học sinh phải tự xác định qua từng đoạn. Đáp án không được suy ra được \
nếu thiếu bất kỳ đoạn nào (kể cả bằng kiến thức thường thức hay đoán).

YÊU CẦU BẮT BUỘC:
1. Câu hỏi tự đứng được như câu hỏi kiểm tra cho học sinh (tối đa khoảng 110 từ, văn phong tự nhiên, là MỘT bài toán/tình huống thống nhất chứ không phải nhiều câu hỏi nhỏ nối bằng \"và\"): TUYỆT ĐỐI không nhắc tới "đoạn \
trích", "đoạn văn", "S1", "theo thông tin trên"...; không chép nguyên văn cả câu trong đoạn trích.
2. Đáp án đúng duy nhất, suy ra chắc chắn từ các đoạn trích (chỉ dùng kiến thức trong đoạn trích; KHÔNG yêu cầu tính toán số học). \
Không dùng kiến thức ngoài sách.
3. Ba phương án nhiễu hợp lí đối với người chưa đọc kĩ sách, cùng loại và độ dài tương đương đáp án đúng, ưu tiên dùng kết quả trung gian \
của chuỗi (dừng giữa chừng), kết quả khi bỏ một bước, hoặc khái niệm/số liệu có trong các đoạn, nhưng CHẮC CHẮN SAI theo các đoạn trích.
4. Câu hỏi phải CỤ THỂ: nêu tên, hiện tượng, số liệu hay tình huống lấy từ các đoạn; không dùng các cụm mơ hồ kiểu "chuỗi quá trình liên ngành", "mối quan hệ giữa các yếu tố".
   Không dùng phương án "tất cả đều đúng/sai", "cả A và B"; tránh câu hỏi phủ định.
5. Đáp án đúng không xuất hiện nguyên văn trong câu hỏi. ĐỘ DÀI (số ký tự) của mỗi phương án nhiễu phải nằm trong khoảng 85%-115% độ dài đáp án đúng, và đáp án đúng KHÔNG được là phương án dài nhất; các phương án cùng cấu trúc câu.
6. Tiếng Việt chuẩn, đúng thuật ngữ của sách. Viết như một câu hỏi kiểm tra thật của giáo viên: một mạch, rõ ràng, một tình huống hoặc một đối tượng cụ thể xuyên suốt; không liệt kê tên các môn học, không dùng các cụm như \"chuỗi liên ngành\", \"xuất phát từ hậu quả của\", không nhắc tới số bước.
7. Nếu các đoạn trích không cho phép lập được chuỗi tự nhiên với ít nhất {kmin} đoạn mà mỗi đoạn đều cần thiết (ví dụ các đoạn chỉ cùng chủ đề \
chứ không phụ thuộc nhau, hoặc chỉ nối với nhau bằng những khái niệm quá chung chung), đặt "feasible": false (các trường khác để rỗng). Không ghép gượng ép các ý không liên quan.

Trả về JSON: feasible, hop_type (một trong: {compat}), link (một câu mô tả chuỗi), chain, question, correct_answer, \
distractors (đúng 3, mỗi cái kèm lí do sai), gold_spans (các đoạn thật sự cần thiết)."""

_QUALITY_PROMPT = """Bạn là chuyên gia khảo thí. Đánh giá câu hỏi trắc nghiệm nhiều bước dưới đây, được ra từ {k} ĐOẠN TRÍCH sách giáo khoa. \
Chấm điểm 1-5 cho:
- "natural": câu hỏi tự nhiên, rõ ràng, học sinh đọc hiểu được mà không cần thấy đoạn trích; không gượng ép ghép các chủ đề không liên quan.
- "needs_both": muốn trả lời đúng phải kết hợp thông tin từ TẤT CẢ các đoạn (mỗi đoạn một mắt xích khác nhau, không phải kiến thức chung).
- "distractors": ba phương án nhiễu hợp lí, cùng dạng với đáp án, nhưng chắc chắn sai theo đoạn trích.
Chỉ dựa vào đoạn trích.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
{options}
Đáp án đúng: {answer}

Trả về JSON: {{"natural": 1-5, "needs_both": 1-5, "distractors": 1-5}}"""

_LOO_PROMPT = """Chỉ dựa vào CÁC ĐOẠN TRÍCH sách giáo khoa dưới đây (không dùng kiến thức bên ngoài, không đoán): các đoạn trích này có \
ĐỦ thông tin để xác định chắc chắn đáp án đúng của câu hỏi không? Nếu đủ, cho biết đáp án; nếu thiếu một thông tin cần thiết nào \
(dù có thể đoán được bằng hiểu biết thông thường), trả lời "sufficient": false và "answer": "none", và nêu thông tin còn thiếu.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
{options}

Trả về JSON: {{"sufficient": true|false, "answer": "A|B|C|D|none", "missing": "<thông tin còn thiếu, nếu có>"}}"""

_VERIFY_PROMPT = """Chỉ dựa vào các ĐOẠN TRÍCH sách giáo khoa dưới đây (không dùng kiến thức bên ngoài), hãy trả lời câu hỏi \
trắc nghiệm và đánh giá từng phương án.
- "correct": đoạn trích cho phép khẳng định phương án đúng.
- "incorrect": đoạn trích cho phép khẳng định phương án sai.
- "undetermined": đoạn trích không đủ để khẳng định đúng hay sai.
"answer" là phương án đúng duy nhất nếu xác định được, ngược lại "none". "used_spans" liệt kê MỌI đoạn bạn đã dùng.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
{options}

Trả về JSON: {{"option_status": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A|B|C|D|none", \
"used_spans": ["E1", ...], "confidence": 0-100}}"""


def deep_gen_schema(labels: list[str], hop_types: list[str]) -> dict:
    s = {"type": "string"}
    k = len(labels)
    return {"type": "object", "properties": {
        "feasible": {"type": "boolean"}, "hop_type": {"type": "string", "enum": hop_types},
        "link": s,
        "chain": {"type": "array", "maxItems": k, "items": {"type": "object", "properties": {
            "span": {"type": "string", "enum": labels}, "input": s, "fact": s, "output": s},
            "required": ["span", "input", "fact", "output"]}},
        "question": s, "correct_answer": s,
        "distractors": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
            "text": s, "why_wrong": s}, "required": ["text", "why_wrong"]}},
        "gold_spans": {"type": "array", "items": {"type": "string", "enum": labels}, "maxItems": k},
    }, "required": ["feasible", "hop_type", "link", "chain", "question", "correct_answer", "distractors", "gold_spans"]}


def _link_hints(seed: Seed, labels: list[str]) -> str:
    pos = dict(zip(seed.span_ids, labels))
    out = []
    for p in (seed.link or {}).get("deep", {}).get("chain", []):
        extra = (" | thực thể xuất hiện nguyên văn trong cả hai đoạn: " + ", ".join(p["labels"])) if p.get("labels") else ""
        extra += (" | khái niệm chung: " + ", ".join(p["shared_concepts"])) if p.get("shared_concepts") else ""
        extra += (" | từ chung: " + ", ".join(p["shared_terms"])) if p.get("shared_terms") else ""
        out.append(f"- [{pos[p['from']]}] ~ [{pos[p['to']]}] (độ tương đồng ngữ nghĩa {p['sim']}){extra}")
    return "\n".join(out) or "(không có)"


def generate_deep(seed: Seed, corpus: SpanCorpus, llm, *, sample: int = 0, temperature: float = 0.7, max_tokens: int = 4500,
                  kmin: Optional[int] = None, extra: str = "", cand_key: Optional[str] = None) -> dict[str, Any]:
    spans = [corpus.get(i) for i in seed.span_ids]
    k = len(spans)
    labels = [f"S{i + 1}" for i in range(k)]
    kmin = max(4, min(kmin if kmin is not None else k - 2, k))
    subs = list(dict.fromkeys(DOMAIN_TO_SUBJECT_VI.get(s.domain, s.domain) for s in spans))
    grades = sorted({s.grade for s in spans if s.grade is not None})
    compat = list(DEEP_HOPS)
    prompt = _GEN_PROMPT.format(subject=", ".join(subs), grades=", ".join(map(str, grades)) or "?",
                                k=k, kmin=kmin, blocks="\n\n".join(f"(môn {DOMAIN_TO_SUBJECT_VI.get(s.domain, s.domain)})\n" + span_block(l, s, 1500) for l, s in zip(labels, spans)),
                                links=_link_hints(seed, labels), compat=", ".join(c for c in compat if c != "calculation")) + extra
    cand: dict[str, Any] = {"cand_id": "cand_" + sha1(cand_key or f"{seed.seed_id}{DEEP_GEN_VERSION}{llm.model}{sample}")[:12],
                            "seed_id": seed.seed_id, "domain": seed.domain, "grade": seed.grade, "seed_span_ids": seed.span_ids,
                            "strategy": seed.strategy, "target_hop_type": seed.target_hop_type, "deep_k": k, "min_spans": kmin,
                            "generator": {"model": llm.model, "prompt_version": DEEP_GEN_VERSION, "temperature": temperature}}

    def ok(o):
        return (not _bool(o.get("feasible"))) or (bool(str(o.get("question", "")).strip()) and isinstance(o.get("distractors"), list))
    try:
        obj, r = chat_json(llm, [{"role": "system", "content": _GEN_SYS}, {"role": "user", "content": prompt}],
                           deep_gen_schema(labels, [c for c in compat if c != "calculation"]), temperature=temperature,
                           max_tokens=max_tokens, validate=ok)
        cand["generator"].update(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens)
    except Exception as e:  # noqa: BLE001
        cand.update(feasible=False, error=f"{type(e).__name__}: {e}")
        return cand
    lab2id = dict(zip(labels, seed.span_ids))
    obj["distractors"] = [d if isinstance(d, dict) else {"text": str(d), "why_wrong": ""} for d in obj.get("distractors") or []]
    gold = [lab2id[l] for l in dict.fromkeys(_label(x, labels) for x in obj.get("gold_spans") or []) if l]
    chain = [{"span_id": lab2id.get(_label(st.get("span"), labels)), "input": str(st.get("input", "")), "fact": str(st.get("fact", "")),
              "output": str(st.get("output", ""))} for st in obj.get("chain") or [] if isinstance(st, dict)]
    dis = [str(d.get("text", "")).strip() for d in obj["distractors"] if isinstance(d, dict)]
    feasible = _bool(obj.get("feasible")) and bool(str(obj.get("question", "")).strip())
    hop = obj.get("hop_type") if obj.get("hop_type") in DEEP_HOPS else "bridge"
    cand.update(feasible=feasible, hop_type=hop, link=obj.get("link", ""), chain=chain,
                facts=[{"span_id": lab2id.get(_label(f.get("span"), labels)), "fact": f.get("fact", "")}
                       for f in obj.get("facts", []) if isinstance(f, dict)],
                question=str(obj.get("question", "")).strip(), correct_answer=str(obj.get("correct_answer", "")).strip(),
                distractors=dis, distractor_notes=[str(d.get("why_wrong", "")) for d in obj["distractors"] if isinstance(d, dict)],
                gold_span_ids=gold,
                reasoning_steps=[{"span_id": c["span_id"], "step": f"{c['fact']} => {c['output']}"} for c in chain])
    if feasible and len(dis) == 3 and cand["correct_answer"]:
        cand["options"], cand["answer"] = assemble_options(cand["cand_id"], cand["correct_answer"], dis)
    return cand


def deep_structural(c: dict[str, Any], corpus: SpanCorpus) -> dict[str, Any]:
    """Existing structural checks (as for 2-3 hops) plus: >=4 gold spans, gold within the seed, every gold span carries a chain step."""
    f = structural_checks(c, corpus)
    gold = c.get("gold_span_ids") or []
    steps = {st["span_id"] for st in c.get("chain", []) if st.get("span_id")}
    f["deep_min_spans"] = len(gold) >= 4
    f["deep_gold_in_seed"] = set(gold) <= set(c.get("seed_span_ids") or [])
    f["deep_chain_covers_gold"] = set(gold) <= steps
    f["deep_distinct_chain_steps"] = len([st for st in c.get("chain", []) if st.get("span_id") in gold]) >= len(gold)
    from otter.evaluation.hard_seeds import _STOP
    from otter.evaluation.text import tokens
    gold_steps = [st for st in c.get("chain", []) if st.get("span_id") in gold]
    # every cited fact comes from its span (>= 75% of its syllables occur in the span) -> no facts outside the spans
    f["deep_facts_grounded"] = all(
        (lambda ft, sp: bool(ft) and sum(1 for x in ft if x in sp) / len(ft) >= 0.75)(tokens(st.get("fact", "")), set(tokens(corpus.get(st["span_id"]).text)))
        for st in gold_steps if corpus.get(st["span_id"]))
    # consecutive steps are chained: the input of step i+1 restates the output of step i (>= 2 shared content syllables)
    content = lambda s: {x for x in tokens(s) if x not in _STOP and len(x) >= 2}
    f["deep_chain_linked"] = all(len(content(a.get("output", "")) & content(b.get("input", ""))) >= 2
                                 for a, b in zip(gold_steps, gold_steps[1:]))
    extra = ("deep_min_spans", "deep_gold_in_seed", "deep_chain_covers_gold", "deep_distinct_chain_steps", "deep_facts_grounded",
             "deep_chain_linked")
    f["structural_fail"] = list(f["structural_fail"]) + [k for k in extra if not f[k]]
    f["structural_ok"] = bool(f["structural_ok"]) and all(f[k] for k in extra)
    return f


# ------------------------------------------------------------------ LLM checks
QUALITY_SCHEMA = {"type": "object", "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5}
                                                   for k in ("natural", "needs_both", "distractors")},
                  "required": ["natural", "needs_both", "distractors"]}


def check_quality_deep(c: dict[str, Any], corpus: SpanCorpus, llm) -> dict[str, Any]:
    gold = list(c["gold_span_ids"])
    labels = [f"E{i + 1}" for i in range(len(gold))]
    p = _QUALITY_PROMPT.format(k=len(gold), blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)),
                               question=c["question"], options=fmt_options(c["options"]), answer=c["answer"])

    def num(o, k):
        try:
            return int(float(o.get(k)))
        except (TypeError, ValueError):
            return None
    o, _ = chat_json(llm, [{"role": "user", "content": p}], QUALITY_SCHEMA, max_tokens=120,
                     validate=lambda o: all(num(o, k) is not None for k in QUALITY_SCHEMA["required"]))
    return {**{k: num(o, k) for k in QUALITY_SCHEMA["required"]}, "model": llm.model, "prompt_version": "otter_hard_deep_quality_v1"}


def check_verify_deep(c: dict[str, Any], corpus: SpanCorpus, llm) -> dict[str, Any]:
    """Key is entailed by ALL gold spans and is the only entailed option (same contract as hard_gen.check_verify, k spans)."""
    from otter.evaluation.hard_gen import _status
    gold = sorted(c["gold_span_ids"], key=lambda s: sha1(c["cand_id"] + s))
    labels = [f"E{i + 1}" for i in range(len(gold))]
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + ":verify")
    p = _VERIFY_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)),
                              question=c["question"], options=fmt_options(opts))
    valid = lambda o: isinstance(o.get("option_status"), dict) and all(_status(o["option_status"].get(k)) for k in LETTERS)
    schema = {"type": "object", "properties": {
        "option_status": {"type": "object", "properties": {k: {"type": "string", "enum": ["correct", "incorrect", "undetermined"]}
                                                            for k in LETTERS}, "required": list(LETTERS)},
        "answer": {"type": "string", "enum": list(LETTERS) + ["none"]},
        "used_spans": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["option_status", "answer", "used_spans", "confidence"]}
    o, _ = chat_json(llm, [{"role": "user", "content": p}], schema, max_tokens=500, validate=valid)
    st = {back[k]: _status(o["option_status"].get(k)) for k in LETTERS}
    a = _letter(o.get("answer"))
    a = back.get(a) if a else "none"
    lab2 = dict(zip(labels, gold))
    used = [_label(x, labels) for x in o.get("used_spans") or []]
    return {"answer": a, "option_status": st, "confidence": o.get("confidence"), "used_span_ids": [lab2[x] for x in used if x],
            "model": llm.model, "prompt_version": "otter_hard_deep_verify_v1"}


def check_loo(c: dict[str, Any], held_out: str, corpus: SpanCorpus, llm, perm: int = 0) -> dict[str, Any]:
    """Leave-one-out necessity probe: do the OTHER gold spans suffice to derive the key? (option order re-permuted per `perm`)."""
    rest = sorted((s for s in c["gold_span_ids"] if s != held_out), key=lambda s: sha1(f"{c['cand_id']}:loo{perm}:{s}"))
    labels = [f"E{i + 1}" for i in range(len(rest))]
    opts, ans, back = permute(c["options"], c["answer"], f"{c['cand_id']}:loo:{held_out}:{perm}")
    p = _LOO_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, rest)),
                           question=c["question"], options=fmt_options(opts))
    o, _ = chat_json(llm, [{"role": "user", "content": p}], SINGLE_SCHEMA, max_tokens=300, validate=lambda o: "sufficient" in o)
    a = _letter(o.get("answer"))
    return {"held_out": held_out, "perm": perm, "sufficient": _bool(o.get("sufficient")), "answer": back.get(a) if a else "none",
            "missing": o.get("missing", ""), "model": llm.model, "prompt_version": LOO_VERSION}


def loo_verdict(c: dict[str, Any], get: Callable[[str], Optional[dict]], perms: int = 2) -> dict[str, Any]:
    """Fold the leave-one-out records. An item passes only if, for EVERY gold span s and EVERY permutation, the remaining
    spans are not judged sufficient-with-the-key. `loo_missing` lists spans whose check is absent (item cannot pass)."""
    return loo_fold(c["gold_span_ids"], c["answer"], {s: [get(f"loo:{s}:{p}") for p in range(perms)] for s in c["gold_span_ids"]})


def loo_fold(gold: list[str], key: str, recs: dict[str, list[Optional[dict]]]) -> dict[str, Any]:
    missing = [s for s in gold if any(r is None for r in recs.get(s, [None]))]
    derivable = [s for s in gold if any(r is not None and r["sufficient"] and r["answer"] == key for r in recs.get(s, []))]
    return {"loo_missing": missing, "loo_derivable_without": derivable, "loo_all_necessary": not missing and not derivable and len(gold) > 0}


# ------------------------------------------------------------------ enrichment (k spans; larger token budget than hard_v2.enrich_item)
def enrich_deep(row: dict, corpus: SpanCorpus, llm) -> dict[str, Any]:
    from otter.evaluation import hard_v2 as v2
    gold = row["gold_span_ids"]
    labels = [f"E{i + 1}" for i in range(len(gold))]
    p = v2._ENRICH_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), question=row["question"],
                                 options=fmt_options(row["options"]), answer=f"{row['answer']}. {row['options'][row['answer']]}",
                                 ped=", ".join(v2.PED_TYPES_VI))
    best = None
    for k in range(3):
        try:
            o, _ = chat_json(llm, [{"role": "user", "content": p}], {}, max_tokens=1800, temperature=0.0 if k == 0 else 0.4,
                             validate=lambda o: isinstance(o.get("reference_answer"), str) and o.get("reference_answer").strip())
        except Exception:  # noqa: BLE001
            continue
        q = {l: str(v) for l, v in (o.get("quotes") or {}).items() if isinstance(v, str)}
        ok = {gold[labels.index(l)]: v2._squash(v) in v2._squash(corpus.get(gold[labels.index(l)]).text)
              for l, v in q.items() if l in labels and v.strip()}
        res = {"bloom": o.get("bloom") if o.get("bloom") in v2.BLOOM else None,
               "ped_types": {gold[labels.index(l)]: v for l, v in (o.get("ped_types") or {}).items() if l in labels},
               "reference_answer": o["reference_answer"].strip(), "explanation": str(o.get("explanation", "")).strip(),
               "quotes": {gold[labels.index(l)]: v for l, v in q.items() if l in labels},
               "quotes_verbatim": ok, "enrich_model": llm.model}
        res["quotes_ok"] = len(ok) == len(gold) and all(ok.values())
        if best is None or (res["quotes_ok"] and not best["quotes_ok"]):
            best = res
        if res["quotes_ok"]:
            break
    return best or {"error": "enrich_failed"}


# ------------------------------------------------------------------ literal-entity links (ontology concept labels / aliases)
def label_index(elig: list[Span], onto_db, *, min_df: int = 2, max_df: int = 15) -> dict[str, list[int]]:
    """Ontology concept label / alias (>= 4 chars, <= 5 words) -> indices of eligible spans that mention it literally (whole phrase)."""
    import sqlite3
    labs: set[str] = set()
    with sqlite3.connect(f"file:{onto_db}?mode=ro", uri=True) as c:
        labs |= {r[0].lower() for r in c.execute("select label_norm from concepts")}
        labs |= {r[0].lower() for r in c.execute("select alias_norm from concept_aliases")}
    txt = [re.sub(r"\s+", " ", s.text.lower()) for s in elig]
    inv: dict[str, list[int]] = {}
    for ln in labs:
        if len(ln) < 4 or len(ln.split()) > 5:
            continue
        rx = re.compile(r"(?<![\w])" + re.escape(ln) + r"(?![\w])")
        hit = [i for i, tx in enumerate(txt) if ln in tx and rx.search(tx)]
        if min_df <= len(hit) <= max_df:
            inv[ln] = hit
    return inv


def label_anchor_edges(elig: list[Span], dense: np.ndarray, inv: dict[str, list[int]], *, min_sim: float = 0.40) -> list[dict]:
    """Cross-subject span pairs that literally mention the same ontology concept; usable as `anchors` of cross_deep_seeds."""
    pairs: dict[tuple[int, int], list[str]] = defaultdict(list)
    for ln, hit in inv.items():
        for x, a in enumerate(hit):
            for b in hit[x + 1:]:
                if elig[a].domain != elig[b].domain and float(dense[a] @ dense[b]) >= min_sim:
                    pairs[(a, b)].append(ln)
    return [{"a": elig[a].span_id, "b": elig[b].span_id, "sources": ["label"], "labels": sorted(ls, key=lambda s: -len(s))[:3],
             "sim": float(dense[a] @ dense[b])} for (a, b), ls in pairs.items()]
