"""OTTER-Hard v2 additions: ontology-driven seeds (rules -> conditional_rule / calculation, relation chains ->
causal_chain / bridge), deterministic arithmetic verification (otter/tools/calculate.py), distractor length
matching (rewrite + re-verification) and subject/hop/difficulty-balanced selection.

Nothing here relaxes the correctness filters of v1 (key entailed by the gold spans and unique; gold spans are never
exercise/question spans; the item is not answerable from a single gold span).
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import chat_json, fmt_options, span_block
from otter.evaluation.hard_seeds import Seed, eligible
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.text import short_hash, sha1, tokens

SEEDS_V2_VERSION = "otter_hard_seeds_v2"
GEN_V3_VERSION = "otter_hard_gen_v3"
REWRITE_VERSION = "otter_hard_distractor_rewrite_v1"

CAUSAL_REL = {"causes", "caused_by", "affects"}
BRIDGE_REL = {"is_a", "has_part", "located_in", "part_of", "has_property", "occurs_in", "measured_in", "defined_as",
              "authored_by", "example_of"}
_JUNK_LABEL = re.compile(r"^(có|được|là|sự|việc)\b|^\W*$", re.I)


def eligible_rule_span(s: Span, min_chars: int = 60) -> bool:
    """Looser than hard_seeds.eligible (formula-bearing spans are short and symbol-heavy) but never an exercise."""
    t = s.text.strip()
    if not s.domain or s.is_exercise() or not (s.section_id or s.heading_path) or not (min_chars <= len(t) <= 2200):
        return False
    if s.pedagogical_type in ("objective", "figure_caption", "question", "exercise", "activity") or t.startswith("["):
        return False
    if sum(ch.isalpha() for ch in t) / max(1, len(t)) < 0.35:
        return False
    return len(tokens(t)) >= 12


# ------------------------------------------------------------------ ontology access
class Onto:
    def __init__(self, db: Path):
        self.c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        self.c.row_factory = sqlite3.Row
        self.label = {r["concept_id"]: r["label"] for r in self.c.execute("SELECT concept_id,label FROM concepts")}
        self.item_spans: dict[tuple[str, str], list[str]] = defaultdict(list)
        for r in self.c.execute("SELECT item_kind,item_id,span_id FROM evidence_links"):
            self.item_spans[(r[0], r[1])].append(r[2])
        # concept -> spans through concept/fact/rule/relation evidence
        self.concept_spans: dict[str, set[str]] = defaultdict(set)
        for (k, i), ss in self.item_spans.items():
            if k == "concept":
                self.concept_spans[i].update(ss)
        for r in self.c.execute("SELECT assertion_id,concept_id FROM fact_concepts"):
            self.concept_spans[r[1]].update(self.item_spans.get(("fact", r[0]), []))
        for r in self.c.execute("SELECT rule_id,concept_id FROM rule_concepts"):
            self.concept_spans[r[1]].update(self.item_spans.get(("rule", r[0]), []))
        for r in self.c.execute("SELECT subject_id,object_id,relation_id FROM semantic_relations WHERE source_support='supported'"):
            for cid in (r[0], r[1]):
                self.concept_spans[cid].update(self.item_spans.get(("relation", r[2]), []))

    def rules(self) -> list[dict]:
        out = []
        for r in self.c.execute("SELECT * FROM rules WHERE strict_executable=1 AND source_support='supported' "
                                "AND status!='derived_conclusion'"):
            j = json.loads(r["json"])
            d = dict(r)
            d["j"] = j
            d["spans"] = list(dict.fromkeys(j.get("source_span_ids") or self.item_spans.get(("rule", r["rule_id"]), [])))
            d["concepts"] = self.rule_concepts(j)
            out.append(d)
        return out

    @staticmethod
    def rule_concepts(j: dict) -> set[str]:
        cs = set()
        for part in ("premises", "conclusions", "guards"):
            for p in j.get(part) or []:
                for a in p.get("args") or []:
                    if isinstance(a, str) and ":c:" in a:
                        cs.add(a)
        for v in (j.get("variables") or {}).values():
            if isinstance(v, str) and ":c:" in v:
                cs.add(v)
        return cs

    def relations(self) -> list[dict]:
        out = []
        for r in self.c.execute("SELECT * FROM semantic_relations WHERE source_support='supported'"):
            j = json.loads(r["json"])
            d = dict(r)
            d["spans"] = list(dict.fromkeys(j.get("source_span_ids") or []))
            d["qualifier"] = j.get("qualifier")
            out.append(d)
        return out


def _mk_seed(strategy: str, ids: list[str], corpus: SpanCorpus, hop: str, compat: list[str], link: dict, tag: str) -> Seed:
    ss = [corpus.get(i) for i in ids]
    grades = [s.grade for s in ss]
    return Seed(seed_id="seed_" + short_hash(ids + [tag]), strategy=strategy, span_ids=ids, domain=ss[0].domain,
                grade=max((g for g in grades if g is not None), default=None), grades=grades,
                book_ids=list(dict.fromkeys(s.book_id for s in ss)),
                lesson_keys=list(dict.fromkeys(s.lesson_key for s in ss)), target_hop_type=hop,
                compatible_hop_types=compat, link=link, version=SEEDS_V2_VERSION)


def rule_seeds(corpus: SpanCorpus, onto: Onto, *, seed: int = 21, max_span_uses: int = 3,
               min_chars: int = 60) -> list[Seed]:
    rng = random.Random(seed)
    elig = {s.span_id for s in corpus.spans if eligible_rule_span(s, min_chars)}
    rules = [r for r in onto.rules() if r["subject"] != "history" and r["rule_type"] in ("formula", "conditional")]
    for r in rules:
        r["espans"] = [s for s in r["spans"] if s in elig]
    uses: Counter = Counter()
    seeds: list[Seed] = []
    seen: set = set()

    def commit(ids, strategy, hop, compat, link, tag):
        key = frozenset(ids)
        if key in seen or any(uses[i] >= max_span_uses for i in ids):
            return False
        if len({corpus.get(i).book_id for i in ids}) > 2:
            return False
        seen.add(key)
        uses.update(ids)
        seeds.append(_mk_seed(strategy, ids, corpus, hop, compat, link, tag))
        return True

    def rinfo(r):
        j = r["j"]
        return {"rule_id": r["rule_id"], "rule_type": r["rule_type"], "text": r["text"][:500], "formula": r["formula"],
                "variables": {k: (onto.label.get(v, v) if isinstance(v, str) else v) for k, v in (j.get("variables") or {}).items()},
                "premises": [p.get("text") for p in j.get("premises") or []],
                "guards": [p.get("text") for p in j.get("guards") or []],
                "conclusions": [p.get("text") for p in j.get("conclusions") or []]}

    by_subj_formula: dict[str, list[dict]] = defaultdict(list)
    for r in rules:
        if r["rule_type"] == "formula" and r["formula"] and r["espans"]:
            by_subj_formula[r["subject"]].append(r)
    # (1) two chained formula rules: the output concept of one is an input variable of the other
    for subj, fr in by_subj_formula.items():
        for r1 in fr:
            outs = {a for p in r1["j"].get("conclusions") or [] for a in p.get("args") or [] if isinstance(a, str)}
            outs |= {v for v in (r1["j"].get("variables") or {}).values() if isinstance(v, str)}
            for r2 in fr:
                if r2["rule_id"] == r1["rule_id"] or not set(r2["espans"]).isdisjoint(r1["espans"]):
                    continue
                ins = {v for v in (r2["j"].get("variables") or {}).values() if isinstance(v, str) and ":c:" in v}
                shared = outs & ins
                if shared and r1["formula"] != r2["formula"]:
                    ids = [r1["espans"][0], r2["espans"][0]]
                    commit(ids, "rule_formula_chain", "calculation", ["calculation", "bridge"],
                           {"onto": {"kind": "formula_chain", "rules": [rinfo(r1), rinfo(r2)],
                                     "shared_concepts": [onto.label.get(c, c) for c in sorted(shared)[:4]]}},
                           r1["rule_id"] + r2["rule_id"])
    # (2) a rule plus a concept-linked partner span (formula -> calculation, conditional -> conditional_rule)
    deg = {c: len(s) for c, s in onto.concept_spans.items()}
    for r in rules:
        if not r["espans"]:
            continue
        hop = "calculation" if (r["rule_type"] == "formula" and r["formula"]) else "conditional_rule"
        own = r["espans"][:2]
        cands: Counter = Counter()
        for c in r["concepts"]:
            if deg.get(c, 0) > 60:
                continue
            for sp in onto.concept_spans.get(c, ()):
                if sp in elig and sp not in own and corpus.get(sp).domain == r["subject"]:
                    cands[sp] += 1.0 / (1 + deg.get(c, 1)) ** 0.5
        pool: list[tuple[float, str]] = []
        for sp, sc in cands.items():
            a = corpus.get(sp)
            bonus = 0.3 if a.lesson_key != corpus.get(own[0]).lesson_key else 0.0
            pool.append((sc + bonus + rng.random() * 0.2, sp))
        pool.sort(reverse=True)
        made = 0
        if len(own) >= 2 and made < 1:
            if commit(own[:2], "rule_own_spans", hop, [hop, "bridge"], {"onto": {"kind": "rule", "rules": [rinfo(r)]}}, r["rule_id"]):
                made += 1
        for _, sp in pool[:6]:
            if made >= (3 if hop == "calculation" else 1):
                break
            if commit([own[0], sp], "rule_concept_partner", hop, [hop, "bridge"],
                      {"onto": {"kind": "rule", "rules": [rinfo(r)]}}, r["rule_id"] + sp):
                made += 1
    return seeds


def relation_chain_seeds(corpus: SpanCorpus, onto: Onto, *, seed: int = 22, max_per_subject: int = 80,
                         max_span_uses: int = 2, min_chars: int = 80) -> list[Seed]:
    rng = random.Random(seed)
    elig = {s.span_id for s in corpus.spans if eligible_rule_span(s, min_chars)}
    rels = [r for r in onto.relations() if any(s in elig for s in r["spans"])
            and not _JUNK_LABEL.match(onto.label.get(r["subject_id"], "")) and not _JUNK_LABEL.match(onto.label.get(r["object_id"], ""))
            and onto.label.get(r["subject_id"]) and onto.label.get(r["object_id"])]
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in rels:
        by_sub[r["subject_id"]].append(r)
    by_obj: dict[str, list[dict]] = defaultdict(list)
    for r in rels:
        by_obj[r["object_id"]].append(r)
    uses: Counter = Counter()
    per_subj: Counter = Counter()
    seeds, seen = [], set()
    pairs = []
    for r1 in rels:
        for r2 in by_sub.get(r1["object_id"], []):
            if r2["relation_id"] == r1["relation_id"] or r2["object_id"] == r1["subject_id"]:
                continue
            e1 = [s for s in r1["spans"] if s in elig]
            e2 = [s for s in r2["spans"] if s in elig]
            if not e1 or not e2 or set(e1) == set(e2) or e1[0] == e2[0]:
                continue
            ca = r1["relation_type"] in CAUSAL_REL and r2["relation_type"] in CAUSAL_REL
            br = r1["relation_type"] in BRIDGE_REL | CAUSAL_REL and r2["relation_type"] in BRIDGE_REL | CAUSAL_REL
            if not (ca or br):
                continue
            pairs.append((r1, r2, e1[0], e2[0], "causal_chain" if ca else "bridge"))
    rng.shuffle(pairs)
    pairs.sort(key=lambda p: (p[4] != "causal_chain",))          # causal chains first
    for r1, r2, s1, s2, hop in pairs:
        subj = r1["subject"]
        if per_subj[subj] >= max_per_subject or uses[s1] >= max_span_uses or uses[s2] >= max_span_uses:
            continue
        key = frozenset((s1, s2))
        if key in seen:
            continue
        a, b, c = (onto.label[r1["subject_id"]], onto.label[r1["object_id"]], onto.label[r2["object_id"]])
        seen.add(key)
        uses.update([s1, s2])
        per_subj[subj] += 1
        compat = [hop] + (["bridge"] if hop == "causal_chain" else ["causal_chain"])
        seeds.append(_mk_seed("relation_chain", [s1, s2], corpus, hop, compat,
                              {"onto": {"kind": "relation_chain", "chain": [
                                  {"from": a, "rel": r1["relation_type"], "to": b, "qualifier": r1.get("qualifier"), "span_idx": 0},
                                  {"from": b, "rel": r2["relation_type"], "to": c, "qualifier": r2.get("qualifier"), "span_idx": 1}]}},
                              r1["relation_id"] + r2["relation_id"]))
    return seeds


def relation_chain3_seeds(corpus: SpanCorpus, onto: Onto, *, seed: int = 23, max_total: int = 450, max_span_uses: int = 2,
                          min_chars: int = 80) -> list[Seed]:
    """3-hop seeds: three supported relations a->b->c->d whose evidence spans are three distinct eligible spans."""
    rng = random.Random(seed)
    elig = {s.span_id for s in corpus.spans if eligible_rule_span(s, min_chars)}
    rels = [r for r in onto.relations() if any(s in elig for s in r["spans"]) and r["relation_type"] in (CAUSAL_REL | BRIDGE_REL)
            and onto.label.get(r["subject_id"]) and onto.label.get(r["object_id"])
            and not _JUNK_LABEL.match(onto.label[r["subject_id"]]) and not _JUNK_LABEL.match(onto.label[r["object_id"]])]
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in rels:
        by_sub[r["subject_id"]].append(r)
    first = lambda r: next(s for s in r["spans"] if s in elig)
    trip = []
    for r1 in rels:
        for r2 in by_sub.get(r1["object_id"], []):
            if r2["object_id"] in (r1["subject_id"],):
                continue
            for r3 in by_sub.get(r2["object_id"], []):
                if r3["object_id"] in (r1["subject_id"], r2["subject_id"]):
                    continue
                ids = [first(r1), first(r2), first(r3)]
                if len(set(ids)) == 3 and len({corpus.get(i).domain for i in ids}) == 1:
                    trip.append((r1, r2, r3, ids))
    rng.shuffle(trip)
    uses, out, seen, per = Counter(), [], set(), Counter()
    for r1, r2, r3, ids in trip:
        if len(out) >= max_total or any(uses[i] >= max_span_uses for i in ids) or frozenset(ids) in seen:
            continue
        dom = corpus.get(ids[0]).domain
        if per[dom] >= max_total // 5:
            continue
        seen.add(frozenset(ids)); uses.update(ids); per[dom] += 1
        causal = all(r["relation_type"] in CAUSAL_REL for r in (r1, r2, r3))
        hop = "causal_chain" if causal else "bridge"
        ch = [{"from": onto.label[r["subject_id"]], "rel": r["relation_type"], "to": onto.label[r["object_id"]], "qualifier": r.get("qualifier"), "span_idx": k}
              for k, r in enumerate((r1, r2, r3))]
        out.append(_mk_seed("relation_chain3", ids, corpus, hop, [hop, "bridge", "causal_chain"], {"onto": {"kind": "relation_chain", "chain": ch}},
                            r1["relation_id"] + r2["relation_id"] + r3["relation_id"]))
    return out


# ------------------------------------------------------------------ prompts
LIT_NOTE = ("LƯU Ý MÔN NGỮ VĂN: chỉ hỏi về NỘI DUNG được nêu rõ trong đoạn (tác giả, hoàn cảnh sáng tác, thể loại, nhân vật, "
            "sự kiện, chi tiết, đặc điểm nghệ thuật/phân tích đã được sách trình bày). Không yêu cầu cảm thụ hay diễn giải "
            "riêng; không bịa thêm ý nghĩa ngoài đoạn.")

CALC_NOTE = ("YÊU CẦU RIÊNG CHO CÂU TÍNH TOÁN: đưa số liệu cụ thể (số đẹp, đơn vị rõ ràng) vào câu hỏi; công thức/quan hệ phải lấy "
             "từ các đoạn trích (không nêu sẵn công thức trong câu hỏi). Đáp án đúng và 3 phương án nhiễu đều là kết quả SỐ kèm đơn vị "
             "(nhiễu = kết quả khi dùng sai công thức, quên một bước hoặc đổi đơn vị sai) và khác nhau rõ rệt. Thêm trường JSON "
             '"calc": {"expression": "<biểu thức số học thuần túy chỉ gồm các số trong câu hỏi và + - * / ** ( ), ví dụ (2*3.5)/0.5; '
             'không dùng tên biến>", "unit": "<đơn vị của kết quả>"} cho ra ĐÚNG giá trị số của đáp án đúng (cùng đơn vị, cùng cách làm tròn).')


def extra_prompt(seed: Seed) -> str:
    parts = []
    o = (seed.link or {}).get("onto")
    if o:
        if o["kind"] in ("rule", "formula_chain"):
            lines = []
            for r in o["rules"]:
                l = f"- {r['text']}"
                if r.get("formula"):
                    l += f" | công thức: {r['formula']}"
                if r.get("variables"):
                    l += " | biến: " + "; ".join(f"{k} = {v}" for k, v in list(r["variables"].items())[:8])
                lines.append(l)
            parts.append("GỢI Ý TỪ ONTOLOGY (quy tắc/công thức đã trích từ chính các đoạn, chỉ để định hướng; đáp án vẫn phải suy ra "
                         "được từ các đoạn trích):\n" + "\n".join(lines))
            if o["kind"] == "formula_chain":
                parts.append("Hai công thức nối nhau qua đại lượng chung: " + ", ".join(o.get("shared_concepts") or [])
                             + ". Hãy ra bài toán 2 bước cần cả hai công thức (kết quả của bước 1 là đầu vào của bước 2).")
        elif o["kind"] == "cross":
            sv = [DOMAIN_TO_SUBJECT_VI.get(x, x) for x in o["subjects"]]
            parts.append(f"CÂU HỎI LIÊN MÔN: [S1] thuộc môn {sv[0]}, [S2] thuộc môn {sv[1]} (sách giáo khoa khác nhau). Ra một câu hỏi TÍCH HỢP kiến "
                         "thức hai môn theo phong cách bài đánh giá năng lực (một tình huống/hiện tượng thực tế cần dùng cả hai phần lý thuyết). "
                         "Phải cần cả hai đoạn mới trả lời được; không nêu tên môn học; chỉ dùng kiến thức trong hai đoạn trích; "
                         "không sao chép hay mô phỏng đề thi có sẵn.")
        elif o["kind"] == "relation_chain":
            ch = o["chain"]
            parts.append("GỢI Ý TỪ ONTOLOGY (chuỗi quan hệ nối hai đoạn; mỗi mắt xích nằm ở một đoạn theo thứ tự [S1], [S2], ...; "
                         "KHÔNG nêu tên khái niệm trung gian trong câu hỏi):\n"
                         + "\n".join(f"- {c['from']} --{c['rel']}--> {c['to']}" + (f" ({c['qualifier']})" if c.get("qualifier") else "")
                                     for c in ch))
    if seed.target_hop_type == "calculation" or (o and o.get("kind") == "formula_chain"):
        parts.append(CALC_NOTE)
    if seed.domain == "literature":
        parts.append(LIT_NOTE)
    return "\n\n".join(parts)


# ------------------------------------------------------------------ arithmetic verification
_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?:\s*(?:[×x·*]|\\times|\\cdot)\s*10\s*\^?\s*\{?\s*[-−]?\d+\s*\}?)?")


def parse_numbers(text: str) -> list[float]:
    out = []
    t = text.replace("−", "-").replace("$", " ")
    t = re.sub(r"(\d)\s(?=\d{3}\b)", r"\1", t)                    # thousands separator "1 000"
    for m in _NUM.finditer(t):
        s = m.group(0)
        mm = re.match(r"(-?\d+(?:[.,]\d+)?)\s*(?:[×x·*]|\\times|\\cdot)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?", s)
        try:
            if mm:
                out.append(float(mm.group(1).replace(",", ".")) * 10 ** int(mm.group(2)))
            else:
                out.append(float(s.replace(",", ".")))
        except ValueError:
            pass
    return out


def check_calc(c: dict[str, Any], rel_tol: float = 0.03) -> dict[str, Any]:
    """Deterministic: evaluate the generator's arithmetic expression with the safe calculator and require that it
    reproduces the keyed answer's number and that no distractor carries that number."""
    from otter.tools.calculate import calculate
    res: dict[str, Any] = {"calc_present": False, "calc_ok": None}
    calc = c.get("calc")
    if not isinstance(calc, dict) or not calc.get("expression"):
        return res
    res["calc_present"] = True
    try:
        r = calculate(None, expression=str(calc["expression"]))
    except Exception as e:  # noqa: BLE001
        res.update(calc_ok=False, calc_error=f"{type(e).__name__}: {e}")
        return res
    if r.get("status") != "ok" or r.get("value") is None:
        res.update(calc_ok=False, calc_error=str(r.get("reason") or r.get("status")))
        return res
    v = float(r["value"])
    res["calc_value"] = v
    near = lambda x: abs(x - v) <= rel_tol * abs(v) + 1e-12
    key_ok = any(near(x) for x in parse_numbers(c.get("correct_answer", "")))
    dis_hit = [d for d in c.get("distractors", []) if any(near(x) for x in parse_numbers(d))]
    res.update(calc_key_matches=key_ok, calc_distractors_matching_key=len(dis_hit), calc_ok=bool(key_ok and not dis_hit))
    return res


# ------------------------------------------------------------------ distractor length matching
_REWRITE_SYS = "Bạn là giáo viên giàu kinh nghiệm, đang hoàn thiện câu hỏi trắc nghiệm. Chỉ trả về JSON hợp lệ."
_REWRITE_PROMPT = """Dưới đây là các ĐOẠN TRÍCH sách giáo khoa và một câu hỏi trắc nghiệm có đáp án đúng đã được kiểm chứng.

ĐOẠN TRÍCH:
{blocks}

Câu hỏi: {question}
Đáp án đúng: {correct}
Phương án nhiễu hiện tại:
{current}

NHIỆM VỤ: viết lại đúng 3 phương án nhiễu mới sao cho:
1. CÙNG KIỂU, CÙNG CẤU TRÚC câu và mức độ chi tiết với đáp án đúng (nếu đáp án đúng có hai vế/cặp thì nhiễu cũng có hai vế/cặp).
2. ĐỘ DÀI theo số ký tự: phương án nhiễu 1 khoảng {t1} ký tự, nhiễu 2 khoảng {t2} ký tự, nhiễu 3 khoảng {t3} ký tự (đáp án đúng dài {n} ký tự); sai lệch không quá 10%. Đừng làm nhiễu ngắn hơn bằng cách bỏ chi tiết; cũng đừng nhồi thêm từ vô nghĩa.
3. Hợp lí, hấp dẫn với người chưa đọc kĩ sách (dựa trên khái niệm/số liệu có trong các đoạn: chỉ dùng một đoạn, bỏ điều kiện, đảo nhân quả, tính sai một bước...), nhưng CHẮC CHẮN SAI theo các đoạn trích và khác nhau rõ rệt; không chứa ý của đáp án đúng.
4. Không dùng "tất cả đều đúng/sai", "cả A và B"; không nhắc "đoạn trích".
{extra}
Trả về JSON: {{"distractors": [{{"text": "...", "why_wrong": "..."}}, {{"text": "...", "why_wrong": "..."}}, {{"text": "...", "why_wrong": "..."}}]}}"""

LEN_LO, LEN_HI = 0.8, 1.2


def length_ok(correct: str, distractors: list[str], lo: float = LEN_LO, hi: float = LEN_HI) -> bool:
    n = max(1, len(correct))
    return all(lo <= len(d) / n <= hi or abs(len(d) - n) <= 3 for d in distractors)     # +-20%, or +-3 chars for very short keys


def unique_longest(options: dict[str, str], answer: str) -> bool:
    L = {k: len(v) for k, v in options.items()}
    m = max(L.values())
    return L[answer] == m and list(L.values()).count(m) == 1


def needs_rewrite(c: dict[str, Any]) -> bool:
    return (not length_ok(c["correct_answer"], c["distractors"])) or unique_longest(c["options"], c["answer"])


def rewrite_distractors(c: dict[str, Any], corpus: SpanCorpus, llm, *, attempts: int = 3) -> Optional[dict[str, Any]]:
    """Return a new candidate record (cand_id suffixed `_F`) with length-matched distractors, or None."""
    rng = random.Random(sha1(c["cand_id"] + "len")[:8])
    n = len(c["correct_answer"])
    gold = c["gold_span_ids"]
    blocks = "\n\n".join(span_block(f"E{i + 1}", corpus.get(s)) for i, s in enumerate(gold))
    feedback = ""
    for k in range(attempts):
        ts = [max(4, int(round(n * rng.uniform(*r)))) for r in ((1.02, 1.18), (0.92, 1.15), (0.85, 1.1))]
        cur = "\n".join(f"- {d}" for d in c["distractors"])
        p = _REWRITE_PROMPT.format(blocks=blocks, question=c["question"], correct=c["correct_answer"], current=cur,
                                   t1=ts[0], t2=ts[1], t3=ts[2], n=n, extra=feedback)
        try:
            o, _ = chat_json(llm, [{"role": "system", "content": _REWRITE_SYS}, {"role": "user", "content": p}], {},
                             max_tokens=700, temperature=0.5 if k else 0.3,
                             validate=lambda o: isinstance(o.get("distractors"), list) and len(o["distractors"]) >= 3)
        except Exception:  # noqa: BLE001
            continue
        ds = [str((d.get("text") if isinstance(d, dict) else d) or "").strip() for d in o["distractors"][:3]]
        notes = [str(d.get("why_wrong", "")) if isinstance(d, dict) else "" for d in o["distractors"][:3]]
        if not all(ds) or len(set(x.lower() for x in ds + [c["correct_answer"].lower()])) < 4:
            feedback = "Các phương án phải khác nhau và khác đáp án đúng.\n"
            continue
        longest_ok = max(len(d) for d in ds) >= n          # the key must not be the unique longest option
        if length_ok(c["correct_answer"], ds) and (longest_ok or k == attempts - 1):
            from otter.evaluation.hard_gen import assemble_options
            new = dict(c)
            new["parent_cand_id"] = c["cand_id"]
            new["cand_id"] = c["cand_id"] + "_F"
            new["distractors"], new["distractor_notes"] = ds, notes
            new["options"], new["answer"] = assemble_options(new["cand_id"], c["correct_answer"], ds)
            new["distractor_rewrite"] = {"version": REWRITE_VERSION, "model": llm.model, "attempts": k + 1,
                                         "old_distractors": c["distractors"]}
            return new
        feedback = ("Lần trước độ dài chưa đạt (ít nhất một phương án nhiễu phải dài ít nhất bằng đáp án đúng): " + ", ".join(f"{len(d)} ký tự" for d in ds)
                    + f" (đáp án đúng {n} ký tự). Hãy chỉnh lại đúng độ dài yêu cầu.\n")
    return None


def finalize_copy(c: dict[str, Any]) -> dict[str, Any]:
    """Item that needs no rewriting still enters the final-check stage under its own `_F` id."""
    new = dict(c)
    new["parent_cand_id"] = c["cand_id"]
    new["cand_id"] = c["cand_id"] + "_F"
    new["distractor_rewrite"] = None
    return new


# ------------------------------------------------------------------ selection
def balanced_select(pool: list[dict], *, total_target: int = 340, max_total: int = 400, domain_floor: int = 45,
                    domain_cap: int = 75, hop_cap_frac: float = 0.35, seed: int = 0) -> list[dict]:
    """Pick items: all non-easy first (hard, medium), then easy to reach per-subject floors and the total target;
    within a subject interleave hop types so that no hop type dominates."""
    rng = random.Random(seed)
    rank = {"hard": 0, "medium": 1, "easy": 2, None: 3}
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_dom[c["domain"]].append(c)
    chosen: dict[str, list[dict]] = {}
    for d, cs in by_dom.items():
        rng.shuffle(cs)
        ne = [c for c in cs if c["filters"].get("difficulty") != "easy"]
        ez = [c for c in cs if c["filters"].get("difficulty") == "easy"]
        chosen[d] = ne[:domain_cap]
        chosen[d] += ez[: max(0, min(domain_floor, domain_cap) - len(chosen[d]))]

    def hop_balance(items: list[dict], cap: int):
        # drop surplus of the dominant hop type, easy first
        cnt = Counter(c["hop_type"] for c in items)
        h, k = cnt.most_common(1)[0] if cnt else (None, 0)
        lim = max(8, int(hop_cap_frac * cap))
        if k <= lim:
            return items
        keep, extra = [], []
        for c in sorted(items, key=lambda c: rank[c["filters"].get("difficulty")]):
            if c["hop_type"] == h and sum(1 for x in keep if x["hop_type"] == h) >= lim:
                extra.append(c)
            else:
                keep.append(c)
        return keep
    for d in chosen:
        chosen[d] = hop_balance(chosen[d], domain_cap)
    out = [c for cs in chosen.values() for c in cs]
    # top up with easy items (subjects with the least items first) until the total target
    chosen_ids = {c["cand_id"] for c in out}
    rest = {d: [c for c in by_dom[d] if c["cand_id"] not in chosen_ids] for d in by_dom}
    for d in rest:
        rest[d].sort(key=lambda c: (rank[c["filters"].get("difficulty")], rng.random()))
    while len(out) < min(total_target, max_total):
        cand_doms = [d for d in rest if rest[d] and sum(1 for c in out if c["domain"] == d) < domain_cap]
        if not cand_doms:
            break
        d = min(cand_doms, key=lambda d: sum(1 for c in out if c["domain"] == d))
        out.append(rest[d].pop(0))
    return out


# ------------------------------------------------------------------ 1-hop (single-span, "source location") items
ONE_HOP_VERSION = "otter_hard_1hop_gen_v1"
BLOOM = ["nhận biết", "thông hiểu", "vận dụng"]
_ONE_PROMPT = """Môn: {subject}. Lớp: {grade}.
Dưới đây là MỘT đoạn trích từ sách giáo khoa ({where}).

[S1]
{text}

NHIỆM VỤ: viết MỘT câu hỏi trắc nghiệm 4 phương án kiểm tra LÝ THUYẾT được nêu trong đoạn (định nghĩa, định luật, công thức, điều kiện, \
ngoại lệ, quá trình, đặc điểm, sự kiện...), mức nhận thức: {bloom}. Học sinh phải đọc đúng phần lý thuyết này trong sách mới chắc chắn trả lời được.

YÊU CẦU:
1. Câu hỏi tự đứng được (như đề kiểm tra), KHÔNG nhắc "đoạn trích", "văn bản trên", "S1"; không chép nguyên văn cả câu của đoạn; không hỏi chi tiết vụn vặt (số hình, số trang).
2. Đáp án đúng duy nhất, suy ra chắc chắn từ đoạn trích; không dùng kiến thức ngoài sách. Đáp án đúng không xuất hiện nguyên văn trong câu hỏi.
3. Ba phương án nhiễu hợp lí (sai nhưng nghe đúng với người chưa đọc kĩ), CHẮC CHẮN SAI theo đoạn, cùng dạng, cùng độ dài (±20% số ký tự) và mức chi tiết với đáp án đúng; không dùng "tất cả đều đúng/sai", "cả A và B".
4. Mức "vận dụng": đưa một tình huống/số liệu cụ thể rồi hỏi áp dụng lý thuyết trong đoạn (không cần kiến thức ngoài).
5. Nếu đoạn không đủ nội dung lý thuyết để ra câu hỏi tốt (mục lục, đề bài tập, mô tả hình...), đặt "feasible": false.
{extra}
Trả về JSON: {{"feasible": true|false, "question": "...", "correct_answer": "...", "distractors": [{{"text": "...", "why_wrong": "..."}}, ... đúng 3], "link": "<ý chính của đoạn mà câu hỏi kiểm tra>"}}"""


def one_hop_seeds(corpus: SpanCorpus, *, per_domain: dict[str, int], seed: int = 31, exclude: set[str] = frozenset()) -> list[Seed]:
    """Single-span seeds stratified by subject x grade x pedagogical type (round-robin), one seed per span."""
    rng = random.Random(seed)
    cells: dict[tuple, list[Span]] = defaultdict(list)
    for s in corpus.spans:
        if s.span_id in exclude or not eligible_rule_span(s, 80) or s.pedagogical_type in ("table", "other", "reading_text") and s.domain != "literature":
            continue
        cells[(s.domain, s.grade, s.pedagogical_type)].append(s)
    out = []
    for dom, quota in per_domain.items():
        keys = [k for k in cells if k[0] == dom]
        lists = [rng.sample(cells[k], len(cells[k])) for k in sorted(keys, key=str)]
        rng.shuffle(lists)
        picked = []
        while len(picked) < quota and any(lists):
            for l in lists:
                if l and len(picked) < quota:
                    picked.append(l.pop())
        for s in picked:
            sd = _mk_seed("single_span", [s.span_id], corpus, "single", ["single"], {"ped_label": s.pedagogical_type}, "one")
            sd.version = ONE_HOP_VERSION
            out.append(sd)
    return out


def generate_one_hop(seed: Seed, corpus: SpanCorpus, llm, *, temperature: float = 0.7) -> dict[str, Any]:
    from otter.evaluation.hard_gen import _GEN_SYS, _bool
    s = corpus.get(seed.span_ids[0])
    bloom = BLOOM[int(sha1(seed.seed_id)[:6], 16) % 3]
    where = f"{s.book_title or s.book_id}; " + " › ".join(s.heading_path[-2:])
    extra = LIT_NOTE if seed.domain == "literature" else ""
    if seed.domain in ("physics", "chemistry") and bloom == "vận dụng":
        extra += "\n" + ("Với câu tính toán, thêm trường JSON \"calc\": {\"expression\": \"<biểu thức số học thuần túy>\", \"unit\": \"...\"} cho đúng giá trị số của đáp án đúng.")
    p = _ONE_PROMPT.format(subject=DOMAIN_TO_SUBJECT_VI.get(seed.domain, seed.domain), grade=s.grade, where=where,
                           text=s.text.strip()[:2000], bloom=bloom, extra=extra)
    cand = {"cand_id": "cand_" + sha1(seed.seed_id + ONE_HOP_VERSION + llm.model)[:12], "seed_id": seed.seed_id, "domain": seed.domain,
            "grade": s.grade, "seed_span_ids": seed.span_ids, "strategy": "single_span", "target_hop_type": "single",
            "hop_type": "single", "min_spans": 1, "gold_span_ids": list(seed.span_ids), "bloom_target": bloom,
            "reasoning_steps": [{"span_id": seed.span_ids[0], "step": ""}],
            "generator": {"model": llm.model, "prompt_version": ONE_HOP_VERSION, "temperature": temperature}}
    try:
        o, r = chat_json(llm, [{"role": "system", "content": _GEN_SYS}, {"role": "user", "content": p}], {}, max_tokens=1200,
                         temperature=temperature, validate=lambda o: True)
    except Exception as e:  # noqa: BLE001
        cand.update(feasible=False, error=f"{type(e).__name__}: {e}")
        return cand
    dis = [str((d.get("text") if isinstance(d, dict) else d) or "").strip() for d in o.get("distractors") or []]
    cand.update(feasible=_bool(o.get("feasible")) and bool(str(o.get("question", "")).strip()), question=str(o.get("question", "")).strip(),
                correct_answer=str(o.get("correct_answer", "")).strip(), distractors=dis, link=o.get("link", ""),
                calc=o.get("calc") if isinstance(o.get("calc"), dict) else None, facts=[],
                distractor_notes=[str(d.get("why_wrong", "")) if isinstance(d, dict) else "" for d in o.get("distractors") or []])
    cand["reasoning_steps"] = [{"span_id": seed.span_ids[0], "step": str(o.get("link", ""))}]
    if cand["feasible"] and len(dis) == 3 and cand["correct_answer"]:
        from otter.evaluation.hard_gen import assemble_options
        cand["options"], cand["answer"] = assemble_options(cand["cand_id"], cand["correct_answer"], dis)
    return cand


# ------------------------------------------------------------------ enrichment: Bloom label, pedagogical type, open-ended reference
PED_TYPES_VI = ["định nghĩa", "định luật/nguyên lí", "công thức", "điều kiện", "ngoại lệ", "quá trình", "đặc điểm/phân loại",
                "sự kiện/nhân vật", "ví dụ/bài tập đã giải", "nội dung tác phẩm/phân tích", "khác"]
_ENRICH_PROMPT = """Chỉ dựa vào các ĐOẠN TRÍCH sách giáo khoa dưới đây (không dùng kiến thức ngoài), hoàn thiện thông tin cho câu hỏi trắc nghiệm.

{blocks}

Câu hỏi: {question}
{options}
Đáp án đúng: {answer}

Trả về JSON:
{{"bloom": "nhận biết|thông hiểu|vận dụng",
 "ped_types": {{"E1": "<một trong: {ped}>", ...}},
 "reference_answer": "<câu trả lời ngắn (1-2 câu) tiếng Việt cho câu hỏi dưới dạng TỰ LUẬN, không nhắc đến chữ cái A/B/C/D>",
 "explanation": "<một câu giải thích vì sao, TRÍCH NGUYÊN VĂN một cụm từ liên tục từ đoạn trích đặt trong dấu ngoặc kép>",
 "quotes": {{"E1": "<cụm nguyên văn (8-25 âm tiết) từ đoạn E1 làm cơ sở cho đáp án>", ...}}}}"""


def _squash(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def enrich_item(row: dict, corpus: SpanCorpus, llm) -> dict[str, Any]:
    gold = row["gold_span_ids"]
    labels = [f"E{i + 1}" for i in range(len(gold))]
    p = _ENRICH_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), question=row["question"],
                              options=fmt_options(row["options"]), answer=f"{row['answer']}. {row['options'][row['answer']]}",
                              ped=", ".join(PED_TYPES_VI))
    best = None
    for k in range(2):
        try:
            o, _ = chat_json(llm, [{"role": "user", "content": p}], {}, max_tokens=700, temperature=0.0 if k == 0 else 0.4,
                             validate=lambda o: isinstance(o.get("reference_answer"), str) and o.get("reference_answer").strip())
        except Exception:  # noqa: BLE001
            continue
        q = {l: str(v) for l, v in (o.get("quotes") or {}).items() if isinstance(v, str)}
        ok = {gold[labels.index(l)]: _squash(v) in _squash(corpus.get(gold[labels.index(l)]).text)
              for l, v in q.items() if l in labels and v.strip()}
        res = {"bloom": o.get("bloom") if o.get("bloom") in BLOOM else None,
               "ped_types": {gold[labels.index(l)]: v for l, v in (o.get("ped_types") or {}).items() if l in labels},
               "reference_answer": o["reference_answer"].strip(), "explanation": str(o.get("explanation", "")).strip(),
               "quotes": {gold[labels.index(l)]: v for l, v in q.items() if l in labels},
               "quotes_verbatim": ok, "enrich_model": llm.model}
        res["quotes_ok"] = bool(ok) and all(ok.values())
        if best is None or (res["quotes_ok"] and not best["quotes_ok"]):
            best = res
        if res["quotes_ok"]:
            break
    return best or {"error": "enrich_failed"}


# ------------------------------------------------------------------ cross-subject seeds (ViEduHop-Cross)
CROSS_PAIRS = [("physics", "chemistry"), ("chemistry", "biology"), ("physics", "biology"), ("history", "geography"),
               ("literature", "history"), ("geography", "biology"), ("geography", "physics"), ("geography", "chemistry")]


def cross_seeds(corpus: SpanCorpus, onto: "Onto", db: Path, *, per_pair: int = 140, seed: int = 41, sim_lo: float = 0.5,
                sim_hi: float = 0.9, max_span_uses: int = 2) -> list[Seed]:
    """Span pairs from DIFFERENT subjects: bge-m3 nearest neighbours (span vectors from the ontology DB) re-scored with shared
    ontology concept labels / aliases across subjects."""
    import numpy as np
    rng = random.Random(seed)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        vec = {i: np.frombuffer(b, dtype=np.float32) for i, b in
               c.execute("SELECT item_id, vec FROM embeddings WHERE kind='span' AND model='BAAI/bge-m3'")}
        alias = defaultdict(set)                    # normalised label/alias -> subjects having a concept with it
        for cid, lab, ln, subj in c.execute("SELECT concept_id,label,label_norm,subject FROM concepts"):
            alias[ln].add(subj)
        for cid, al in c.execute("SELECT concept_id, alias_norm FROM concept_aliases") if _has_col(c, "concept_aliases", "alias_norm") else []:
            alias[al].add(cid.split(":")[0])
    cross_terms = {t for t, ss in alias.items() if len(ss) >= 2 and len(t) >= 4}
    elig = [s for s in corpus.spans if eligible_rule_span(s, 100)]
    by = defaultdict(list)
    for s in elig:
        if s.span_id in vec:
            by[s.domain].append(s)
    def mat(ss):
        m = np.stack([vec[s.span_id] for s in ss]).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    uses: Counter = Counter()
    seeds = []
    for a, b in CROSS_PAIRS:
        A, B = by[a], by[b]
        if not A or not B:
            continue
        sim = mat(A) @ mat(B).T
        cand = []
        ii, jj = np.where((sim >= sim_lo) & (sim <= sim_hi))
        tA = [set(tokens(s.text)) for s in A]
        for i, j in zip(ii, jj):
            shared = [t for t in cross_terms if " " not in t and t in tA[i] and t in set(tokens(B[j].text))]
            cand.append((float(sim[i, j]) + 0.08 * min(len(shared), 4), i, j, shared[:4]))
        cand.sort(reverse=True)
        n = 0
        for sc, i, j, shared in cand:
            if n >= per_pair:
                break
            sa, sb = A[i], B[j]
            if uses[sa.span_id] >= max_span_uses or uses[sb.span_id] >= max_span_uses:
                continue
            uses.update([sa.span_id, sb.span_id])
            sd = _mk_seed("cross_subject", [sa.span_id, sb.span_id], corpus, "bridge", ["bridge", "comparison", "causal_chain", "conditional_rule"],
                          {"onto": {"kind": "cross", "subjects": [a, b], "sim": round(sc, 3), "shared_terms": shared}}, "cross")
            sd.version = "otter_hard_seeds_cross_v1"
            seeds.append(sd)
            n += 1
    return seeds


def _has_col(c, table, col):
    try:
        return col in [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
    except Exception:  # noqa: BLE001
        return False
