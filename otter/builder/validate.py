"""Builder step 5: structural validation (hard) and SEPARATE source-support check.

Structural (deterministic): cited spans must exist in the provenance DB (hard: items left without an
existing span are rejected), dangling concept ids, self-loops, relation domain/range sanity, rule
structure (premises, conclusions, formula variables), duplicates, suspected missing guards.

Support (heuristic + LLM, disclosed as such): lexical overlap of content syllable uni/bi-grams between
the item and its cited spans, plus an entailment judgment by a DIFFERENT model than the extractor
(role "small" by default). Result: supported / partially_supported / unsupported. Unsupported rules
are kept but flagged `excluded_from_strict_proof`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from otter.builder.ingest import existing_span_ids
from otter.builder.normalize import GENERIC_TYPES, Ontology
from otter.llm.interface import LLM, parse_json_loose
from otter.schemas.textbook import SourceSpan
from otter.storage.textutil import lexical_overlap, norm_key
from otter.storage.writer import rule_text

log = logging.getLogger("otter.builder.validate")
VERIFY_VERSION = "verify-v2"

REL_VI = {"is_a": "là một loại/dạng của", "part_of": "là một bộ phận/thành phần của",
          "has_part": "có bộ phận/thành phần là", "has_property": "có tính chất/đặc điểm là",
          "involves": "có liên quan đến (có sự tham gia của)", "affects": "ảnh hưởng đến", "causes": "gây ra / dẫn đến",
          "caused_by": "được gây ra bởi", "occurs_before": "xảy ra trước", "occurs_after": "xảy ra sau",
          "occurs_in": "diễn ra ở/trong", "located_in": "nằm ở/thuộc", "measured_in": "được đo bằng đơn vị",
          "defined_as": "được định nghĩa là", "example_of": "là ví dụ của", "contrasts_with": "khác biệt/đối lập với",
          "synonym_of": "đồng nghĩa với", "authored_by": "là tác phẩm của / do ... sáng tác",
          "belongs_to_period": "thuộc thời kì/giai đoạn", "related_to": "có liên hệ với"}

# relation_type -> (allowed subject types, allowed object types); None = unconstrained. Generic types pass.
DOMAIN_RANGE = {
    "measured_in": (None, {"Unit"}),
    "authored_by": ({"Work", "LiteraryDevice"}, {"Person"}),
    "belongs_to_period": (None, {"Period", "Event"}),
    "located_in": (None, {"Place", "Structure", "Entity", "Organism"}),
    "occurs_in": ({"Process", "Event", "Phenomenon", "State"}, None),
}
GUARD_CUES = ("với điều kiện", "chỉ khi", "trong giới hạn", "trong một giới hạn", "trừ khi", "ngoại trừ",
              "trừ trường hợp", "điều kiện", "chỉ xảy ra khi", "nếu không", "miễn là")

SupportLabel = Literal["supported", "partially_supported", "unsupported"]


class _J(BaseModel):
    id: int
    label: SupportLabel


class _Judgments(BaseModel):
    judgments: list[_J]


VERIFY_SYSTEM = """Bạn là người kiểm định độc lập cho một ontology trích xuất từ sách giáo khoa. Với mỗi mệnh đề, hãy đánh giá các đoạn văn được trích dẫn (mã E1, E2, ...) có hỗ trợ mệnh đề đó hay không, CHỈ dựa vào nội dung các đoạn văn (không dùng kiến thức bên ngoài):
- "supported": đoạn văn nêu rõ mệnh đề, hoặc nêu một ý tương đương về nghĩa (kể cả đủ các điều kiện).
- "partially_supported": đoạn văn chỉ hỗ trợ một phần (thiếu điều kiện, thiếu chi tiết, khái quát hoá quá mức, hoặc có thêm thông tin không có trong đoạn).
- "unsupported": đoạn văn không nêu, hoặc mâu thuẫn với mệnh đề.
Trả về JSON: {"judgments": [{"id": <số thứ tự mệnh đề>, "label": "..."}]} cho TẤT CẢ các mệnh đề."""


def claim_text(kind: str, obj, onto: Ontology) -> str:
    if kind == "concept":
        return f"{obj.label}: {obj.definition}" if obj.definition else obj.label
    if kind == "relation":
        s = onto.concepts.get(obj.subject_id)
        o = onto.concepts.get(obj.object_id)
        t = f"{s.label if s else obj.subject_id} {REL_VI.get(obj.relation_type, obj.relation_type)} " \
            f"{o.label if o else obj.object_id}"
        return t + (f" ({obj.qualifier})" if obj.qualifier else "")
    if kind == "fact" or kind == "objective":
        return obj.text
    if kind == "rule":
        return rule_text(obj)
    raise ValueError(kind)


# ----------------------------------------------------------------------- structural
def structural_checks(onto: Ontology, db_path: Path) -> None:
    all_ids = [s for _, _, o in onto.items() for s in o.source_span_ids] + \
              [s for l in onto.links.values() for s in l.source_span_ids]
    exists = existing_span_ids(db_path, all_ids)

    def drop(kind, d, iid, reason, obj):
        onto.rejected.append({"kind": kind, "reason": reason, "chunk_id": ",".join(sorted(onto.item_chunks.get(iid, []))),
                              "item": json.loads(obj.model_dump_json())})
        d.pop(iid, None)

    tables = {"concept": onto.concepts, "relation": onto.relations, "fact": onto.facts, "rule": onto.rules,
              "objective": onto.objectives}
    for kind, d in tables.items():
        for iid in list(d):
            obj = d[iid]
            good = [s for s in obj.source_span_ids if s in exists]
            if len(good) != len(obj.source_span_ids):
                if hasattr(obj, "validation"):
                    obj.validation.notes.append(f"dropped_missing_spans:{len(obj.source_span_ids) - len(good)}")
                obj.source_span_ids = good
            if not good:
                drop(kind, d, iid, "span_not_in_db", obj)
                continue
            if hasattr(obj, "validation"):
                obj.validation.source_ids_exist = True
                obj.validation.schema_valid = True
    # dangling + domain/range
    for rid in list(onto.relations):
        r = onto.relations[rid]
        s, o = onto.concepts.get(r.subject_id), onto.concepts.get(r.object_id)
        if s is None or o is None:
            drop("relation", onto.relations, rid, "dangling_concept", r)
            continue
        dom, rng = DOMAIN_RANGE.get(r.relation_type, (None, None))
        bad_dom = dom is not None and s.concept_type not in GENERIC_TYPES and s.concept_type not in dom
        bad_rng = rng is not None and o.concept_type not in GENERIC_TYPES and o.concept_type not in rng
        if bad_dom and bad_rng:
            drop("relation", onto.relations, rid, "domain_range_violation", r)
        elif bad_dom or bad_rng:
            r.validation.notes.append("domain_range_warning")
    for f in onto.facts.values():
        f.concept_ids = [c for c in f.concept_ids if c in onto.concepts]
    for rid in list(onto.rules):
        r = onto.rules[rid]
        if not r.premises or not r.conclusions:
            drop("rule", onto.rules, rid, "rule_incomplete_structure", r)
            continue
        for a in [*r.premises, *r.guards, *r.conclusions, *r.exceptions]:
            a.args = [x for x in a.args if ":c:" not in x or x in onto.concepts]
        if r.rule_type == "formula" and not r.variables:
            r.validation.notes.append("formula_missing_variables")
        prem_keys = {norm_key(a.text) for a in r.premises}
        if any(norm_key(a.text) in prem_keys for a in r.conclusions):
            drop("rule", onto.rules, rid, "circular_rule", r)
            continue
        if r.status == "hypothesized_relation":
            r.validation.notes.append("hypothesized")
    for lid in list(onto.links):
        l = onto.links[lid]
        l.source_span_ids = [s for s in l.source_span_ids if s in exists]
        known = set(onto.concepts) | set(onto.rules) | set(onto.objectives)
        if not l.source_span_ids or l.source_id not in known or l.target_id not in known:
            onto.links.pop(lid)


def missing_guard_check(onto: Ontology, spans: dict[str, SourceSpan]) -> None:
    for r in onto.rules.values():
        if r.guards:
            continue
        ev = " ".join(spans[s].text for s in r.source_span_ids if s in spans).casefold()
        cues = [c for c in GUARD_CUES if c in ev]
        if not cues:
            continue
        atoms = " ".join(a.text for a in [*r.premises, *r.conclusions]).casefold()
        if not any(c in atoms for c in cues):
            r.validation.notes.append("possible_missing_guard")


# ----------------------------------------------------------------------- support
def _combine(lex: float, llm_label: Optional[str]) -> tuple[str, float, list[str]]:
    notes = []
    if llm_label is None:
        notes.append("support_lexical_only")
        label = "supported" if lex >= 0.6 else "partially_supported" if lex >= 0.3 else "unsupported"
        return label, round(lex, 4), notes
    label = llm_label
    if label == "supported" and lex < 0.15:
        label = "partially_supported"
        notes.append("low_lexical_overlap")
    score = 0.5 * lex + 0.5 * {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}[llm_label]
    return label, round(score, 4), notes


class SupportChecker:
    def __init__(self, verifier: Optional[LLM], cache_dir: Optional[Path] = None, *, batch_size: int = 12,
                 max_span_chars: int = 1500):
        self.llm, self.cache_dir, self.batch_size, self.max_span_chars = verifier, cache_dir, batch_size, max_span_chars
        self.schema = _Judgments.model_json_schema()
        self.stats = {"llm_calls": 0, "cached_batches": 0, "failed_batches": 0, "prompt_tokens": 0,
                      "completion_tokens": 0}

    def _judge(self, batch: list[tuple[str, list[str]]], spans: dict[str, SourceSpan]) -> list[Optional[str]]:
        ev_ids = list(dict.fromkeys(s for _, ss in batch for s in ss))
        lab = {s: f"E{i + 1}" for i, s in enumerate(ev_ids)}
        lines = ["ĐOẠN VĂN:"]
        for s in ev_ids:
            lines.append(f"[{lab[s]}] {spans[s].text.strip()[: self.max_span_chars]}")
        lines.append("\nMỆNH ĐỀ:")
        for i, (claim, ss) in enumerate(batch):
            lines.append(f"{i + 1}. (trích dẫn: {', '.join(lab[s] for s in ss)}) {claim}")
        user = "\n".join(lines)
        model = getattr(self.llm, "model", "")
        h = hashlib.sha256(f"{VERIFY_VERSION}\x1f{model}\x1f{user}".encode("utf-8")).hexdigest()
        cp = self.cache_dir / h[:2] / f"{h}.json" if self.cache_dir else None
        if cp and cp.exists():
            try:
                labels = json.loads(cp.read_text(encoding="utf-8"))["labels"]
                self.stats["cached_batches"] += 1
                return labels
            except Exception:
                pass
        msgs = [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]
        for attempt in range(5):
            try:
                r = self.llm.chat(msgs, temperature=0.0, max_tokens=40 * len(batch) + 64, json_schema=self.schema)
                self.stats["llm_calls"] += 1
                self.stats["prompt_tokens"] += r.prompt_tokens
                self.stats["completion_tokens"] += r.completion_tokens
                obj = _Judgments.model_validate(parse_json_loose(r.text))
                labels: list[Optional[str]] = [None] * len(batch)
                for j in obj.judgments:
                    if 1 <= j.id <= len(batch):
                        labels[j.id - 1] = j.label
                if cp:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_text(json.dumps({"labels": labels, "model": model}, ensure_ascii=False), encoding="utf-8")
                return labels
            except Exception as e:
                log.debug("verify batch failed (%s): %s", attempt, e)
                time.sleep(min(60, 5 * (attempt + 1)) if not isinstance(e, ValueError) else 0.5)
        self.stats["failed_batches"] += 1
        return [None] * len(batch)

    def run(self, onto: Ontology, spans: dict[str, SourceSpan], workers: int = 32) -> None:
        todo: dict[str, list[tuple[str, str, object, str]]] = defaultdict(list)   # group -> items
        for kind, iid, obj in onto.items():
            ev = " ".join(spans[s].text for s in obj.source_span_ids if s in spans)
            claim = claim_text(kind, obj, onto)
            if kind == "objective":
                continue                                    # LearningObjective has no validation field
            if kind == "concept" and not obj.definition:
                keys = [norm_key(a) for a in [obj.label, *obj.aliases]]
                evk = f" {norm_key(ev)} "
                hit = any(k and f" {k} " in evk for k in keys)
                obj.validation.source_support = "supported" if hit else "unsupported"
                obj.validation.support_score = 1.0 if hit else 0.0
                obj.validation.notes.append("support_term_occurrence")
                continue
            lex = lexical_overlap(claim, ev)
            obj.validation.support_score = round(lex, 4)
            group = sorted(onto.item_chunks.get(iid) or {"_"})[0]
            todo[group].append((kind, iid, obj, claim))
        batches = []
        for group, items in todo.items():
            for i in range(0, len(items), self.batch_size):
                batches.append(items[i:i + self.batch_size])
        log.info("support check: %d items in %d batches (verifier=%s)", sum(len(b) for b in batches), len(batches),
                 getattr(self.llm, "model", None))
        results: dict[int, list[Optional[str]]] = {}
        if self.llm is not None and batches:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = {ex.submit(self._judge, [(c, o.source_span_ids[:6]) for _, _, o, c in b], spans): i
                        for i, b in enumerate(batches)}
                for n, f in enumerate(as_completed(futs), 1):
                    results[futs[f]] = f.result()
                    if n % 100 == 0 or n == len(batches):
                        log.info("verify %d/%d batches (%.1f batches/min)", n, len(batches),
                                 60 * n / max(time.time() - t0, 1e-6))
        for i, b in enumerate(batches):
            labels = results.get(i) or [None] * len(b)
            for (kind, iid, obj, claim), lab in zip(b, labels):
                lex = obj.validation.support_score or 0.0
                label, score, notes = _combine(lex, lab)
                obj.validation.source_support = label
                obj.validation.support_score = score
                obj.validation.notes.extend(notes)
                if kind == "rule" and label == "unsupported":
                    obj.validation.notes.append("excluded_from_strict_proof")
