"""Builder step 4: map local span refs, canonicalise concepts (subject-scoped), build typed items.

Concept canonicalisation never merges across subjects. Inside a subject:
1. exact match on the diacritic-preserving key (NFC + casefold + punctuation stripped);
2. label <-> alias exact matches;
3. optional bge-m3 label similarity >= threshold, only if concept types are compatible, the labels do
   not differ only by diacritics (Vietnamese "bào" vs "bảo"), and they contain the same numbers.
Every merge is recorded (for the build report and the audit sheet).
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from otter.builder.extract import ExtractionResult, XAtom
from otter.schemas.ontology import (Atom, Concept, ConditionalRule, Fact, LearningObjective, PedagogicalLink,
                                    SemanticRelation, Validation)
from otter.schemas.textbook import SourceSpan
from otter.storage.textutil import ascii_fold, lexical_overlap, nfc, norm_key, short_hash, slugify

log = logging.getLogger("otter.builder.normalize")
MAX_CONCEPT_SPANS = 40
_VAR = re.compile(r"^\?\w+$")
_DIGITS = re.compile(r"\d+")
GENERIC_TYPES = {"Concept", "Other"}


@dataclass
class Ontology:
    concepts: dict[str, Concept] = field(default_factory=dict)
    relations: dict[str, SemanticRelation] = field(default_factory=dict)
    facts: dict[str, Fact] = field(default_factory=dict)
    rules: dict[str, ConditionalRule] = field(default_factory=dict)
    objectives: dict[str, LearningObjective] = field(default_factory=dict)
    links: dict[str, PedagogicalLink] = field(default_factory=dict)
    grade: dict[str, Optional[int]] = field(default_factory=dict)
    item_chunks: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    item_books: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    rejected: list[dict] = field(default_factory=list)
    merges: list[dict] = field(default_factory=list)

    def items(self):
        for kind, d in (("concept", self.concepts), ("relation", self.relations), ("fact", self.facts),
                        ("rule", self.rules), ("objective", self.objectives)):
            for iid, obj in d.items():
                yield kind, iid, obj


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class _Mention:
    label: str
    aliases: list[str]
    ctype: str
    definition: Optional[str]
    spans: list[str]
    grade: Optional[int]
    chunk_id: str
    book_id: str


def _map_spans(refs: list[str], local: dict[str, str], chunk_ids: set[str]) -> tuple[list[str], list[str]]:
    ok, bad = [], []
    for r in refs:
        r = (r or "").strip().strip("[]")
        if r in local:
            ok.append(local[r])
        elif r in chunk_ids:
            ok.append(r)
        elif r.upper() in local:
            ok.append(local[r.upper()])
        else:
            bad.append(r)
    return list(dict.fromkeys(ok)), bad


def _min_grade(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _compatible(a_label: str, b_label: str, a_type: str, b_type: str) -> bool:
    if ascii_fold(norm_key(a_label)) == ascii_fold(norm_key(b_label)):
        return False                                   # differ only by diacritics -> different words
    if set(_DIGITS.findall(a_label)) != set(_DIGITS.findall(b_label)):
        return False
    return a_type == b_type or a_type in GENERIC_TYPES or b_type in GENERIC_TYPES


def normalize(results: list[ExtractionResult], spans: dict[str, SourceSpan], *, embedder=None,
              sim_threshold: float = 0.93) -> Ontology:
    onto = Ontology()
    span_text = {k: v.text for k, v in spans.items()}

    def reject(kind, reason, item, chunk_id, detail=None):
        onto.rejected.append({"kind": kind, "reason": reason, "chunk_id": chunk_id, "item": item,
                              **({"detail": detail} if detail else {})})

    # ---------------------------------------------------------- pass 1: concept mentions
    mentions: dict[str, list[_Mention]] = defaultdict(list)
    for res in results:
        ch = res.chunk
        local = ch.local_ids()
        cids = {s.span_id for s in ch.spans}
        for e in res.item_errors:
            reject(e["kind"], e["reason"], e.get("item"), ch.chunk_id, e.get("detail"))
        for xc in res.output.concepts:
            ok, bad = _map_spans(xc.spans, local, cids)
            label = nfc(xc.label).strip()
            if not norm_key(label):
                reject("concept", "empty_label", xc.model_dump(), ch.chunk_id)
                continue
            if not ok:
                reject("concept", "invalid_span_ref", xc.model_dump(), ch.chunk_id, bad)
                continue
            defk = norm_key(xc.definition or "")
            aliases = [nfc(a).strip() for a in xc.aliases
                       if norm_key(a) and len(a) <= 60 and norm_key(a) != defk and norm_key(a) != norm_key(label)]
            mentions[ch.subject].append(_Mention(label, aliases,
                                                 xc.concept_type, (xc.definition or "").strip() or None, ok,
                                                 ch.grade, ch.chunk_id, ch.book_id))

    # ---------------------------------------------------------- pass 2: cluster per subject
    label_maps: dict[str, dict[str, str]] = {}
    for subject, ms in mentions.items():
        uf = _UF()
        key_surface: dict[str, Counter] = defaultdict(Counter)
        key_types: dict[str, Counter] = defaultdict(Counter)
        for m in ms:
            k = norm_key(m.label)
            uf.find(k)
            key_surface[k][m.label] += 1
            key_types[k][m.ctype] += 1
        label_keys = set(key_surface)
        for m in ms:
            k = norm_key(m.label)
            for a in m.aliases:
                ak = norm_key(a)
                if ak != k and ak in label_keys and len(ak) >= 2:
                    if uf.find(ak) != uf.find(k):
                        onto.merges.append({"subject": subject, "a": m.label, "b": a, "method": "alias"})
                    uf.union(k, ak)
        if embedder is not None and len(label_keys) > 1:
            keys = sorted(label_keys)
            surf = [key_surface[k].most_common(1)[0][0] for k in keys]
            types = [key_types[k].most_common(1)[0][0] for k in keys]
            try:
                vecs = embedder.embed(surf)
                for i0 in range(0, len(keys), 2048):
                    sims = vecs[i0:i0 + 2048] @ vecs.T
                    rows, cols = np.nonzero(sims >= sim_threshold)
                    for r, c in zip(rows, cols):
                        i = i0 + int(r)
                        j = int(c)
                        if j <= i:
                            continue
                        if not _compatible(surf[i], surf[j], types[i], types[j]):
                            continue
                        if uf.find(keys[i]) != uf.find(keys[j]):
                            onto.merges.append({"subject": subject, "a": surf[i], "b": surf[j], "method": "embedding",
                                                "sim": round(float(sims[r, c]), 4)})
                            uf.union(keys[i], keys[j])
            except Exception as e:  # embedding server down -> string-only canonicalisation
                log.warning("embedding merge skipped for %s: %s", subject, e)
        clusters: dict[str, list[_Mention]] = defaultdict(list)
        for m in ms:
            clusters[uf.find(norm_key(m.label))].append(m)
        # canonical labels, ids
        slug_count = Counter()
        canon: dict[str, str] = {}
        for root, cm in clusters.items():
            canon[root] = Counter(m.label for m in cm).most_common(1)[0][0]
            slug_count[slugify(canon[root])] += 1
        lmap: dict[str, str] = {}
        for root, cm in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            label = canon[root]
            slug = slugify(label)
            cid = f"{subject}:c:{slug}" if slug_count[slug] == 1 else f"{subject}:c:{slug}_{short_hash(norm_key(label), n=6)}"
            aliases = []
            seen = {norm_key(label)}
            for m in cm:
                for a in [m.label, *m.aliases]:
                    if norm_key(a) not in seen:
                        seen.add(norm_key(a))
                        aliases.append(a)
            span_ids = list(dict.fromkeys(s for m in cm for s in m.spans))
            defs = [(m.definition, m.spans) for m in cm if m.definition]
            definition = None
            if defs:
                definition = max(defs, key=lambda d: (lexical_overlap(d[0], " ".join(span_text.get(s, "") for s in d[1])),
                                                      -abs(len(d[0]) - 200)))[0][:800]
                # prioritise the definition spans in the (capped) evidence list
                dspans = [s for d in defs for s in d[1]]
                span_ids = list(dict.fromkeys(dspans + span_ids))
            ctype = Counter(m.ctype for m in cm).most_common()
            ctype_best = next((t for t, _ in ctype if t not in GENERIC_TYPES), ctype[0][0])
            grade = None
            for m in cm:
                grade = _min_grade(grade, m.grade)
            c = Concept(concept_id=cid, label=label, aliases=aliases[:20], concept_type=ctype_best, subject=subject,
                        grade=grade, definition=definition, source_span_ids=span_ids[:MAX_CONCEPT_SPANS],
                        validation=Validation())
            onto.concepts[cid] = c
            onto.grade[cid] = grade
            for m in cm:
                onto.item_chunks[cid].add(m.chunk_id)
                onto.item_books[cid].add(m.book_id)
            for a in [label, *aliases]:
                lmap.setdefault(norm_key(a), cid)
        label_maps[subject] = lmap

    def resolve(subject: str, label: str, cited: list[str], chunk, create: bool = True) -> Optional[str]:
        k = norm_key(label)
        if not k:
            return None
        lm = label_maps.setdefault(subject, {})
        if k in lm:
            return lm[k]
        if not create:
            return None
        raw_ev = " ".join(span_text.get(s, "") for s in cited)
        ev = norm_key(raw_ev)
        if f" {k} " not in f" {ev} " and (len(k.split()) > 6 or lexical_overlap(label, raw_ev) < 0.75):
            return None
        slug = slugify(label)
        cid = f"{subject}:c:{slug}"
        if cid in onto.concepts:
            cid = f"{cid}_{short_hash(k, n=6)}"
        onto.concepts[cid] = Concept(concept_id=cid, label=nfc(label).strip(), subject=subject, grade=chunk.grade,
                                     source_span_ids=cited[:MAX_CONCEPT_SPANS],
                                     validation=Validation(notes=["auto_created_from_reference"]))
        onto.grade[cid] = chunk.grade
        onto.item_chunks[cid].add(chunk.chunk_id)
        onto.item_books[cid].add(chunk.book_id)
        lm[k] = cid
        return cid

    def register(kind: str, iid: str, chunk) -> None:
        onto.item_chunks[iid].add(chunk.chunk_id)
        onto.item_books[iid].add(chunk.book_id)
        onto.grade[iid] = _min_grade(onto.grade.get(iid), chunk.grade) if iid in onto.grade else chunk.grade

    # ---------------------------------------------------------- pass 3: relations / facts / rules / objectives
    pred_votes: dict[tuple[str, str, bool], Counter] = defaultdict(Counter)   # (subject, text_key, neg) -> preds
    chunk_objectives: dict[str, list[str]] = defaultdict(list)
    chunk_items: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for res in results:
        ch = res.chunk
        local = ch.local_ids()
        cids = {s.span_id for s in ch.spans}
        subj = ch.subject
        for xr in res.output.relations:
            ok, bad = _map_spans(xr.spans, local, cids)
            if not ok:
                reject("relation", "invalid_span_ref", xr.model_dump(), ch.chunk_id, bad)
                continue
            s_id = resolve(subj, xr.subject, ok, ch)
            o_id = resolve(subj, xr.object, ok, ch)
            if not s_id or not o_id:
                reject("relation", "dangling_concept", xr.model_dump(), ch.chunk_id)
                continue
            if s_id == o_id:
                reject("relation", "self_loop", xr.model_dump(), ch.chunk_id)
                continue
            rid = f"{subj}:rel:{short_hash(s_id, xr.relation_type, o_id)}"
            if rid in onto.relations:
                r = onto.relations[rid]
                r.source_span_ids = list(dict.fromkeys(r.source_span_ids + ok))
                if xr.qualifier and not r.qualifier:
                    r.qualifier = xr.qualifier
            else:
                onto.relations[rid] = SemanticRelation(relation_id=rid, subject_id=s_id, relation_type=xr.relation_type,
                                                       object_id=o_id, status=xr.status, qualifier=xr.qualifier,
                                                       source_span_ids=ok)
            register("relation", rid, ch)
            chunk_items[ch.chunk_id].append(("relation", rid))
        for xf in res.output.facts:
            ok, bad = _map_spans(xf.spans, local, cids)
            text = nfc(xf.text).strip()
            if not norm_key(text):
                reject("fact", "empty_text", xf.model_dump(), ch.chunk_id)
                continue
            if not ok:
                reject("fact", "invalid_span_ref", xf.model_dump(), ch.chunk_id, bad)
                continue
            concept_ids = [c for c in (resolve(subj, l, ok, ch) for l in xf.concepts) if c]
            fid = f"{subj}:a:{short_hash(norm_key(text))}"
            if fid in onto.facts:
                f = onto.facts[fid]
                f.source_span_ids = list(dict.fromkeys(f.source_span_ids + ok))
                f.concept_ids = list(dict.fromkeys(f.concept_ids + concept_ids))
            else:
                onto.facts[fid] = Fact(assertion_id=fid, text=text, subject=subj, concept_ids=concept_ids,
                                       status=xf.status, source_span_ids=ok)
            register("fact", fid, ch)
            chunk_items[ch.chunk_id].append(("fact", fid))
        for xr in res.output.rules:
            ok, bad = _map_spans(xr.spans, local, cids)
            if not ok:
                reject("rule", "invalid_span_ref", xr.model_dump(), ch.chunk_id, bad)
                continue

            def atom(x: XAtom) -> Atom:
                args = []
                for a in x.args:
                    a = nfc(a).strip()
                    if _VAR.match(a):
                        args.append(a)
                    elif a:
                        args.append(resolve(subj, a, ok, ch, create=False) or a)
                pred = slugify(x.predicate, 80) if slugify(x.predicate) != "x" else slugify(x.text, 80)
                pred_votes[(subj, norm_key(x.text), x.negated)][pred] += 1
                return Atom(predicate=pred, args=args, text=nfc(x.text).strip(), negated=x.negated)
            prem = [atom(a) for a in xr.premises]
            guards = [atom(a) for a in xr.guards]
            concl = [atom(a) for a in xr.conclusions]
            exc = [atom(a) for a in xr.exceptions]
            rtype, rnotes = xr.rule_type, []
            if rtype == "formula" and not (xr.formula or "").strip():
                rtype, rnotes = "conditional", ["rule_type_coerced_from_formula"]
            variables = {}
            for v in xr.variables:
                cid = resolve(subj, v.meaning, ok, ch, create=False)
                desc = cid or v.meaning
                variables[v.symbol.strip()] = f"{desc} ({v.unit})" if v.unit else desc
            sig = "|".join([subj, rtype,
                            ",".join(sorted(f"{'!' if a.negated else ''}{norm_key(a.text)}" for a in prem)),
                            ",".join(sorted(f"{'!' if a.negated else ''}{norm_key(a.text)}" for a in guards)),
                            ",".join(sorted(f"{'!' if a.negated else ''}{norm_key(a.text)}" for a in concl)),
                            (xr.formula or "").replace(" ", "")])
            rid = f"{subj}:rule:{short_hash(sig)}"
            if rid in onto.rules:
                r = onto.rules[rid]
                r.source_span_ids = list(dict.fromkeys(r.source_span_ids + ok))
            else:
                onto.rules[rid] = ConditionalRule(rule_id=rid, subject=subj, rule_type=rtype, premises=prem,
                                                  guards=guards, conclusions=concl, exceptions=exc,
                                                  formula=((xr.formula or "").strip() or None), variables=variables,
                                                  status=xr.status, source_span_ids=ok,
                                                  validation=Validation(notes=rnotes))
            register("rule", rid, ch)
            chunk_items[ch.chunk_id].append(("rule", rid))
        for xo in res.output.objectives:
            ok, bad = _map_spans(xo.spans, local, cids)
            text = nfc(xo.text).strip()
            if not ok or not norm_key(text):
                reject("objective", "invalid_span_ref" if not ok else "empty_text", xo.model_dump(), ch.chunk_id, bad)
                continue
            oid = f"{subj}:obj:{short_hash(norm_key(text), ch.book_id)}"
            if oid not in onto.objectives:
                onto.objectives[oid] = LearningObjective(objective_id=oid, text=text,
                                                         lesson_section_id=ch.lesson_section_id, subject=subj,
                                                         grade=ch.grade, source_span_ids=ok)
            register("objective", oid, ch)
            chunk_objectives[ch.chunk_id].append(oid)

    # predicate canonicalisation: identical atom text (+ polarity) => majority predicate
    canon_pred = {k: v.most_common(1)[0][0] for k, v in pred_votes.items()}
    for r in onto.rules.values():
        for a in [*r.premises, *r.guards, *r.conclusions, *r.exceptions]:
            a.predicate = canon_pred.get((r.subject, norm_key(a.text), a.negated), a.predicate)

    # pedagogical links: concept/rule taught by an objective of the same chunk if its label occurs in it
    for res in results:
        ch = res.chunk
        objs = chunk_objectives.get(ch.chunk_id, [])
        if not objs:
            continue
        concept_ids = {cid for cid, chs in onto.item_chunks.items() if ch.chunk_id in chs and cid in onto.concepts}
        for oid in objs:
            okey = f" {norm_key(onto.objectives[oid].text)} "
            for cid in concept_ids:
                c = onto.concepts[cid]
                if len(norm_key(c.label)) >= 3 and f" {norm_key(c.label)} " in okey:
                    lid = f"link:{short_hash(cid, 'teaches', oid)}"
                    onto.links[lid] = PedagogicalLink(link_id=lid, source_id=cid, link_type="teaches", target_id=oid,
                                                      source_span_ids=onto.objectives[oid].source_span_ids)
                    for kind, iid in chunk_items.get(ch.chunk_id, []):
                        if kind == "rule":
                            r = onto.rules[iid]
                            if any(cid in a.args for a in [*r.premises, *r.guards, *r.conclusions]) and \
                                    oid not in r.learning_objective_ids:
                                r.learning_objective_ids.append(oid)
    return onto
