#!/usr/bin/env python3
"""Build the OTTER ontology explorer (static, offline) from the frozen ontology.

Command (from the repository root, no extra packages, stdlib only):

    .venv/bin/python scripts/release/viz_build_explorer.py

Reads (read-only):  data/ontology/ontology.sqlite3   (opened with mode=ro)
                    data/textbooks/parsed/nxbgd_*/{spans,sections}.jsonl   (source pointers + verbatim check)
                    paper/sections/method.tex          (the paper's ontology counts)
Writes (only):      release/github/OTTER/viz/ontology_explorer.html  (self-contained, data inlined)
                    release/github/OTTER/viz/data/*.json
                    release/github/OTTER/viz/README.md

Copyright guard: no textbook passage is written. Statement texts (facts, rule atoms, formulas, objective lines,
concept labels) are written only if they have at most 25 words and share no run of 8 consecutive words with any
of their evidence spans; otherwise only the id/type/pointers are kept. Only items whose DB label is 'supported'
are used (objectives and 'teaches' links carry no support label in the DB and are marked 'not labelled').
Optional machine glosses (English, flagged as unchecked) are read from viz/data/gloss_en.json when it exists.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import random
import re
import sqlite3
import sys
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.path.join(ROOT, "data/ontology/ontology.sqlite3")
PARSED = os.path.join(ROOT, "data/textbooks/parsed")
OUT = os.path.join(ROOT, "release/github/OTTER/viz")
MAX_NODES = 150
MAX_WORDS = 25
RUN = 8  # a statement fails if it shares RUN consecutive words with an evidence span
MAX_PTRS = 8

SUBJECTS = ["physics", "chemistry", "biology", "history", "geography", "literature"]
SUBJ_GLOSS = {"physics": ("Vật lí", "Physics"), "chemistry": ("Hóa học", "Chemistry"), "biology": ("Sinh học", "Biology"),
              "history": ("Lịch sử", "History"), "geography": ("Địa lí", "Geography"), "literature": ("Ngữ văn", "Literature")}
# the extra demo lesson: the lesson cited by the question shown in the video (VIEDUHOP-C2-PHY-017, gold lesson_key)
DEMO_LESSON = "nxbgd_g10_physics_d0daa774:ch03:l14"

_tok = re.compile(r"\w+", re.UNICODE)


def words(t: str) -> list[str]:
    return _tok.findall(unicodedata.normalize("NFC", t or "").lower())


def grams(ws: list[str], n: int = RUN) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


class Checker:
    """Verbatim / length check of statements against evidence spans. Keeps counters."""

    def __init__(self, span_text, all_grams=None):
        self.span_text = span_text
        self.all_grams = all_grams  # 8-grams of EVERY span of the corpus (stricter than the evidence spans alone)
        self.cache = {}
        self.counts = collections.defaultdict(lambda: [0, 0])  # kind -> [passed, failed]
        self.fail_reason = collections.Counter()

    def span_grams(self, sid):
        if sid not in self.cache:
            self.cache[sid] = grams(words(self.span_text.get(sid, "")))
        return self.cache[sid]

    def ok(self, kind: str, text, span_ids) -> bool:
        if text is None or not str(text).strip():
            return False
        ws = words(str(text))
        good = True
        if len(ws) > MAX_WORDS:
            good = False
            self.fail_reason[kind + ":too_long"] += 1
        else:
            g = grams(ws)
            if g and (any(g & self.span_grams(s) for s in span_ids) or (self.all_grams is not None and g & self.all_grams)):
                good = False
                self.fail_reason[kind + ":verbatim_run"] += 1
        self.counts[kind][0 if good else 1] += 1
        return good


def load_provenance():
    spans, sections, books = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(PARSED, "nxbgd_*", "spans.jsonl"))):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            spans[d["span_id"]] = {"book": d["book_id"], "lesson": d["lesson_id"], "page": d["printed_page_label"],
                                   "verified": bool(d["printed_page_verified"]), "kind": d["span_kind"], "text": d["text"]}
    for f in sorted(glob.glob(os.path.join(PARSED, "nxbgd_*", "sections.jsonl"))):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            sections[d["section_id"]] = d
            if d["level"] == "book":
                books[d["book_id"]] = d["title"]
    return spans, sections, books


def lesson_heading(sections, lid):
    s = sections.get(lid)
    if not s:
        return None
    return ((s.get("ordinal") or "") + " " + (s.get("title") or "")).strip()


def read_paper_counts():
    txt = open(os.path.join(ROOT, "paper/sections/method.tex"), encoding="utf-8").read()
    m = re.search(r"The build yields ([\d,]+) concepts, ([\d,]+) relations, ([\d,]+) facts, ([\d,]+) rules \(([\d,]+) strict-executable\), "
                  r"([\d,]+) objectives and ([\d,]+) evidence links, with ([\d,]+) rejected", txt)
    if not m:
        raise SystemExit("could not parse the paper counts in method.tex")
    k = ["concepts", "relations", "facts", "rules", "rules_strict_executable", "objectives", "evidence_links", "rejected"]
    return {a: int(b.replace(",", "")) for a, b in zip(k, m.groups())}


def safe_args(args, chk, kind, span_ids):
    """Atom args are normally concept ids; free-text args are shown only if they pass the check."""
    out = []
    for a in args:
        a = str(a)
        if re.fullmatch(r"[a-z]+:[a-z]+:[\w\-]+", a) or len(a) <= 3:
            out.append(a)
        else:
            out.append(a if chk.ok(kind, a, span_ids) else None)
    return out


def fr_layout(nodes, edges, seed=7, iters=320):
    """Deterministic Fruchterman-Reingold layout in a 1000x700 box (pure python)."""
    n = len(nodes)
    rnd = random.Random(seed)
    idx = {nid: i for i, nid in enumerate(nodes)}
    pos = [[rnd.uniform(-1, 1) * 300 + 500, rnd.uniform(-1, 1) * 200 + 350] for _ in range(n)]
    E = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx and a != b]
    W, H = 1000.0, 700.0
    k = math.sqrt(W * H / max(n, 1)) * 0.85
    t = W / 8
    for it in range(iters):
        disp = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            xi, yi = pos[i]
            for j in range(i + 1, n):
                dx, dy = xi - pos[j][0], yi - pos[j][1]
                d2 = dx * dx + dy * dy + 0.01
                f = k * k / d2
                disp[i][0] += dx * f; disp[i][1] += dy * f
                disp[j][0] -= dx * f; disp[j][1] -= dy * f
        for a, b in E:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.sqrt(dx * dx + dy * dy) + 0.01
            f = d / k
            disp[a][0] -= dx * f; disp[a][1] -= dy * f
            disp[b][0] += dx * f; disp[b][1] += dy * f
        for i in range(n):  # weak gravity keeps components together
            disp[i][0] += (500 - pos[i][0]) * 0.02
            disp[i][1] += (350 - pos[i][1]) * 0.02
            dl = math.sqrt(disp[i][0] ** 2 + disp[i][1] ** 2) + 0.01
            m = min(dl, t)
            pos[i][0] += disp[i][0] / dl * m
            pos[i][1] += disp[i][1] / dl * m
        t = max(t * 0.985, 1.0)
    xs = [p[0] for p in pos]; ys = [p[1] for p in pos]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = 900 / max(x1 - x0, 1); sy = 600 / max(y1 - y0, 1)
    s = min(sx, sy)
    ox = 500 - (x0 + x1) / 2 * s; oy = 350 - (y0 + y1) / 2 * s
    return {nid: (round(pos[i][0] * s + ox, 1), round(pos[i][1] * s + oy, 1)) for nid, i in idx.items()}


def main():
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    spans, sections, books = load_provenance()
    span_text = {k: v["text"] for k, v in spans.items()}
    all_grams = set()
    for t in span_text.values():
        all_grams |= grams(words(t))
    chk = Checker(span_text, all_grams)
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)

    # ---------- whole-ontology statistics (read from the DB) ----------
    q = lambda s, *a: con.execute(s, a).fetchall()
    stats = {"db": "data/ontology/ontology.sqlite3 (v1, frozen)", "build_id": q("select build_id from build_versions")[0][0]}
    stats["counts"] = {
        "concepts": q("select count(*) from concepts")[0][0], "relations": q("select count(*) from semantic_relations")[0][0],
        "facts": q("select count(*) from facts")[0][0], "rules": q("select count(*) from rules")[0][0],
        "rules_strict_executable": q("select count(*) from rules where strict_executable=1")[0][0],
        "objectives": q("select count(*) from objectives")[0][0], "evidence_links": q("select count(*) from evidence_links")[0][0],
        "rejected": q("select count(*) from rejected_items")[0][0], "pedagogical_links": q("select count(*) from pedagogical_links")[0][0],
        "spans_in_parsed_books": len(spans), "books": len(books)}
    stats["support"] = {t: dict(q(f"select source_support,count(*) from {t} group by 1")) for t in
                        ("concepts", "semantic_relations", "facts", "rules")}
    stats["rules_by_type"] = [{"rule_type": a, "n": b, "strict_executable": c} for a, b, c in
                              q("select rule_type,count(*),sum(strict_executable) from rules group by 1 order by 2 desc")]
    stats["strict_executable_support"] = dict(q("select source_support,count(*) from rules where strict_executable=1 group by 1"))
    stats["evidence_by_kind"] = dict(q("select item_kind,count(*) from evidence_links group by 1"))
    stats["rejected_by_kind"] = dict(q("select item_kind,count(*) from rejected_items group by 1"))
    stats["rejected_by_reason"] = dict(q("select reason,count(*) from rejected_items group by 1 order by 2 desc"))
    stats["by_subject"] = {s: {"concepts": q("select count(*) from concepts where subject=?", s)[0][0],
                               "facts": q("select count(*) from facts where subject=?", s)[0][0],
                               "rules": q("select count(*) from rules where subject=?", s)[0][0],
                               "objectives": q("select count(*) from objectives where subject=?", s)[0][0]} for s in SUBJECTS}
    stats["relation_types_supported"] = dict(q("select relation_type,count(*) from semantic_relations "
                                               "where source_support='supported' group by 1 order by 2 desc"))
    paper = read_paper_counts()
    stats["paper"] = paper
    cmp_rows = []
    for k in ("concepts", "relations", "facts", "rules", "rules_strict_executable", "objectives", "evidence_links", "rejected"):
        cmp_rows.append({"item": k, "paper": paper[k], "db": stats["counts"][k], "equal": paper[k] == stats["counts"][k]})
    stats["comparison"] = cmp_rows

    # ---------- load supported items ----------
    ev = collections.defaultdict(list)
    for k, i, s in con.execute("select item_kind,item_id,span_id from evidence_links order by item_kind,item_id,span_id"):
        ev[(k, i)].append(s)
    items = {}  # id -> dict
    for cid, label, ctype, subj, grade, js in con.execute(
            "select concept_id,label,concept_type,subject,grade,json from concepts where source_support='supported'"):
        d = json.loads(js)
        items[cid] = {"id": cid, "k": "concept", "label": label, "ctype": ctype, "subject": subj, "grade": grade,
                      "support": "supported", "score": d["validation"].get("support_score"), "spans": d["source_span_ids"]}
    for aid, text, subj, grade, status, js in con.execute(
            "select assertion_id,text,subject,grade,status,json from facts where source_support='supported'"):
        d = json.loads(js)
        items[aid] = {"id": aid, "k": "fact", "text": text, "subject": subj, "grade": grade, "status": status, "support": "supported",
                      "score": d["validation"].get("support_score"), "spans": d["source_span_ids"], "concept_ids": d.get("concept_ids", [])}
    for rid, subj, grade, rtype, status, formula, text, strict, js in con.execute(
            "select rule_id,subject,grade,rule_type,status,formula,text,strict_executable,json from rules where source_support='supported'"):
        d = json.loads(js)
        items[rid] = {"id": rid, "k": "rule", "text": text, "subject": subj, "grade": grade, "rule_type": rtype, "status": status,
                      "formula": formula, "strict": bool(strict), "support": "supported", "score": d["validation"].get("support_score"),
                      "spans": d["source_span_ids"], "premises": d["premises"], "guards": d["guards"], "conclusions": d["conclusions"],
                      "exceptions": d["exceptions"], "variables": d.get("variables") or {}}
    for oid, text, subj, grade, js in con.execute("select objective_id,text,subject,grade,json from objectives"):
        d = json.loads(js)
        items[oid] = {"id": oid, "k": "objective", "text": text, "subject": subj, "grade": grade, "support": "not_labelled",
                      "spans": d["source_span_ids"]}
    rels = {}
    for rid, sid, rt, oid, st, subj, js in con.execute(
            "select relation_id,subject_id,relation_type,object_id,status,subject,json from semantic_relations where source_support='supported'"):
        d = json.loads(js)
        rels[rid] = {"id": rid, "s": sid, "t": oid, "rt": rt, "status": st, "subject": subj, "spans": d["source_span_ids"],
                     "score": d["validation"].get("support_score")}
    fact_c = collections.defaultdict(list)
    for a, c in con.execute("select assertion_id,concept_id from fact_concepts order by 1,2"):
        fact_c[a].append(c)
    rule_c = collections.defaultdict(list)
    for a, c in con.execute("select rule_id,concept_id from rule_concepts order by 1,2"):
        rule_c[a].append(c)
    teach = {}
    for lid, s, t, js in con.execute("select link_id,source_id,target_id,json from pedagogical_links where link_type='teaches'"):
        teach[lid] = {"id": lid, "s": s, "t": t, "spans": json.loads(js)["source_span_ids"]}
    con.close()

    # whole-DB verbatim check (statements of supported facts / rules / objectives) for the report
    whole = Checker(span_text, all_grams)
    for it in items.values():
        if it["k"] in ("fact", "rule", "objective"):
            whole.ok(it["k"], it["text"], it["spans"])
    stats["verbatim_check_whole_supported"] = {k: {"passed": v[0], "failed": v[1]} for k, v in whole.counts.items()}
    stats["verbatim_check_rule"] = {"max_words": MAX_WORDS, "max_shared_run": RUN - 1,
                                    "statement": "fails if > 25 words or shares a run of >= 8 consecutive words with any evidence span"}

    # ---------- per-lesson graphs ----------
    lesson_nodes = collections.defaultdict(set)
    for iid, it in items.items():
        for s in it["spans"]:
            L = spans.get(s, {}).get("lesson")
            if L:
                lesson_nodes[L].add(iid)
    lesson_edges = collections.defaultdict(list)

    def add_edges(L, nodes):
        out = []
        for r in rels.values():
            if r["s"] in nodes and r["t"] in nodes:
                out.append({"id": r["id"], "s": r["s"], "t": r["t"], "k": "relation", "rt": r["rt"], "spans": r["spans"], "support": "supported"})
        for nid in nodes:
            it = items[nid]
            if it["k"] == "fact":
                for c in fact_c.get(nid, []):
                    if c in nodes:
                        out.append({"id": nid + "->" + c, "s": nid, "t": c, "k": "mentions"})
            elif it["k"] == "rule":
                for c in rule_c.get(nid, []):
                    if c in nodes:
                        out.append({"id": nid + "->" + c, "s": nid, "t": c, "k": "uses"})
        for l in teach.values():
            if l["s"] in nodes and l["t"] in nodes:
                out.append({"id": l["id"], "s": l["s"], "t": l["t"], "k": "teaches", "spans": l["spans"], "support": "not_labelled"})
        return out

    def components(nodes, edges):
        par = {x: x for x in nodes}

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for e in edges:
            par[f(e["s"])] = f(e["t"])
        comp = collections.defaultdict(list)
        for x in sorted(nodes):
            comp[f(x)].append(x)
        return sorted(comp.values(), key=lambda c: (-len(c), c[0]))

    cand = {}
    for L, nodes in lesson_nodes.items():
        ed = add_edges(L, nodes)
        comps = components(nodes, ed)
        cand[L] = {"lesson_id": L, "nodes_in_lesson": len(nodes), "largest_component": len(comps[0]), "edges": ed, "comps": comps}
    subj_of = lambda L: L.split("_")[2]
    chosen = {}
    selection = {"rule": "per subject: among lessons whose largest connected component of the supported subgraph has <= %d nodes, "
                         "take the lesson with the largest such component (ties: smaller lesson_id). Only that component is drawn. "
                         "One extra 'demo' lesson is added: the gold lesson of VIEDUHOP-C2-PHY-017 (used in the video)." % MAX_NODES,
                 "subjects": {}}
    for s in SUBJECTS:
        rows = sorted([c for L, c in cand.items() if subj_of(L) == s], key=lambda c: (-c["largest_component"], c["lesson_id"]))
        elig = [c for c in rows if c["largest_component"] <= MAX_NODES]
        chosen[s] = elig[0]["lesson_id"]
        selection["subjects"][s] = {
            "chosen": elig[0]["lesson_id"], "lessons_with_items": len(rows),
            "top_candidates": [{"lesson_id": c["lesson_id"], "nodes_in_lesson": c["nodes_in_lesson"],
                                "largest_component": c["largest_component"], "eligible": c["largest_component"] <= MAX_NODES} for c in rows[:5]]}
    gloss_path = os.path.join(OUT, "data/gloss_en.json")
    gloss = json.load(open(gloss_path, encoding="utf-8")) if os.path.exists(gloss_path) else {}
    lesson_gloss_path = os.path.join(ROOT, "scripts/release/viz_lesson_gloss.json")
    lgloss = json.load(open(lesson_gloss_path, encoding="utf-8")) if os.path.exists(lesson_gloss_path) else {}

    def pointer(sid):
        sp = spans.get(sid)
        if not sp:
            return {"span": sid}
        return {"span": sid, "book": books.get(sp["book"], sp["book"]), "book_id": sp["book"],
                "lesson": lesson_heading(sections, sp["lesson"]) if sp["lesson"] else None,
                "page": sp["page"], "page_verified": sp["verified"], "kind": sp["kind"]}

    lessons_out = []
    order = [(s, chosen[s], "selected") for s in SUBJECTS] + [("physics", DEMO_LESSON, "demo")]
    for s, L, role in order:
        c = cand[L]
        comp = set(c["comps"][0]) if role == "selected" else set(c["comps"][0])
        if len(comp) > MAX_NODES:
            raise SystemExit("demo lesson component too large")
        nodes = sorted(comp)
        edges = [e for e in c["edges"] if e["s"] in comp and e["t"] in comp]
        # dedupe edges
        seen, ed2 = set(), []
        for e in edges:
            key = (e["s"], e["t"], e["k"], e.get("rt"))
            if key not in seen:
                seen.add(key); ed2.append(e)
        edges = ed2
        pos = fr_layout(nodes, [(e["s"], e["t"]) for e in edges])
        lchk = Checker(span_text, all_grams)
        pointers, nout = {}, []
        for nid in nodes:
            it = items[nid]
            sp_in = [x for x in it["spans"] if spans.get(x, {}).get("lesson") == L] + [x for x in it["spans"] if spans.get(x, {}).get("lesson") != L]
            for x in sp_in[:MAX_PTRS]:
                pointers.setdefault(x, pointer(x))
            n = {"id": nid, "k": it["k"], "x": pos[nid][0], "y": pos[nid][1], "support": it["support"],
                 "n_spans": len(it["spans"]), "spans": sp_in[:MAX_PTRS]}
            if it.get("score") is not None:
                n["score"] = it["score"]
            if it["k"] == "concept":
                n["ctype"] = it["ctype"]
                n["label"] = it["label"] if lchk.ok("concept_label", it["label"], it["spans"]) else None
                if n["label"] and gloss.get(nid):
                    n["gloss"] = gloss[nid]
            elif it["k"] == "fact":
                n["status"] = it["status"]
                n["text"] = it["text"] if lchk.ok("fact", it["text"], it["spans"]) else None
            elif it["k"] == "objective":
                n["text"] = it["text"] if lchk.ok("objective", it["text"], it["spans"]) else None
            else:  # rule
                r = {"rule_type": it["rule_type"], "status": it["status"], "strict": it["strict"]}
                n["text"] = it["text"] if lchk.ok("rule", it["text"], it["spans"]) else None
                for part in ("premises", "guards", "conclusions", "exceptions"):
                    r[part] = [{"pred": a["predicate"], "args": safe_args(a["args"], lchk, "rule_arg", it["spans"]), "neg": bool(a.get("negated")),
                                "text": a["text"] if lchk.ok("rule_atom", a["text"], it["spans"]) else None} for a in it[part]]
                if it["formula"]:
                    r["formula"] = it["formula"] if lchk.ok("rule_formula", it["formula"], it["spans"]) else None
                    r["has_formula"] = True
                r["variables"] = {k: (v if (v.startswith(("physics:", "chemistry:", "biology:", "history:", "geography:", "literature:"))
                                           or lchk.ok("rule_variable", v, it["spans"])) else None) for k, v in it["variables"].items()}
                n["rule"] = r
            nout.append(n)
        for e in edges:
            for x in e.get("spans", [])[:MAX_PTRS]:
                pointers.setdefault(x, pointer(x))
            if "spans" in e:
                e["n_spans"] = len(e["spans"]); e["spans"] = e["spans"][:MAX_PTRS]
        bid = L.split(":")[0]
        spn = subj_of(L)
        k_counts = collections.Counter(n["k"] for n in nout)
        lessons_out.append({
            "key": L, "role": role, "subject": spn, "subject_vi": SUBJ_GLOSS[spn][0], "subject_en": SUBJ_GLOSS[spn][1],
            "book": books.get(bid, bid), "book_id": bid, "grade": int(L.split("_")[1][1:]),
            "heading": lesson_heading(sections, L), "heading_en": lgloss.get(L),
            "nodes_in_lesson": c["nodes_in_lesson"], "largest_component": c["largest_component"],
            "n_nodes": len(nout), "n_edges": len(edges), "n_by_type": dict(k_counts),
            "n_strict_rules": sum(1 for n in nout if n["k"] == "rule" and n["rule"]["strict"]),
            "components_sizes_top5": [len(x) for x in c["comps"][:5]], "n_components": len(c["comps"]),
            "nodes": nout, "edges": edges, "pointers": pointers,
            "text_check": {k: {"passed": v[0], "failed": v[1]} for k, v in lchk.counts.items()}})
        for k, v in lchk.counts.items():
            chk.counts[k][0] += v[0]; chk.counts[k][1] += v[1]
        for k, v in lchk.fail_reason.items():
            chk.fail_reason[k] += v
    # concept labels that failed the check are displayed by id only
    stats["verbatim_check_shipped"] = {k: {"passed": v[0], "failed": v[1]} for k, v in sorted(chk.counts.items())}
    stats["verbatim_fail_reasons_shipped"] = dict(chk.fail_reason)

    # ---------- rule gallery: strict-executable rules, 2 per subject (transparent rule) ----------
    gallery = []
    gchk = Checker(span_text, all_grams)
    for s in SUBJECTS:
        cands = sorted([it for it in items.values() if it["k"] == "rule" and it["strict"] and it["subject"] == s],
                       key=lambda it: (0 if it["formula"] else 1, it["id"]))
        picked = 0
        for it in cands:
            if picked >= 2:
                break
            r = {"id": it["id"], "subject": s, "rule_type": it["rule_type"], "status": it["status"], "strict": True,
                 "support": "supported", "score": it.get("score"), "n_spans": len(it["spans"]), "spans": it["spans"][:MAX_PTRS]}
            allok = True
            for part in ("premises", "guards", "conclusions", "exceptions"):
                r[part] = []
                for a in it[part]:
                    okk = gchk.ok("gallery_atom", a["text"], it["spans"])
                    r[part].append({"pred": a["predicate"], "args": safe_args(a["args"], gchk, "gallery_arg", it["spans"]), "neg": bool(a.get("negated")), "text": a["text"] if okk else None})
            if it["formula"]:
                okf = gchk.ok("gallery_formula", it["formula"], it["spans"])
                r["formula"] = it["formula"] if okf else None; r["has_formula"] = True
            r["variables"] = {k: (v if (v.split(":")[0] in SUBJECTS or gchk.ok("gallery_variable", v, it["spans"])) else None)
                              for k, v in it["variables"].items()}
            gallery.append(r); picked += 1
    gptr = {}
    for r in gallery:
        for x in r["spans"]:
            gptr[x] = pointer(x)
    for r in gallery:
        r["pointers"] = {x: gptr[x] for x in r["spans"]}
    stats["verbatim_check_gallery"] = {k: {"passed": v[0], "failed": v[1]} for k, v in gchk.counts.items()}

    # ---------- write data files ----------
    meta = {"generator": "scripts/release/viz_build_explorer.py", "ontology": stats["db"], "build_id": stats["build_id"],
            "support_policy": "Only items labelled 'supported' by the DB are drawn. partially_supported / unsupported items are excluded. "
                              "Objectives and 'teaches' links have no support label in the DB and are marked not_labelled.",
            "copyright_policy": "No textbook passage is published; statements are kept only if <= 25 words and no run of 8 words "
                                "shared with an evidence span, otherwise only ids and pointers are kept."}
    wj = lambda name, obj: json.dump(obj, open(os.path.join(OUT, "data", name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    wj("ontology_stats.json", stats)
    wj("selection.json", selection)
    wj("lessons.json", {"meta": meta, "lessons": lessons_out})
    wj("rule_gallery.json", {"meta": meta, "rules": gallery})

    app = {"meta": meta, "stats": stats, "lessons": lessons_out, "gallery": gallery, "selection": selection, "gloss_on": bool(gloss)}
    tpl = open(os.path.join(os.path.dirname(__file__), "viz_explorer_template.html"), encoding="utf-8").read()
    payload = json.dumps(app, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", payload)
    open(os.path.join(OUT, "ontology_explorer.html"), "w", encoding="utf-8").write(html)
    write_readme(stats, lessons_out, selection, chk)
    print("lessons:", [(l["subject"], l["key"], l["n_nodes"], l["n_edges"]) for l in lessons_out])
    print("text check shipped:", stats["verbatim_check_shipped"])
    print("whole supported:", stats["verbatim_check_whole_supported"])
    print("paper vs db:", [(r["item"], r["paper"], r["db"]) for r in cmp_rows if not r["equal"]] or "all equal")


def write_readme(stats, lessons, selection, chk):
    c = stats["counts"]
    ws = stats["verbatim_check_whole_supported"]
    sh = stats["verbatim_check_shipped"]
    L = []
    L.append("# OTTER ontology explorer\n")
    L.append("`ontology_explorer.html` is a single self-contained page (no CDN, no external fonts, works from `file://`). "
             "It draws one lesson graph per subject from the frozen OTTER ontology (`data/ontology/ontology.sqlite3`, read-only) "
             "and shows, for every node, its type, id, support label and **source pointers** (book, lesson heading, printed page, span id). "
             "Spans are not shown: the point is that every item links to a span you can open in the book.\n")
    L.append("Open it with any browser: `xdg-open viz/ontology_explorer.html`. Regenerate with "
             "`.venv/bin/python scripts/release/viz_build_explorer.py` from the repository root of the research code.\n")
    L.append("## Which lessons are shown (selection rule)\n")
    L.append("For each of the six subjects, one lesson is shown. Rule (stated here, applied mechanically by the script, no manual picking): build the supported subgraph "
             "of every lesson (a node is an item with at least one evidence span in that lesson; edges are typed relations, fact/rule-to-concept mentions "
             "and `teaches` links between such nodes); take the largest connected component; among the lessons whose largest component has "
             "**at most 150 nodes**, take the lesson with the **largest** component (ties: smaller lesson id). Only that component is drawn; "
             "other nodes of the lesson are not shown. A seventh *demo* entry is the lesson cited by the video's example question "
             "(`VIEDUHOP-C2-PHY-017`, gold lesson `nxbgd_g10_physics_d0daa774:ch03:l14`); it is not chosen by the rule.\n")
    L.append("| subject | lesson (as in the book) | book | nodes drawn | edges | nodes of the lesson in the DB (supported) | role |\n|---|---|---|---|---|---|---|")
    for l in lessons:
        L.append("| %s (%s) | %s | %s | %d | %d | %d | %s |" % (l["subject_vi"], l["subject_en"], l["heading"], l["book"], l["n_nodes"], l["n_edges"],
                                                                 l["nodes_in_lesson"], l["role"]))
    L.append("")
    L.append("## What is and is not drawn\n")
    L.append("* Only items labelled `supported` by the DB (source-support check of Qwen3-8B, an automatic label, **not human verification**) are used. "
             "`partially_supported` and `unsupported` items are excluded; their counts are in the Stats tab. Learning objectives and `teaches` links carry "
             "no support label in the DB; they are drawn in their own shape and marked *not labelled*.")
    L.append("* Node types: concept (circle), fact (square), rule (diamond), objective (triangle). Edges: typed relation (solid, arrow), fact/rule mentions concept (dashed), `teaches` (dotted).")
    L.append("* No textbook passage or quote is shipped. A statement (fact text, rule atom text, formula, objective line, concept label) is shown only if it has "
             "<= 25 words **and** shares no run of 8 consecutive words with any of its evidence spans, nor (stricter than required) with any other span of the 18 books "
             "(NFC, lower case, word tokens; checked programmatically against `data/textbooks/parsed/*/spans.jsonl`). Free-text rule arguments get the same check. Otherwise the page shows only the id/type/pointers.")
    L.append("* Vietnamese: concept names are the textbook's own terms and are shown untranslated (the DB has no English); subject names, lesson headings and the legend carry English glosses"
             + ("; English glosses of concept names are machine-generated, unchecked, and labelled as such." if os.path.exists(os.path.join(OUT, "data/gloss_en.json")) else "."))
    L.append("")
    L.append("## Verbatim check: how many statements passed\n")
    L.append("Shipped in the 7 lesson graphs (all text fields that could have been shown):\n")
    L.append("| statement kind | passed (shown) | failed (id only) |\n|---|---|---|")
    for k, v in sh.items():
        L.append("| %s | %d | %d |" % (k, v["passed"], v["failed"]))
    L.append("\nAll supported facts / rules / objectives in the whole ontology (for reference; not all are shipped):\n")
    L.append("| kind | passed | failed |\n|---|---|---|")
    for k, v in ws.items():
        L.append("| %s | %d | %d |" % (k, v["passed"], v["failed"]))
    L.append("")
    L.append("## Whole-ontology numbers vs. the paper (`paper/sections/method.tex`)\n")
    L.append("| item | paper | DB (v1 frozen) | equal |\n|---|---|---|---|")
    for r in stats["comparison"]:
        L.append("| %s | %s | %s | %s |" % (r["item"], format(r["paper"], ","), format(r["db"], ","), "yes" if r["equal"] else "**NO**"))
    L.append("")
    L.append("The paper's counts include items of every support label (supported / partially / unsupported); the explorer draws only `supported` ones.\n")
    L.append("## Files\n")
    L.append("* `ontology_explorer.html`: the page (data inlined, so it also works offline from a file).\n* `data/lessons.json`: the 7 lesson graphs (nodes, edges, positions, pointers).\n"
             "* `data/rule_gallery.json`: 2 strict-executable rules per subject (rule view).\n* `data/ontology_stats.json`: counts, support breakdown, comparison with the paper, text-check counters.\n"
             "* `data/selection.json`: the selection rule and the top candidates per subject.\n")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
