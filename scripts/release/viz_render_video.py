#!/usr/bin/env python3
"""Render the OTTER demo video (captions only, no audio) from the real ontology explorer data and real run outputs.

Command (from the repository root; needs ImageMagick `convert` and `ffmpeg`; run viz_build_explorer.py first):

    .venv/bin/python scripts/release/viz_render_video.py            # frames -> /tmp, writes media/otter_demo.mp4 + _preview.gif
    .venv/bin/python scripts/release/viz_render_video.py --work DIR # keep frames in DIR (default: a temp dir, removed)

Inputs (read only): release/github/OTTER/viz/data/lessons.json   (scene 3: the demo lesson graph, from the DB)
                    data/textbooks/parsed/nxbgd_g10_physics_d0daa774/spans.jsonl   (scene 2: page layout of real spans, NO text drawn;
                                                                      scene 4: printed-page check)
                    benchmarks/vieduhop_v2/vieduhop_all.jsonl, runs/vieduhop_unrev/otter_v2/main/{answers,records}.jsonl  (scene 4)
                    paper/figures/fig_worked.tex   (English glosses of the question, copied by the authors; checked by substring)
Outputs (only): release/github/OTTER/media/otter_demo.mp4  (H.264, yuv420p, 1280x720, 24 fps), otter_demo_preview.gif,
                otter_demo_verification.json (every fact shown in scene 4 with the file it was read from).

Frames are SVG written by this script (pure Python), converted with ImageMagick to PNG (DejaVu Sans, which has the Vietnamese
diacritics), assembled by ffmpeg. No textbook passage or quote is drawn anywhere; only ids, headings, page numbers and counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ = os.path.join(ROOT, "release/github/OTTER/viz")
MEDIA = os.path.join(ROOT, "release/github/OTTER/media")
W, H, FPS = 1280, 720, 24
QID = "VIEDUHOP-C2-PHY-017"
BOOK = "nxbgd_g10_physics_d0daa774"

# palette (Okabe-Ito based, dark background)
BG = "#11151a"; PANEL = "#1b2127"; INK = "#eceff1"; MUTED = "#9aa7b3"; LINE = "#3a4550"
C_CONCEPT = "#56B4E9"; C_FACT = "#E69F00"; C_RULE = "#2fcf9f"; C_OBJ = "#e49bc3"; C_SPAN = "#8c98a4"; C_HI = "#F0E442"
C_OK = "#2fcf9f"; C_BAD = "#e8743b"
SANS = "DejaVu Sans"; MONO = "DejaVu Sans Mono"

T1, T2, T3, T4, T5, TEND = 0.0, 5.0, 17.0, 33.0, 60.0, 68.0


# ------------------------------------------------------------------ helpers
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ramp(t, a, b):
    return clamp((t - a) / (b - a)) if b > a else (1.0 if t >= a else 0.0)


def ease(x):
    return x * x * (3 - 2 * x)


def text(x, y, s, size=20, fill=INK, weight="normal", anchor="start", family=SANS, op=1.0, style="normal"):
    if op <= 0.003:
        return ""
    return ('<text x="%.1f" y="%.1f" font-family="%s" font-size="%s" font-weight="%s" font-style="%s" fill="%s" text-anchor="%s"%s>%s</text>'
            % (x, y, family, size, weight, style, fill, anchor, (' fill-opacity="%.3f"' % op) if op < 1 else "", esc(s)))


def rect(x, y, w, h, fill="none", stroke="none", sw=1, rx=0, op=1.0, sop=1.0):
    if not ((fill != "none" and op > 0.003) or (stroke != "none" and sop > 0.003)):
        return ""
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%s" fill="%s" fill-opacity="%.3f" stroke="%s" stroke-opacity="%.3f" stroke-width="%s"/>'
            % (x, y, max(w, 0), max(h, 0), rx, fill, op, stroke, sop, sw))


def line(x1, y1, x2, y2, stroke=LINE, sw=1, op=1.0, dash=None):
    if op <= 0.003:
        return ""
    return ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s" stroke-opacity="%.3f"%s/>'
            % (x1, y1, x2, y2, stroke, sw, op, (' stroke-dasharray="%s"' % dash) if dash else ""))


def wrap(s, size, maxw, bold=False):
    cw = size * (0.64 if bold else 0.58)
    maxc = max(8, int(maxw / cw))
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= maxc:
            cur = (cur + " " + w) if cur else w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def para(x, y, s, size=20, maxw=1100, fill=INK, lh=1.35, op=1.0, weight="normal", style="normal"):
    out, yy = [], y
    for ln in wrap(s, size, maxw, weight == "bold"):
        out.append(text(x, yy, ln, size, fill, weight, op=op, style=style)); yy += size * lh
    return "".join(out), yy


def pill(x, y, w, h, label, fill, ink="#0b0e11", size=16, op=1.0, weight="bold"):
    return rect(x, y, w, h, fill, rx=h / 2, op=op) + text(x + w / 2, y + h / 2 + size * 0.35, label, size, ink, weight, "middle", op=op)


def caption(s, t_local, dur, sub=None):
    op = ramp(t_local, 0.2, 0.7) * (1 - ramp(t_local, dur - 0.5, dur - 0.1))
    out = [rect(0, 652, W, 68, "#0b0e11", op=0.92 * op)]
    lines = wrap(s, 22, 1180)
    y = 683 if len(lines) == 1 else 678
    for i, ln in enumerate(lines[:2]):
        out.append(text(W / 2, y + i * 26, ln, 22, INK, "normal", "middle", op=op))
    if sub:
        out.append(text(W / 2, 706, sub, 14, MUTED, "normal", "middle", op=op))
    return "".join(out)


def header(label):
    return text(32, 40, "OTTER", 22, C_CONCEPT, "bold") + text(124, 40, label, 16, MUTED)


# ------------------------------------------------------------------ data
def load_data():
    d = {}
    d["lessons"] = json.load(open(os.path.join(VIZ, "data/lessons.json"), encoding="utf-8"))["lessons"]
    d["demo"] = [l for l in d["lessons"] if l["role"] == "demo"][0]
    # spans of the physics-10 book (layout only; text length is used for bar heights, the text itself is never drawn)
    spans, n_spans_book = {}, 0
    for line_ in open(os.path.join(ROOT, "data/textbooks/parsed/%s/spans.jsonl" % BOOK), encoding="utf-8"):
        s = json.loads(line_)
        n_spans_book += 1
        spans[s["span_id"]] = {"page": s["pdf_page_index"], "printed": s["printed_page_label"], "verified": s["printed_page_verified"],
                               "len": len(s["text"]), "kind": s["span_kind"], "lesson": s["lesson_id"]}
    d["spans"] = spans
    tot_books, tot_spans = 0, 0
    pdir = os.path.join(ROOT, "data/textbooks/parsed")
    for b in sorted(os.listdir(pdir)):
        if b.startswith("nxbgd_"):
            tot_books += 1
            with open(os.path.join(pdir, b, "spans.jsonl"), encoding="utf-8") as f:
                tot_spans += sum(1 for _ in f)
    d["n_books"], d["n_spans"] = tot_books, tot_spans
    # scene 4 sources
    def find(path, key):
        for l in open(os.path.join(ROOT, path), encoding="utf-8"):
            r = json.loads(l)
            if r.get(key) == QID or r.get("id") == QID:
                return r
        raise SystemExit("%s not found in %s" % (QID, path))
    d["bench"] = find("benchmarks/vieduhop_v2/vieduhop_all.jsonl", "id")
    d["ans"] = find("runs/vieduhop_unrev/otter_v2/main/answers.jsonl", "question_id")
    d["rec"] = find("runs/vieduhop_unrev/otter_v2/main/records.jsonl", "id")
    d["meta"] = json.load(open(os.path.join(ROOT, "runs/vieduhop_unrev/otter_v2/main/run_meta.json"), encoding="utf-8"))
    d["tex"] = open(os.path.join(ROOT, "paper/figures/fig_worked.tex"), encoding="utf-8").read()
    d["method"] = open(os.path.join(ROOT, "paper/sections/method.tex"), encoding="utf-8").read()
    return d


def verify_scene4(d):
    """Check every fact scene 4 shows against the files; return a log (also written to media/otter_demo_verification.json)."""
    b, a, r, meta, tex = d["bench"], d["ans"], d["rec"], d["meta"], d["tex"]
    log = []

    def chk(name, cond, value, src):
        log.append({"fact": name, "value": value, "source": src, "ok": bool(cond)})
        if not cond:
            raise SystemExit("VERIFICATION FAILED: " + name)
    tu = a["tool_usage"]
    chk("item id / domain / grade / hops", b["id"] == QID and b["subjects"] == ["physics"] and b["grade"] == 10 and b["hop_count"] == 2 and r["hop_count"] == 2,
        [b["id"], b["subjects"], b["grade"], b["hop_count"]], "benchmarks/vieduhop_v2/vieduhop_all.jsonl; records.jsonl")
    chk("question text + 4 options", set(b["options"]) == set("ABCD"), b["question"], "benchmarks/vieduhop_v2/vieduhop_all.jsonl")
    chk("gold key B", b["answer"] == "B" and r["gold"] == "B", "B", "vieduhop_all.jsonl (answer); records.jsonl (gold)")
    chk("answer B, status supported", a["answer_choice"] == "B" and a["status"] == "supported" and r["choice"] == "B" and r["correct"] is True,
        {"answer_choice": a["answer_choice"], "status": a["status"], "correct": r["correct"]}, "answers.jsonl; records.jsonl")
    verd = {v["choice_id"]: v["status"] for v in a["choice_verdicts"]}
    chk("per-option verdicts", verd == {"A": "contradicted", "B": "entailed", "C": "contradicted", "D": "contradicted"}, verd, "answers.jsonl (choice_verdicts)")
    chk("grounding level 3", tu["v2_grounding"] == 3 and tu["v2_answer_supported"] is True, tu["v2_grounding"], "answers.jsonl (tool_usage.v2_grounding)")
    chk("2 LLM calls, 2,546 tokens", tu["llm_calls"] == 2 and tu["tokens"] == 2546 and r["cost_llm_calls"] == 2.0, [tu["llm_calls"], tu["tokens"]], "answers.jsonl; records.jsonl")
    chk("Judge: 1 call, 1,758 tokens; Verifier: 1 call, 788 tokens", tu["main_llm_calls"] == 1 and tu["main_tokens"] == 1758 and tu["verifier_llm_calls"] == 1 and tu["verifier_tokens"] == 788,
        [tu["main_llm_calls"], tu["main_tokens"], tu["verifier_llm_calls"], tu["verifier_tokens"]], "answers.jsonl (tool_usage)")
    chk("backbone Qwen3-32B; verifier Qwen3-8B", meta["backbone_model"] == "Qwen/Qwen3-32B" and "Qwen3-8B" in open(os.path.join(ROOT, "configs/otter_v2.yaml")).read(),
        meta["backbone_model"], "run_meta.json; configs/otter_v2.yaml (verifier_role: small = Qwen3-8B)")
    chk("7 retrieval calls, 10 spans retrieved, 1 round", tu["retrieval_calls"] == 7 and len(tu["retrieved_span_ids"]) == 10 and tu["v2_rounds"] == 1 and tu["v2_route"] == "single_round",
        [tu["retrieval_calls"], len(tu["retrieved_span_ids"]), tu["v2_rounds"], tu["v2_route"]], "answers.jsonl (tool_usage)")
    cl = tu["v2_claims"]
    chk("3 claims, all supported with quote found, 0 dropped", len(cl) == 3 and all(c["verdict"] == "supported" and c["quote_ok"] for c in cl) and tu["v2_n_claims_dropped"] == 0,
        [(c["verdict"], c["quote_ok"]) for c in cl], "answers.jsonl (tool_usage.v2_claims)")
    wr = a["where_to_read"]
    gold = b["gold_span_ids"]
    chk("two source pointers = the two gold spans", [w["span_id"] for w in wr] == gold == ["%s:p0062:u007" % BOOK, "%s:p0063:u018" % BOOK] and r["ev_recall"] == 1.0,
        gold, "answers.jsonl (where_to_read); vieduhop_all.jsonl (gold_span_ids); records.jsonl (ev_recall)")
    for w, sp in zip(wr, b["source_pointers"]):
        s = d["spans"][w["span_id"]]
        chk("pointer %s: book / lesson / printed page" % w["span_id"],
            w["book_title"] == sp["book_title"] == "Vật lí 10" and w["page_number"] == sp["printed_page"] == s["printed"] and s["verified"] is True and w["lesson_id"] == sp["lesson_key"] == s["lesson"],
            {"book": w["book_title"], "lesson": sp["lesson"], "printed_page": w["page_number"], "printed_page_verified": s["verified"]},
            "answers.jsonl; vieduhop_all.jsonl; spans.jsonl (printed_page_verified)")
    for k in ("question_gloss", "optA", "optB", "optC", "optD", "lessonA", "lessonB"):
        chk("English gloss '%s' occurs in paper/figures/fig_worked.tex" % k, GLOSS[k] in tex.replace("\\emph{", "").replace("}", "").replace("\\_", "_"), GLOSS[k],
            "paper/figures/fig_worked.tex Panel B")
    chk("role order in the paper", "Composer, Router, Retriever, Judge, Verifier and Composer" in d["method"], ROLES, "paper/sections/method.tex")
    chk("item is auto-verified, not teacher reviewed", b["review_status"] == "auto_verified", b["review_status"], "vieduhop_all.jsonl (review_status)")
    chk("counts: 18 books, 16,716 spans", d["n_books"] == 18 and d["n_spans"] == 16716, [d["n_books"], d["n_spans"]], "data/textbooks/parsed/nxbgd_*/spans.jsonl")
    chk("demo lesson graph", d["demo"]["n_nodes"] == 24 and d["demo"]["n_edges"] == 30, [d["demo"]["n_nodes"], d["demo"]["n_edges"]], "viz/data/lessons.json")
    return log


GLOSS = {
    "question_gloss": "What is the difference between Newton's first and second laws when it comes to inertia?",
    "optA": "first law: change of velocity; second law: balance of forces",
    "optB": "first law: the property of preserving a body's state; second law: how hard it is to change a body's velocity",
    "optC": "first: acceleration; second: resting or moving",
    "optD": "first: force acting on the body; second: state of motion",
    "lessonA": "Newton's first law",
    "lessonB": "Newton's second law",
}
ROLES = ["Composer", "Router", "Retriever", "Judge", "Verifier", "Composer"]


# ------------------------------------------------------------------ scenes
def scene1(t, D):
    o = []
    a = ease(ramp(t, 0.2, 1.4))
    o.append(text(W / 2, 330, "OTTER", 150, INK, "bold", "middle", op=a))
    o.append(rect(W / 2 - 260 * ease(ramp(t, 1.0, 2.0)), 360, 520 * ease(ramp(t, 1.0, 2.0)), 4, C_CONCEPT))
    b = ease(ramp(t, 1.6, 2.6))
    o.append(text(W / 2, 425, "Verifiable Multihop Reasoning over Educational Ontology", 29, INK, "normal", "middle", op=b))
    o.append(text(W / 2, 462, "for Vietnamese Education", 29, INK, "normal", "middle", op=b))
    c = ease(ramp(t, 2.6, 3.6))
    o.append(text(W / 2, 520, "Every item and every answer points to a source span in a textbook.", 20, MUTED, "normal", "middle", op=c))
    return "".join(o), "Textbook question answering where every claim can be traced to a page."


def page_layout(D, pdf_page):
    ids = sorted([s for s, v in D["spans"].items() if v["page"] == pdf_page], key=lambda s: int(s.rsplit(":u", 1)[1]))
    return ids


def scene2(t, D):
    o = []
    pages = [61, 62, 63]
    xs = [100, 475, 850]; pw, ph, py = 330, 420, 100
    sweep_t = ramp(t, 0.5, 3.0)
    spans_on = ramp(t, 3.4, 7.4)
    all_blocks = []
    for pi, (pg, x0) in enumerate(zip(pages, xs)):
        appear = ease(ramp(t, 0.1 + pi * 0.25, 0.9 + pi * 0.25))
        o.append(rect(x0, py + (1 - appear) * 30, pw, ph, "#d8d3c6", "#8a8578", 1, 4, op=appear))
        if appear < 0.99:
            continue
        ids = page_layout(D, pg)
        y = py + 18
        for sid in ids:
            sp = D["spans"][sid]
            nl = 1 if sp["kind"] == "heading" else int(clamp(sp["len"] // 75 + 1, 1, 6))
            hblk = nl * 10 + 8
            if y + hblk > py + ph - 14:
                break
            all_blocks.append((pi, sid, x0 + 18, y, pw - 36, hblk, nl, sp["kind"]))
            y += hblk + 5
    n_blocks = len(all_blocks)
    for k, (pi, sid, bx, by, bw, bh, nl, kind) in enumerate(all_blocks):
        # reveal by scanning sweep (top to bottom)
        rel = (by - (py + 18)) / (ph - 36)
        vis = sweep_t >= rel
        if not vis and t < 3.0:
            continue
        rnd = zlib.crc32(sid.encode()) & 0xffff
        for li in range(nl):
            wfrac = 1.0 if li < nl - 1 else 0.35 + ((rnd >> (li % 8)) % 60) / 100.0
            if kind == "heading":
                o.append(rect(bx, by + 4, bw * 0.55, 9, "#5c574c", rx=2))
            else:
                o.append(rect(bx, by + 4 + li * 10, bw * min(wfrac, 1.0), 5, "#8a8578", rx=2))
        # span outline appears one by one
        thr = k / max(n_blocks, 1)
        on = ease(clamp((spans_on - thr * 0.85) / 0.15))
        cited = (sid == "%s:p0062:u007" % BOOK)
        if on > 0:
            o.append(rect(bx - 5, by - 1, bw + 10, bh, "none", C_HI if cited else C_CONCEPT, 2 if cited else 1.5, 3, sop=on))
            if cited or k % 6 == 0:
                lab = ":" + sid.split(":", 1)[1]
                lw = len(lab) * 6.3 + 8
                o.append(rect(bx + bw - lw + 5, by - 9, lw, 13, BG, rx=3, op=0.9 * on))
                o.append(text(bx + bw - lw + 9, by + 1, lab, 10, C_HI if cited else C_CONCEPT, "normal", family=MONO, op=on))
    if sweep_t < 1.0 and sweep_t > 0:
        for x0 in xs:
            yy = py + ph * sweep_t
            o.append(rect(x0, yy, pw, 3, C_CONCEPT, op=0.8))
    for pi, x0 in enumerate(xs):
        o.append(text(x0 + pw / 2, py + ph + 24, "page %s" % D["spans"]["%s:p%04d:u000" % (BOOK, pages[pi])]["printed"] if ("%s:p%04d:u000" % (BOOK, pages[pi])) in D["spans"] else "page", 15, MUTED, "normal", "middle", op=ease(ramp(t, 1.5, 2.5))))
    cnt = ease(ramp(t, 7.8, 8.8))
    nb = int(round(D["n_books"] * ease(ramp(t, 7.8, 9.6))))
    ns = int(round(D["n_spans"] * ease(ramp(t, 7.8, 10.0))))
    o.append(text(W / 2, 598, "%d books  ·  %s source spans" % (nb, format(ns, ",")), 30, INK, "bold", "middle", op=cnt))
    o.append(text(W / 2, 628, "span id = <book>:p<page>:u<k>   e.g. %s:p0062:u007" % BOOK, 15, MUTED, "normal", "middle", family=MONO, op=cnt))
    d = TEND  # unused
    cap = "Scanned textbook pages are split into spans, each with an id. No page text is shown here: only the layout and the ids."
    return "".join(o), cap


def scene3(t, D):
    L = D["demo"]; nodes = L["nodes"]; edges = L["edges"]
    N = len(nodes)
    byid = {n["id"]: n for n in nodes}
    deg = {n["id"]: 0 for n in nodes}
    for e in edges:
        deg[e["s"]] += 1; deg[e["t"]] += 1
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["s"]].append(e["t"]); adj[e["t"]].append(e["s"])
    root = max(nodes, key=lambda n: (deg[n["id"]], n["id"]))["id"]
    order, seen, q = [], {root}, [root]
    while q:
        x = q.pop(0); order.append(x)
        for y in sorted(adj[x], key=lambda i: -deg[i]):
            if y not in seen:
                seen.add(y); q.append(y)
    for n in nodes:
        if n["id"] not in seen:
            order.append(n["id"])
    tin = {nid: 1.2 + i * (9.0 / max(N - 1, 1)) for i, nid in enumerate(order)}
    # geometry
    gx0, gx1, gy0, gy1 = 80, 1040, 165, 535
    xs = [n["x0"] if "x0" in n else n["x"] for n in nodes]; ys = [n["y0"] if "y0" in n else n["y"] for n in nodes]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    pos = {n["id"]: (gx0 + (n["x"] - xmin) / max(xmax - xmin, 1) * (gx1 - gx0), gy0 + (n["y"] - ymin) / max(ymax - ymin, 1) * (gy1 - gy0)) for n in nodes}
    # span strip
    fs = {}
    for n in nodes:
        fs.setdefault(n["spans"][0], []).append(n["id"])
    sp_order = sorted(fs, key=lambda s: sum(pos[i][0] for i in fs[s]) / len(fs[s]))
    spx = {s: 70 + k * (1140 / max(len(sp_order) - 1, 1)) for k, s in enumerate(sp_order)}
    o = [header("scene 3: the ontology of one real lesson")]
    o.append(text(W / 2, 72, "%s  ·  %s" % (L["book"], L["heading"]), 22, INK, "bold", "middle", op=ease(ramp(t, 0.1, 0.8))))
    o.append(text(W / 2, 94, "(%s)  ·  supported items only" % L["heading_en"], 15, MUTED, "normal", "middle", op=ease(ramp(t, 0.3, 1.0))))
    # span strip first
    sy = 590
    o.append(text(60, sy - 16, "source spans in the book (ids)", 13, MUTED, "normal", op=ease(ramp(t, 0.8, 1.6))))
    for s in sp_order:
        a = ease(ramp(t, min(tin[i] for i in fs[s]) - 0.1, min(tin[i] for i in fs[s]) + 0.4))
        o.append(rect(spx[s] - 9, sy, 18, 30, C_SPAN, op=0.9 * a, rx=2))
    # edges
    for e in edges:
        a, b = pos[e["s"]], pos[e["t"]]
        te = max(tin[e["s"]], tin[e["t"]]) + 0.15
        p = ease(ramp(t, te, te + 0.5))
        if p <= 0:
            continue
        col = "#c3ccd4" if e["k"] == "relation" else ("#d58bbd" if e["k"] == "teaches" else "#6f7c88")
        o.append(line(a[0], a[1], a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p, col, 2.2 if e["k"] == "relation" else 1.2, 0.9, None if e["k"] == "relation" else "5 4"))
    # node -> span thin lines, then nodes
    for n in nodes:
        a = ease(ramp(t, tin[n["id"]], tin[n["id"]] + 0.5))
        if a <= 0:
            continue
        sx = spx[n["spans"][0]]
        x, y = pos[n["id"]]
        o.append(line(x, y, x + (sx - x) * a, y + (sy - y) * a, C_SPAN, 0.8, 0.55 * a))
    for n in nodes:
        a = ease(ramp(t, tin[n["id"]], tin[n["id"]] + 0.5))
        if a <= 0:
            continue
        x, y = pos[n["id"]]; r = 10 * (0.4 + 0.6 * a)
        if n["k"] == "concept":
            o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" fill-opacity="%.2f" stroke="%s" stroke-width="2"/>' % (x, y, r, C_CONCEPT, a, BG))
            lb = n.get("label") or n["id"].split(":")[-1]
            lb = lb if len(lb) <= 26 else lb[:25] + "…"
            o.append(text(x + 14, y + 5, lb, 15, INK, op=a))
        elif n["k"] == "fact":
            o.append(rect(x - r, y - r, 2 * r, 2 * r, C_FACT, BG, 2, 1, op=a))
        elif n["k"] == "rule":
            o.append('<path d="M%.1f %.1fL%.1f %.1fL%.1f %.1fL%.1f %.1fz" fill="%s"/>' % (x, y - r * 1.3, x + r * 1.3, y, x, y + r * 1.3, x - r * 1.3, y, C_RULE))
        else:
            o.append('<path d="M%.1f %.1fL%.1f %.1fL%.1f %.1fz" fill="%s"/>' % (x, y - r * 1.2, x + r * 1.2, y + r, x - r * 1.2, y + r, C_OBJ))
    # legend + counts
    la = ease(ramp(t, 10.8, 11.6))
    o.append('<circle cx="100" cy="132" r="8" fill="%s" fill-opacity="%.2f"/>' % (C_CONCEPT, la) + text(116, 137, "concept", 15, INK, op=la))
    o.append(rect(200, 124, 16, 16, C_FACT, op=la) + text(224, 137, "fact", 15, INK, op=la))
    o.append(line(280, 132, 310, 132, "#c3ccd4", 2.2, la) + text(318, 137, "typed relation", 15, INK, op=la))
    o.append(line(440, 132, 470, 132, "#6f7c88", 1.2, la, "5 4") + text(478, 137, "fact mentions concept", 15, INK, op=la))
    o.append(line(660, 132, 690, 132, C_SPAN, 1.0, la) + text(698, 137, "item -> its source span", 15, INK, op=la))
    k = {"concept": 0, "fact": 0}
    for n in nodes:
        k[n["k"]] = k.get(n["k"], 0) + 1
    ca = ease(ramp(t, 10.3, 11.1))
    o.append(rect(W / 2 - 330, 541, 660, 28, BG, op=0.85 * ca, rx=6))
    o.append(text(W / 2, 562, "%d nodes (%d concepts, %d facts), %d edges, %d source spans" % (N, k["concept"], k["fact"], len(edges), len(sp_order)), 18, INK, "bold", "middle", op=ca))
    o.append(text(W / 2, 640, "each grey block is one span, e.g. %s" % sp_order[0], 13, MUTED, "normal", "middle", family=MONO, op=ca))
    cap = "One real lesson becomes a graph of concepts, facts and relations. Each node keeps a thin line to the span that states it."
    return "".join(o), cap


def scene4(t, D):
    b, a, r = D["bench"], D["ans"], D["rec"]
    tu = a["tool_usage"]
    o = [header("scene 4: answering a real question")]
    verd = {v["choice_id"]: v["status"] for v in a["choice_verdicts"]}
    if t < 6.9:
        fade = 1 - ramp(t, 6.4, 6.9)
        o.append(text(32, 78, "ViEduHop item %s  ·  Physics, grade 10  ·  2 hops" % QID, 18, MUTED, op=fade))
        s, y = para(32, 122, b["question"], 27, 1200, INK, 1.3, op=ease(ramp(t, 0.3, 1.0)) * fade, weight="bold")
        o.append(s)
        s, y = para(32, y - 6, "(" + GLOSS["question_gloss"] + ")", 17, 1200, MUTED, 1.3, op=ease(ramp(t, 0.6, 1.3)) * fade)
        o.append(s)
        gl = {"A": GLOSS["optA"], "B": GLOSS["optB"], "C": GLOSS["optC"], "D": GLOSS["optD"]}
        y += 20
        for i, k in enumerate("ABCD"):
            op = ease(ramp(t, 1.8 + i * 0.8, 2.4 + i * 0.8)) * fade
            o.append(rect(32, y - 22, 1216, 84, PANEL, LINE, 1, 8, op=op, sop=op))
            o.append(pill(46, y - 10, 34, 34, k, C_CONCEPT, size=20, op=op))
            s, y2 = para(96, y + 4, b["options"][k], 18, 1130, INK, 1.25, op=op)
            o.append(s)
            o.append(text(96, y2 - 2 if y2 < y + 36 else y2 - 2, "(" + gl[k] + ")", 13, MUTED, op=op))
            y += 96
        o.append(text(32, 628, "English glosses in grey are the authors' own, copied from the paper's worked example (Fig. worked, panel B).", 13, MUTED, op=ease(ramp(t, 4.0, 5.0)) * fade))
        cap = "A two-hop question from the ViEduHop benchmark: it needs two lessons of the book."
        return "".join(o), cap
    # phase B / C
    tt = t - 6.4
    fadein = ease(ramp(tt, 0.0, 0.5))
    o.append(text(32, 78, "ViEduHop item %s  ·  auto-verified item, run on the unreviewed set" % QID, 15, MUTED, op=fadein))
    s, y = para(32, 108, b["question"], 19, 1216, INK, 1.25, op=fadein, weight="bold")
    o.append(s)
    # roles row
    bw, gap, ry, rh = 188, 17.6, 150, 54
    starts = [0.4 + i * 2.0 for i in range(6)]   # relative to tt
    desc = [
        "Receives the question and the four options.",
        "Deterministic. Builds the queries: question with all options, question alone, each (question, option) pair.",
        "Deterministic hybrid search over source spans, seeded by ontology hints. Run record: 7 retrieval calls, 10 spans retrieved.",
        "One Qwen3-32B call: labels every option, chooses an answer, writes 3 claims, each with a span id. One round, no re-retrieval.",
        "Checks that each quote occurs in its cited span; one Qwen3-8B call: 3 of 3 claims verified, 0 dropped.",
        "Computes the grounding level: 3 (entailed, quote found and verified), so the status is supported. Attaches the source pointers.",
    ]
    rec_lines = [None, None, "Retriever: 7 retrieval calls, 10 spans retrieved", "Judge: 1 LLM call, 1,758 tokens (Qwen3-32B)",
                 "Verifier: 1 LLM call, 788 tokens (Qwen3-8B)", "Total: 2 LLM calls, 2,546 tokens; grounding level 3 -> supported"]
    active = -1
    for i in range(6):
        if tt >= starts[i]:
            active = i
    final = tt >= 12.9
    for i, nm in enumerate(ROLES):
        x = 32 + i * (bw + gap)
        on = tt >= starts[i]
        cur = (i == active) and not final
        fill = C_HI if cur else (C_CONCEPT if on else PANEL)
        ink = "#0b0e11" if on else MUTED
        o.append(rect(x, ry, bw, rh, fill, LINE if not on else fill, 1.5, 10, op=fadein, sop=fadein))
        o.append(text(x + bw / 2, ry + 25, "%d %s" % (i + 1, nm), 19, ink, "bold", "middle", op=fadein))
        o.append(text(x + bw / 2, ry + 44, ["input", "deterministic", "deterministic", "Qwen3-32B", "Qwen3-8B + exact match", "output"][i], 12, ink, "normal", "middle", op=fadein))
        if i < 5:
            o.append(text(x + bw + gap / 2, ry + 33, ">", 18, MUTED, "bold", "middle", op=fadein))
    if not final:
        # detail panel
        o.append(rect(32, 224, 1216, 100, PANEL, LINE, 1, 8, op=fadein, sop=fadein))
        if active >= 0:
            ph = ease(ramp(tt - starts[active], 0.0, 0.4))
            o.append(text(48, 254, "%d. %s" % (active + 1, ROLES[active]), 20, C_HI, "bold", op=ph))
            s, _ = para(48, 282, desc[active], 18, 1180, INK, 1.3, op=ph)
            o.append(s)
        # verdicts
        o.append(text(32, 358, "Verdicts (option text = English gloss)", 16, MUTED, "bold", op=ease(ramp(tt, starts[3], starts[3] + 0.5)) ))
        for i, k in enumerate("ABCD"):
            vy = 382 + i * 54
            va = ease(ramp(tt, starts[3] + 0.5 + i * 0.45, starts[3] + 1.0 + i * 0.45))
            col = C_OK if verd[k] == "entailed" else C_BAD
            o.append(rect(32, vy, 700, 44, PANEL, LINE, 1, 8, op=va, sop=va))
            o.append(pill(42, vy + 6, 30, 32, k, C_CONCEPT, size=18, op=va))
            gtxt = {"A": GLOSS["optA"], "B": GLOSS["optB"], "C": GLOSS["optC"], "D": GLOSS["optD"]}[k]
            sg, _ = para(84, vy + 19, gtxt, 13, 480, INK, 1.2, op=va)
            o.append(sg)
            o.append(pill(580, vy + 8, 142, 28, verd[k], col, size=15, op=va))
        # record lines
        o.append(text(770, 358, "Run record (answers.jsonl)", 18, MUTED, "bold", op=ease(ramp(tt, starts[2], starts[2] + 0.5)) ))
        ry2 = 392
        for i, ln in enumerate(rec_lines):
            if ln is None:
                continue
            va = ease(ramp(tt, starts[i] + 1.0, starts[i] + 1.5))
            s, ry2n = para(770, ry2, ln, 16, 470, INK, 1.3, op=va)
            o.append(s); ry2 = ry2n + 12
        ansa = ease(ramp(tt, starts[3] + 2.6, starts[3] + 3.2))
        o.append(text(500, 358, "Judge's choice: B", 18, C_OK, "bold", op=ansa))
        cap = "Six roles in order. In this run, 2 LLM calls in total: one by the Judge, one by the Verifier."
        return "".join(o), cap
    # final pointers
    tf = tt - 12.9
    fa = ease(ramp(tf, 0.0, 0.6))
    o.append(rect(32, 224, 1216, 410, BG, op=fa))
    o.append(text(640, 262, "Answer: B   ·   status: supported   ·   grounding level 3", 28, C_OK, "bold", "middle", op=fa))
    o.append(text(640, 292, "Source pointers (book, lesson, printed page, span id): no passage is shown, open the book at that span.", 17, MUTED, "normal", "middle", op=fa))
    wr = a["where_to_read"]
    gl = [GLOSS["lessonA"], GLOSS["lessonB"]]
    for i, w in enumerate(wr):
        ca = ease(ramp(tf, 0.8 + i * 0.9, 1.5 + i * 0.9))
        x = 32 + i * 616
        o.append(rect(x, 320, 600, 215, PANEL, C_HI, 2, 10, op=ca, sop=ca))
        o.append(text(x + 20, 354, "Source pointer %d" % (i + 1), 18, C_HI, "bold", op=ca))
        o.append(text(x + 20, 392, w["book_title"], 24, INK, "bold", op=ca))
        lesson = "Bài %s %s" % (w["lesson_id"].split(":l")[-1].lstrip("0"), w["lesson_title"])
        o.append(text(x + 20, 426, lesson, 21, INK, op=ca))
        o.append(text(x + 20, 450, "(" + gl[i] + ")", 15, MUTED, op=ca))
        o.append(text(x + 20, 485, "printed page %s (verified)" % w["page_number"], 21, INK, op=ca))
        o.append(text(x + 20, 520, w["span_id"], 14, C_CONCEPT, "normal", family=MONO, op=ca))
    na = ease(ramp(tf, 2.9, 3.6))
    o.append(text(640, 570, "This one item matches the benchmark key (B) and cites both gold spans.", 18, INK, "normal", "middle", op=na))
    o.append(text(640, 600, "A single example, not an accuracy result. Accuracy is reported in the paper.", 16, MUTED, "normal", "middle", op=na))
    cap = "The answer carries a source pointer: book, lesson, printed page and span id."
    return "".join(o), cap


def scene5(t, D):
    o = []
    a = ease(ramp(t, 0.2, 1.0))
    o.append(text(W / 2, 250, "OTTER", 120, INK, "bold", "middle", op=a))
    o.append(rect(W / 2 - 200, 275, 400, 4, C_CONCEPT, op=a))
    b = ease(ramp(t, 1.0, 1.8))
    o.append(rect(240, 330, 800, 64, PANEL, LINE, 1, 12, op=b, sop=b))
    o.append(text(W / 2, 373, "github.com/quin210/OTTER", 32, C_CONCEPT, "bold", "middle", family=MONO, op=b))
    c = ease(ramp(t, 1.8, 2.6))
    o.append(rect(240, 410, 800, 64, PANEL, LINE, 1, 12, op=c, sop=c))
    o.append(text(W / 2, 452, "Hugging Face: ViEduHop", 32, INK, "bold", "middle", op=c))
    d = ease(ramp(t, 2.8, 3.6))
    o.append(text(W / 2, 540, "Research prototype; accuracy results in the paper", 26, MUTED, "normal", "middle", op=d))
    return "".join(o), None


def frame_svg(ti, D):
    t = ti / FPS
    if t < T2:
        body, cap = scene1(t - T1, D); lt, dur = t - T1, T2 - T1
    elif t < T3:
        body, cap = scene2(t - T2, D); lt, dur = t - T2, T3 - T2; body = header("scene 2: scanned pages become source spans") + body
    elif t < T4:
        body, cap = scene3(t - T3, D); lt, dur = t - T3, T4 - T3
    elif t < T5:
        body, cap = scene4(t - T4, D); lt, dur = t - T4, T5 - T4
    else:
        body, cap = scene5(t - T5, D); lt, dur = t - T5, TEND - T5
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (W, H, W, H), rect(0, 0, W, H, BG), body]
    if cap:
        parts.append(caption(cap, lt, dur))
    # scene cross-fade to/from background
    fo = 1 - ramp(lt, 0.0, 0.35) if t >= T2 else 0.0
    fo2 = ramp(lt, dur - 0.35, dur) if t < TEND - 1e-6 else ramp(lt, dur - 0.6, dur)
    ov = max(fo, fo2)
    if ov > 0.003:
        parts.append(rect(0, 0, W, H, BG, op=ov))
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ render
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None, help="directory for frames (kept); default: temp dir, removed")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--only-frames", type=int, default=0, help="debug: render only every N-th frame and skip ffmpeg")
    args = ap.parse_args()
    D = load_data()
    log = verify_scene4(D)
    os.makedirs(MEDIA, exist_ok=True)
    json.dump({"item": QID, "files": ["benchmarks/vieduhop_v2/vieduhop_all.jsonl", "runs/vieduhop_unrev/otter_v2/main/answers.jsonl",
                                      "runs/vieduhop_unrev/otter_v2/main/records.jsonl", "runs/vieduhop_unrev/otter_v2/main/run_meta.json",
                                      "paper/figures/fig_worked.tex", "paper/sections/method.tex", "configs/otter_v2.yaml",
                                      "data/textbooks/parsed/*/spans.jsonl", "release/github/OTTER/viz/data/lessons.json"],
               "checks": log}, open(os.path.join(MEDIA, "otter_demo_verification.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    work = args.work or tempfile.mkdtemp(prefix="otter_frames_")
    os.makedirs(work, exist_ok=True)
    n = int(TEND * FPS)
    cache = {}  # svg hash -> png path
    jobs, names = [], []
    for i in range(n):
        if args.only_frames and i % args.only_frames:
            names.append(None); continue
        svg = frame_svg(i, D)
        h = hashlib.sha1(svg.encode()).hexdigest()[:16]
        png = os.path.join(work, h + ".png")
        if h not in cache:
            cache[h] = png
            if not os.path.exists(png):
                sp = os.path.join(work, h + ".svg")
                open(sp, "w", encoding="utf-8").write(svg)
                jobs.append((sp, png))
        names.append(png)
    print("frames %d, unique %d, to render %d" % (n, len(cache), len(jobs)), flush=True)

    def conv(j):
        sp, png = j
        subprocess.run(["convert", "-background", BG, sp, "PNG24:" + png], check=True)
        os.remove(sp)
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(conv, jobs))
    if args.only_frames:
        print("debug frames in", work); return
    seq = os.path.join(work, "seq")
    os.makedirs(seq, exist_ok=True)
    for i, p in enumerate(names):
        dst = os.path.join(seq, "f%05d.png" % i)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(p, dst)
    mp4 = os.path.join(MEDIA, "otter_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(seq, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-profile:v", "high", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-r", str(FPS), mp4], check=True)
    gif = os.path.join(MEDIA, "otter_demo_preview.gif")
    pal = os.path.join(work, "pal.png")
    vf = "fps=8,scale=640:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf + ",palettegen=max_colors=96:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", pal, "-lavfi", vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", gif], check=True)
    print("wrote", mp4, os.path.getsize(mp4), gif, os.path.getsize(gif))
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
