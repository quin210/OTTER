"""Synthetic 'new edition' of a corpus for the knowledge-maintenance simulation (LLM-free, deterministic).

From the real span set it produces a second edition by
  (i)  deleting a fraction of spans (removed content),
  (ii) editing a fraction of spans that contain an editable token by swapping ONE value everywhere it occurs in
       that span: a number, a year, or a named entity (replaced by another entity of the same book). No negation
       is inserted and no sentence is rewritten, so the edit is exactly the recorded (old -> new) pair,
  (iii) changing nothing else: ids, order, headings, all other texts stay identical.
Every change is recorded (`log`), which is the ground truth for the evaluation. Seeded: same seed, same edition.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Optional

NUM_RE = re.compile(r"(?<![\w.,])(\d{1,4}(?:[.,]\d{1,3})?)(?![\w]|[.,]\d)")
LATEX_RE = re.compile(r"\$[^$]*\$")
REF_RE = re.compile(r"(?:H|Hình|Bảng|Sơ đồ|Biểu đồ|Trang|tr\.|Bài|Câu|Chương|Phụ lục)\s*$", re.I)
WORD_RE = re.compile(r"\w+", re.UNICODE)
SENT_BREAK = set(".?!:;\n#-*>|(")


def _is_cap(w: str) -> bool:
    return len(w) >= 2 and w[0].isupper() and w[1:].islower() and not any(c.isdigit() for c in w)


def find_numbers(text: str) -> list[tuple[int, int, str, str]]:
    """[(start, end, token, kind)] for editable numbers/years, skipping figure/table refs, list numbers, LaTeX."""
    latex = [(m.start(), m.end()) for m in LATEX_RE.finditer(text)]
    out = []
    for m in NUM_RE.finditer(text):
        s, e = m.span(1)
        if any(a <= s < b for a, b in latex):
            continue
        if REF_RE.search(text[max(0, s - 12):s]):
            continue
        line_start = text.rfind("\n", 0, s) + 1
        if text[line_start:s].strip() == "" and e < len(text) and text[e:e + 1] in ".)":       # "1." list marker
            continue
        if text[max(0, s - 1):s] == "(" and text[e:e + 1] == ")":                 # "(2)" list / figure sub-label
            continue
        tok = m.group(1)
        kind = "year" if re.fullmatch(r"(1[0-9]{3}|20[0-9]{2})", tok) else "number"
        out.append((s, e, tok, kind))
    return out


def find_entities(text: str) -> list[tuple[int, int, str]]:
    """Capitalised multi-word sequences (Vietnamese proper names), not at a sentence start."""
    ws = [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]
    out, i = [], 0
    while i < len(ws):
        if _is_cap(ws[i][2]):
            j = i
            while j + 1 < len(ws) and _is_cap(ws[j + 1][2]) and text[ws[j][1]:ws[j + 1][0]].strip() == "":
                j += 1
            if j > i:
                before = text[:ws[i][0]].rstrip()
                if before and before[-1] not in SENT_BREAK and not before.endswith("**"):
                    out.append((ws[i][0], ws[j][1], text[ws[i][0]:ws[j][1]]))
            i = j + 1
        else:
            i += 1
    return out


def _swap_number(tok: str, kind: str, rng: random.Random) -> str:
    sep = "," if "," in tok else ("." if "." in tok else "")
    if sep:
        a, b = tok.split(sep)
        val = int(a + b)
        new = val
        while new == val:
            new = max(0, val + rng.choice([-1, 1]) * rng.randint(1, max(2, val // 4 or 2)))
        s = str(new).rjust(len(b) + 1, "0")
        return s[:-len(b)] + sep + s[-len(b):]
    v = int(tok)
    if kind == "year":
        return str(v + rng.choice([-1, 1]) * rng.randint(1, 9))
    new = v
    while new == v:
        new = max(0, v + rng.choice([-1, 1]) * rng.randint(1, max(2, v // 3 or 2)))
    return str(new).rjust(len(tok), "0") if tok.startswith("0") else str(new)


@dataclass
class EditionLog:
    seed: int
    deleted: dict[str, dict] = field(default_factory=dict)       # span_id -> {"origin": random|targeted}
    edited: dict[str, dict] = field(default_factory=dict)        # span_id -> {type, old, new, n_replaced, origin}
    stats: dict = field(default_factory=dict)


def _replace_all(text: str, old: str, new: str, typ: str) -> tuple[str, int]:
    if typ == "entity":
        pat = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)")
    else:
        pat = re.compile(r"(?<![\w.,])" + re.escape(old) + r"(?![\w]|[.,]\d)")
    return pat.subn(new, text)


def simulate_edition(rows: list[dict], *, seed: int = 0, delete_frac: float = 0.05, edit_frac: float = 0.05,
                     min_chars: int = 20, forced_delete: Optional[dict[str, str]] = None,
                     forced_edit: Optional[dict[str, str]] = None) -> tuple[list[dict], EditionLog]:
    """rows: span dicts (as in spans.jsonl / the spans table). `forced_*` add targeted changes on top of the
    random ones ({span_id: origin_tag} / {span_id: token to edit}); they are counted separately in the log."""
    rng = random.Random(seed)
    forced_delete = dict(forced_delete or {})
    forced_edit = dict(forced_edit or {})
    log = EditionLog(seed=seed)
    by_id = {r["span_id"]: r for r in rows}
    eligible = [r["span_id"] for r in rows if len((r.get("text") or "").strip()) >= min_chars]
    n_del = round(delete_frac * len(eligible))
    n_edit = round(edit_frac * len(eligible))
    pool = [s for s in eligible if s not in forced_delete and s not in forced_edit]
    dels = set(rng.sample(pool, n_del)) if n_del else set()
    # entity pool per book (for entity swaps)
    ents: dict[str, list[str]] = {}
    for r in rows:
        for _, _, e in find_entities(r.get("text") or ""):
            ents.setdefault(r["book_id"], []).append(e)
    ents = {b: sorted(set(v)) for b, v in ents.items()}
    editable = []
    for sid in pool:
        if sid in dels:
            continue
        t = by_id[sid]["text"]
        if find_numbers(t) or (find_entities(t) and len(ents.get(by_id[sid]["book_id"], [])) > 3):
            editable.append(sid)
    edits_pick = rng.sample(editable, min(n_edit, len(editable)))
    log.stats = {"eligible": len(eligible), "editable_pool": len(editable), "n_delete_random": len(dels),
                 "n_edit_random": len(edits_pick)}

    new_text: dict[str, str] = {}

    def do_edit(sid: str, origin: str, want: Optional[str] = None) -> bool:
        r = by_id[sid]
        t = r["text"]
        cands = [(s, e, tok, k) for s, e, tok, k in find_numbers(t)]
        ent = [(s, e, tok, "entity") for s, e, tok in find_entities(t)]
        if want:
            cands = [c for c in cands + ent if c[2] == want]
        else:
            cands = cands + [c for c in ent if len(ents.get(r["book_id"], [])) > 3]
        if not cands:
            return False
        s, e, tok, kind = cands[rng.randrange(len(cands))]
        if kind == "entity":
            others = [x for x in ents.get(r["book_id"], []) if x != tok and x not in t]
            if not others:
                return False
            new = rng.choice(others)
        else:
            new = _swap_number(tok, kind, rng)
        nt, n = _replace_all(t, tok, new, kind)
        if n == 0 or nt == t:
            return False
        new_text[sid] = nt
        log.edited[sid] = {"type": kind, "old": tok, "new": new, "n_replaced": n, "origin": origin}
        return True

    for sid in edits_pick:
        do_edit(sid, "random")
    for sid, tok in forced_edit.items():
        if sid in by_id and sid not in dels:
            do_edit(sid, "targeted", tok)
    for sid in dels:
        log.deleted[sid] = {"origin": "random"}
    for sid, tag in forced_delete.items():
        if sid in by_id:
            log.deleted[sid] = {"origin": "targeted"}
            new_text.pop(sid, None)
            log.edited.pop(sid, None)
    out = []
    for r in rows:
        sid = r["span_id"]
        if sid in log.deleted:
            continue
        if sid in new_text:
            r = dict(r)
            r["text"] = new_text[sid]
            r["text_sha1"] = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()
            if r.get("start_offset") is not None:
                r["end_offset"] = int(r["start_offset"]) + len(r["text"])
        out.append(r)
    log.stats.update({"n_delete": len(log.deleted), "n_edit": len(log.edited), "n_spans_new": len(out),
                      "edit_types": {t: sum(1 for v in log.edited.values() if v["type"] == t)
                                     for t in ("number", "year", "entity")}})
    return out, log
