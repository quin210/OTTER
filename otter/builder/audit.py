"""Human audit of extracted ontology items (annotation/ontology_audit/).

`make_audit`: stratified sample (kind x subject, equal allocation with redistribution) of concepts,
relations, facts and rules, each with its cited span text. Writes
  * ontology_audit.csv   - sheet to fill: judgment in {correct, partial, incorrect} + notes
  * ontology_audit.html  - self-contained page (same rows; dropdown + notes; autosave in the browser;
                           "Download CSV" exports the filled sheet)
  * ontology_audit_key.csv - automatic labels (support check, strict flag) kept OUT of the sheet so
                           annotators are blind to them
`score_audit`: precision from a filled sheet (strict: correct only; lenient: correct + partial), overall,
by kind, by subject, and (if the key file is next to the sheet) by automatic support label.
"""
from __future__ import annotations

import csv
import html
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from otter.builder.validate import REL_VI
from otter.storage.sqlite_store import SqliteOntologyStore
from otter.storage.writer import concept_text, rule_text

KINDS = (("concept", "concepts", "concept_id"), ("relation", "semantic_relations", "relation_id"),
         ("fact", "facts", "assertion_id"), ("rule", "rules", "rule_id"))
FIELDS = ["audit_id", "kind", "subject", "grade", "item_id", "statement", "details", "cited_span_ids",
          "cited_span_text", "judgment", "notes"]
JUDGMENTS = ("correct", "partial", "incorrect")
MAX_SPANS = 3
MAX_SPAN_CHARS = 1200

GUIDE_VI = (
    "Đánh giá từng mục so với (các) đoạn trích dẫn từ sách: "
    "<b>correct</b> = đúng và được đoạn trích hỗ trợ đầy đủ (với rule: đủ tiền đề, điều kiện, kết luận; "
    "với concept: là thuật ngữ của môn học, định nghĩa/loại đúng); "
    "<b>partial</b> = đúng một phần (thiếu điều kiện, quá khái quát, sai loại khái niệm/quan hệ nhưng nội dung đúng, "
    "lỗi nhỏ); <b>incorrect</b> = sai, không có trong đoạn trích, hoặc không phải tri thức (rác OCR, từ thông thường). "
    "Chỉ dựa vào đoạn trích, không dùng kiến thức bên ngoài. Ghi chú lí do khi chọn partial/incorrect.")
GUIDE_EN = ("Judge each item against its cited textbook span(s) only: correct = fully supported and well-formed; "
            "partial = partly right (missing guard, over-generalised, wrong type but right content); incorrect = "
            "wrong / not in the span / not knowledge. Automatic support labels are hidden on purpose.")


def _page(span) -> str:
    p = f"PDF p.{span.pdf_page_index + 1}"
    if span.printed_page_verified and span.printed_page_label:
        p += f", trang in {span.printed_page_label}"
    return p


def _describe(store: SqliteOntologyStore, kind: str, iid: str) -> tuple[Any, str, str, Optional[int], str]:
    if kind == "concept":
        c = store.get_concept(iid)
        return c, c.subject, concept_text(c), c.grade, f"type={c.concept_type}"
    if kind == "relation":
        r = store.get_relation(iid)
        s, o = store.get_concept(r.subject_id), store.get_concept(r.object_id)
        st = f"{s.label if s else r.subject_id} [{r.relation_type}: {REL_VI.get(r.relation_type, '')}] " \
             f"{o.label if o else r.object_id}" + (f" ({r.qualifier})" if r.qualifier else "")
        return r, (s.subject if s else "other"), st, None, f"status={r.status}"
    if kind == "fact":
        f = store.get_fact(iid)
        return f, f.subject, f.text, None, f"status={f.status}"
    r = store.get_rule(iid)
    det = {"rule_type": r.rule_type, "status": r.status,
           "premises": [("NOT " if a.negated else "") + a.text for a in r.premises],
           "guards": [("NOT " if a.negated else "") + a.text for a in r.guards],
           "conclusions": [("NOT " if a.negated else "") + a.text for a in r.conclusions]}
    if r.exceptions:
        det["exceptions"] = [a.text for a in r.exceptions]
    if r.formula:
        det["formula"], det["variables"] = r.formula, r.variables
    return r, r.subject, rule_text(r), None, json.dumps(det, ensure_ascii=False)


def make_audit(onto_path: Path, prov_path: Optional[Path], out_dir: Path, *, n: int = 150, seed: int = 2027
               ) -> dict[str, Any]:
    store = SqliteOntologyStore(onto_path, provenance_path=prov_path, use_dense=False)
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for kind, table, pk in KINDS:
        col = "subject"
        for iid, subj in store.db.execute(f"SELECT {pk}, {col} FROM {table} ORDER BY {pk}"):
            cells[(kind, subj or "other")].append(iid)
    keys = sorted(cells)
    quota = {k: 0 for k in keys}
    remaining = n
    open_keys = [k for k in keys if cells[k]]
    while remaining > 0 and open_keys:          # equal allocation, redistribute leftovers
        share = max(1, remaining // len(open_keys))
        nxt = []
        for k in open_keys:
            if remaining <= 0:
                break
            take = min(share, len(cells[k]) - quota[k], remaining)
            quota[k] += take
            remaining -= take
            if quota[k] < len(cells[k]):
                nxt.append(k)
        open_keys = nxt
    rows, key_rows = [], []
    for k in keys:
        for iid in sorted(rng.sample(cells[k], quota[k])):
            obj, subj, statement, grade, details = _describe(store, k[0], iid)
            sids = list(obj.source_span_ids)[:MAX_SPANS]
            spans = store.get_spans(sids)
            ev = "\n---\n".join(f"[{s.span_id} | {_page(s)}] {s.text[:MAX_SPAN_CHARS]}" for s in spans)
            if grade is None:
                g = store.db.execute(f"SELECT grade FROM {dict((a, b) for a, b, _ in KINDS)[k[0]]} WHERE "
                                     f"{dict((a, c) for a, _, c in KINDS)[k[0]]}=?", (iid,)).fetchone()
                grade = g[0] if g else None
            rows.append({"kind": k[0], "subject": subj, "grade": grade if grade is not None else "", "item_id": iid,
                         "statement": statement, "details": details, "cited_span_ids": " ".join(sids),
                         "cited_span_text": ev, "judgment": "", "notes": ""})
            v = obj.validation
            key_rows.append({"item_id": iid, "kind": k[0], "subject": subj, "source_support": v.source_support,
                             "support_score": v.support_score, "status": getattr(obj, "status", ""),
                             "strict_executable": store.is_strict_executable(obj) if k[0] == "rule" else "",
                             "validation_notes": ";".join(v.notes)})
    order = list(range(len(rows)))
    rng.shuffle(order)                                  # interleave strata so annotators see a mix
    rows = [rows[i] for i in order]
    key_rows = [key_rows[i] for i in order]
    for i, r in enumerate(rows, 1):
        r["audit_id"] = f"A{i:03d}"
        key_rows[i - 1]["audit_id"] = r["audit_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ontology_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "ontology_audit_key.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audit_id", *[k for k in key_rows[0] if k != "audit_id"]] if key_rows
                           else ["audit_id"])
        w.writeheader()
        w.writerows(key_rows)
    build = store.build_info() or {}
    (out_dir / "ontology_audit.html").write_text(render_html(rows, build.get("build_id", "?")), encoding="utf-8")
    strata = Counter((r["kind"], r["subject"]) for r in rows)
    meta = {"n": len(rows), "seed": seed, "build_id": build.get("build_id"), "onto": str(onto_path),
            "strata": {f"{k}|{s}": c for (k, s), c in sorted(strata.items())},
            "population": {f"{k}|{s}": len(v) for (k, s), v in sorted(cells.items())}}
    (out_dir / "sample_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def render_html(rows: list[dict], build_id: str) -> str:
    e = html.escape
    trs = []
    for r in rows:
        opts = "".join(f'<option value="{j}">{j}</option>' for j in ("", *JUDGMENTS))
        trs.append(
            f'<tr data-id="{e(r["audit_id"])}"><td class="id">{e(r["audit_id"])}<br><span class="tag">{e(r["kind"])}'
            f'</span><br><span class="tag">{e(r["subject"])} {e(str(r["grade"]))}</span></td>'
            f'<td><div class="st">{e(r["statement"])}</div><div class="det">{e(r["details"])}</div>'
            f'<div class="iid">{e(r["item_id"])}</div></td>'
            f'<td class="ev">{e(r["cited_span_text"])}</td>'
            f'<td><select class="j">{opts}</select><textarea class="n" rows="3" placeholder="notes"></textarea></td></tr>')
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ontology Audit Sheet</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#ddd;--hl:#f6f7f9;--acc:#2a5db0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16181c;--fg:#e8e8ea;--muted:#a0a0a8;--line:#333;--hl:#1f2228;--acc:#7aa7ff}}}}
:root[data-theme="dark"]{{--bg:#16181c;--fg:#e8e8ea;--muted:#a0a0a8;--line:#333;--hl:#1f2228;--acc:#7aa7ff}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 16px 40px}}
h1{{font-size:20px}} .guide{{background:var(--hl);padding:10px 12px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;margin-top:12px}} td{{border-top:1px solid var(--line);vertical-align:top;padding:8px 6px}}
td.id{{white-space:nowrap;color:var(--muted)}} .tag{{font-size:12px}} .st{{font-weight:600}}
.det,.iid{{color:var(--muted);font-size:12px;word-break:break-word}} td.ev{{white-space:pre-wrap;font-size:13px;max-width:520px}}
select,textarea{{width:100%;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--line)}}
button{{padding:6px 12px;margin-right:8px}} #prog{{color:var(--acc);font-weight:600}}
@media (max-width:760px){{table,tbody,tr,td{{display:block;width:100%}} td.ev{{max-width:none}}}}
</style></head><body>
<h1>OTTER ontology audit — build {e(build_id)}</h1>
<p class="guide">{GUIDE_VI}<br><i>{GUIDE_EN}</i></p>
<p><button id="dl">Download CSV</button><span id="prog"></span></p>
<table><tbody>{''.join(trs)}</tbody></table>
<script>
const ROWS={data};
const KEY="otter-onto-audit-{e(build_id)}";
let saved={{}}; try{{saved=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(_){{saved={{}}}}
function prog(){{const n=document.querySelectorAll("select.j");let d=0;n.forEach(s=>{{if(s.value)d++}});document.getElementById("prog").textContent=d+" / "+n.length+" judged";}}
document.querySelectorAll("tr[data-id]").forEach(tr=>{{const id=tr.dataset.id,s=tr.querySelector("select"),t=tr.querySelector("textarea");
 if(saved[id]){{s.value=saved[id].j||"";t.value=saved[id].n||"";}}
 const save=()=>{{saved[id]={{j:s.value,n:t.value}};try{{localStorage.setItem(KEY,JSON.stringify(saved))}}catch(_){{}};prog();}};
 s.addEventListener("change",save);t.addEventListener("input",save);}});
prog();
const F={json.dumps(FIELDS)};
function q(v){{v=(v==null?"":String(v));return '"'+v.replace(/"/g,'""')+'"';}}
document.getElementById("dl").onclick=()=>{{const lines=[F.join(",")];
 ROWS.forEach(r=>{{const s=saved[r.audit_id]||{{}};const o=Object.assign({{}},r,{{judgment:s.j||"",notes:s.n||""}});lines.push(F.map(k=>q(o[k])).join(","));}});
 const b=new Blob(["\\ufeff"+lines.join("\\n")],{{type:"text/csv"}});const a=document.createElement("a");
 a.href=URL.createObjectURL(b);a.download="ontology_audit_filled.csv";a.click();}};
</script></body></html>
"""


def _prec(js: list[str]) -> dict[str, Any]:
    c = Counter(js)
    n = sum(c[j] for j in JUDGMENTS)
    if n == 0:
        return {"n_judged": 0, "precision_strict": None, "precision_lenient": None}
    ps = c["correct"] / n
    pl = (c["correct"] + c["partial"]) / n
    half = 1.96 * math.sqrt(ps * (1 - ps) / n)
    return {"n_judged": n, **{j: c[j] for j in JUDGMENTS}, "precision_strict": round(ps, 4),
            "precision_lenient": round(pl, 4), "strict_95ci_normal": [round(max(0, ps - half), 4), round(min(1, ps + half), 4)],
            "weighted_score": round((c["correct"] + 0.5 * c["partial"]) / n, 4)}


def score_audit(sheet: Path, key: Optional[Path] = None) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(sheet).open(encoding="utf-8-sig")))
    for r in rows:
        r["judgment"] = (r.get("judgment") or "").strip().lower()
    bad = [r["audit_id"] for r in rows if r["judgment"] and r["judgment"] not in JUDGMENTS]
    judged = [r for r in rows if r["judgment"] in JUDGMENTS]
    res: dict[str, Any] = {"sheet": str(sheet), "rows": len(rows), "invalid_judgments": bad,
                           "overall": _prec([r["judgment"] for r in judged])}
    for dim in ("kind", "subject"):
        g = defaultdict(list)
        for r in judged:
            g[r[dim]].append(r["judgment"])
        res[f"by_{dim}"] = {k: _prec(v) for k, v in sorted(g.items())}
    key = key or Path(sheet).with_name("ontology_audit_key.csv")
    if key.exists():
        km = {r["audit_id"]: r for r in csv.DictReader(key.open(encoding="utf-8"))}
        g = defaultdict(list)
        for r in judged:
            if r["audit_id"] in km:
                g[km[r["audit_id"]]["source_support"]].append(r["judgment"])
        res["by_auto_support"] = {k: _prec(v) for k, v in sorted(g.items())}
        sr = [r["judgment"] for r in judged if km.get(r["audit_id"], {}).get("strict_executable") == "True"]
        res["strict_executable_rules"] = _prec(sr)
    return res
