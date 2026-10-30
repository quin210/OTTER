"""Human annotation sheets: CSV (for spreadsheets) + one self-contained HTML file (no network).

The HTML renders one card per item with the read-only context blocks and editable fields,
autosaves to the browser's localStorage and exports the filled answers as CSV/JSON. Field
names in the export equal the CSV's blank columns so either route can be merged back with
`read_filled_csv`.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Optional

# field spec: {"name": str, "label": str, "type": "choice"|"text"|"bool", "options": [..]}


def write_csv(rows: list[dict[str, Any]], context_cols: list[str], fields: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = context_cols + [f["name"] for f in fields]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # BOM so Excel opens Vietnamese correctly
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {c: _cell(r.get(c)) for c in context_cols}
            for fl in fields:
                out[fl["name"]] = r.get(fl["name"], "")
            w.writerow(out)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def read_filled_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


_CSS = """
:root{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e1da;--acc:#2f6f4f;--warn:#a65b00;--gold:#fff6d6}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;--card:#201f1d;--line:#34332f;--acc:#7cc4a0;--warn:#e0a050;--gold:#3a3420}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
header h1{font-size:17px;margin:0 12px 0 0}header .sp{flex:1}button,select,input[type=text]{font:inherit}
button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:980px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 16px}
.card.done{border-left:4px solid var(--acc)}.card h2{font-size:15px;margin:0 0 8px;color:var(--muted);font-weight:600}
.blk{margin:6px 0}.blk .k{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.blk .v{white-space:pre-wrap;word-wrap:break-word}.opt{margin:2px 0}.opt.ans{background:var(--gold);border-radius:4px;padding:0 4px}
.span{border-left:3px solid var(--line);padding:4px 10px;margin:6px 0;font-size:14px}.span .m{font-size:12px;color:var(--muted)}
.fields{display:grid;grid-template-columns:minmax(160px,240px) 1fr;gap:6px 12px;margin-top:10px;border-top:1px dashed var(--line);padding-top:10px}
.fields label{font-weight:600;font-size:14px}.fields input[type=text],.fields textarea{width:100%;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.radios label{font-weight:400;margin-right:12px;white-space:nowrap}
#prog{color:var(--muted)}.hidden{display:none}
@media (max-width:640px){.fields{grid-template-columns:1fr}}
"""

_JS = """
const KEY='otter-sheet:'+SHEET.sheet_id;
let state={};try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){state={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(state))}catch(e){}prog()}
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function done(id){const s=state[id]||{};return SHEET.fields.filter(f=>f.required!==false&&f.type!=='text').every(f=>s[f.name]!==undefined&&s[f.name]!=='')}
function prog(){const n=SHEET.rows.length,d=SHEET.rows.filter(r=>done(r[SHEET.id_col])).length;document.getElementById('prog').textContent=d+' / '+n+' done';
 SHEET.rows.forEach(r=>{const el=document.getElementById('card-'+r[SHEET.id_col]);if(el)el.classList.toggle('done',done(r[SHEET.id_col]))})}
function render(){const m=document.querySelector('main');m.innerHTML='';
 SHEET.rows.forEach((r,i)=>{const id=r[SHEET.id_col];const c=document.createElement('div');c.className='card';c.id='card-'+id;
  let h='<h2>#'+(i+1)+' · '+esc(id)+'</h2>';
  (r.__blocks||[]).forEach(b=>{if(b.kind==='options'){h+='<div class="blk"><div class="k">'+esc(b.label)+'</div>';
     Object.entries(b.value).forEach(([k,v])=>{h+='<div class="opt'+(k===b.answer?' ans':'')+'"><b>'+esc(k)+'.</b> '+esc(v)+(k===b.answer?' ✓':'')+'</div>'});h+='</div>'}
   else if(b.kind==='spans'){h+='<div class="blk"><div class="k">'+esc(b.label)+'</div>';
     (b.value||[]).forEach(s=>{h+='<div class="span"><div class="m">'+esc(s.meta)+'</div>'+esc(s.text)+'</div>'});if(!(b.value||[]).length)h+='<div class="v">—</div>';h+='</div>'}
   else{h+='<div class="blk"><div class="k">'+esc(b.label)+'</div><div class="v">'+esc(b.value)+'</div></div>'}});
  h+='<div class="fields">';const s=state[id]||{};
  SHEET.fields.forEach(f=>{h+='<label>'+esc(f.label)+'</label><div class="'+(f.type==='choice'?'radios':'')+'">';
   if(f.type==='choice'){f.options.forEach(o=>{h+='<label><input type="radio" name="'+esc(id+'::'+f.name)+'" value="'+esc(o)+'" '+(s[f.name]===o?'checked':'')+'> '+esc(o)+'</label>'})}
   else{h+='<input type="text" data-f="'+esc(f.name)+'" value="'+esc(s[f.name]||'')+'" placeholder="'+esc(f.placeholder||'')+'">'}
   h+='</div>'});h+='</div>';c.innerHTML=h;m.appendChild(c);
  c.querySelectorAll('input[type=radio]').forEach(el=>el.addEventListener('change',e=>{const [rid,fn]=e.target.name.split('::');(state[rid]=state[rid]||{})[fn]=e.target.value;save()}));
  c.querySelectorAll('input[type=text]').forEach(el=>el.addEventListener('input',e=>{(state[id]=state[id]||{})[e.target.dataset.f]=e.target.value;save()}))});
 prog();filt()}
function filt(){const v=document.getElementById('flt').value;SHEET.rows.forEach(r=>{const el=document.getElementById('card-'+r[SHEET.id_col]);const d=done(r[SHEET.id_col]);el.classList.toggle('hidden',(v==='todo'&&d)||(v==='done'&&!d))})}
function csvq(v){v=String(v??'');return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
function dl(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click()}
function exportCSV(){const cols=[SHEET.id_col,'annotator'].concat(SHEET.fields.map(f=>f.name));const ann=document.getElementById('ann').value;
 const lines=[cols.join(',')];SHEET.rows.forEach(r=>{const id=r[SHEET.id_col],s=state[id]||{};lines.push([id,ann].concat(SHEET.fields.map(f=>s[f.name]||'')).map(csvq).join(','))});
 dl(SHEET.sheet_id+'_filled.csv','\\ufeff'+lines.join('\\n'),'text/csv')}
function exportJSON(){dl(SHEET.sheet_id+'_filled.json',JSON.stringify({sheet_id:SHEET.sheet_id,annotator:document.getElementById('ann').value,answers:state},null,1),'application/json')}
function importJSON(ev){const f=ev.target.files[0];if(!f)return;f.text().then(t=>{const o=JSON.parse(t);state=Object.assign(state,o.answers||{});save();render()})}
document.addEventListener('DOMContentLoaded',()=>{try{document.getElementById('ann').value=localStorage.getItem(KEY+':ann')||''}catch(e){}
 document.getElementById('ann').addEventListener('input',e=>{try{localStorage.setItem(KEY+':ann',e.target.value)}catch(_){}});render()});
"""


def write_html(rows: list[dict[str, Any]], fields: list[dict], path: Path, *, title: str, sheet_id: str,
               id_col: str = "id", instructions: str = "") -> None:
    """rows: each has `id_col` and `__blocks`: [{"kind":"text|options|spans","label",...}]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"sheet_id": sheet_id, "id_col": id_col, "fields": fields, "rows": rows},
                         ensure_ascii=False).replace("</", "<\\/")
    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1><span id="prog"></span><span class="sp"></span>
<input type="text" id="ann" placeholder="annotator name" size="14">
<select id="flt" onchange="filt()"><option value="all">all</option><option value="todo">to do</option><option value="done">done</option></select>
<button class="primary" onclick="exportCSV()">Export CSV</button><button onclick="exportJSON()">Export JSON</button>
<label><button onclick="document.getElementById('imp').click()">Import JSON</button><input id="imp" type="file" accept=".json" class="hidden" onchange="importJSON(event)"></label>
</header>
<main></main>
<div style="max-width:980px;margin:0 auto 32px;padding:0 16px;color:var(--muted);font-size:13px;white-space:pre-wrap">{html.escape(instructions)}</div>
<script>const SHEET={payload};</script><script>{_JS}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def span_block(label: str, spans: list, *, max_chars: int = 1500) -> dict:
    """spans: corpus Span objects (or None for unknown ids)."""
    vals = []
    for s in spans:
        if s is None:
            continue
        page = f"pdf p.{s.pdf_page_index + 1}" if s.pdf_page_index is not None else "pdf p.?"
        printed = f", printed p.{s.printed_page_label}" if s.printed_page_verified and s.printed_page_label else ""
        hp = " › ".join(s.heading_path[-3:]) if s.heading_path else (s.section_id or "")
        vals.append({"meta": f"{s.span_id} · {s.book_title or s.book_id} · {page}{printed} · {hp}",
                     "text": s.text[:max_chars] + ("…" if len(s.text) > max_chars else "")})
    return {"kind": "spans", "label": label, "value": vals}


def write_sheet(rows: list[dict], *, context_cols: list[str], fields: list[dict], out_dir: Path, name: str,
                title: str, instructions: str = "", id_col: str = "id") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    csv_p, html_p = out_dir / f"{name}.csv", out_dir / f"{name}.html"
    write_csv(rows, context_cols, fields, csv_p)
    write_html(rows, fields, html_p, title=title, sheet_id=name, id_col=id_col, instructions=instructions)
    if instructions:
        (out_dir / "README.md").write_text(f"# {title}\n\n{instructions}\n\nFiles: `{csv_p.name}` (spreadsheet), "
                                           f"`{html_p.name}` (open in a browser; autosaves locally; Export CSV when done).\n",
                                           encoding="utf-8")
    return csv_p, html_p
