"""Oracle reading check: Qwen3-32B (T=0, short chain of thought) answers each item with ALL gold spans in the prompt.
  .venv/bin/python scripts/eval/oracle_check.py --in a.jsonl [--in b.jsonl ...] --out runs/oracle/x.json [--workers 6]
A correct oracle answer means the key is consistent with the evidence; wrong or unparsable items are listed for review."""
import argparse, json, glob, re, urllib.request, concurrent.futures as cf, collections, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
T = {}
for f in glob.glob(str(ROOT / "data/textbooks/parsed/nxbgd_*/spans.jsonl")):
    for l in open(f):
        s = json.loads(l); T[s["span_id"]] = s["text"]
items = [json.loads(l) for p in a.inp for l in open(ROOT / p) if l.strip()]
URLS = itertools.cycle(["http://127.0.0.1:8004/v1/chat/completions", "http://127.0.0.1:8005/v1/chat/completions"])
def ask(it, url):
    ctx = "\n".join(f"[{i+1}] {T[s]}" for i, s in enumerate(it["gold_span_ids"]) if s in T)
    opts = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
    p = f"Đoạn trích từ sách giáo khoa:\n{ctx}\n\nCâu hỏi: {it['question']}\n{opts}\n\nHãy suy luận ngắn gọn từng bước dựa trên các đoạn trích, rồi kết thúc bằng một dòng \"Đáp án: X\" với X là A, B, C hoặc D."
    body = {"model": "Qwen/Qwen3-32B", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 1800, "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(2):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
            m = re.findall(r"Đáp án\s*[:：]?\s*\**\s*([ABCD])", out)
            if m: return it["id"], m[-1], it["answer"]
        except Exception:
            pass
    return it["id"], "?", it["answer"]
with cf.ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(lambda it: ask(it, next(URLS)), items))
(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(ROOT / a.out, "w"))
ok = sum(1 for _, p, k in res if p == k); un = sum(1 for _, p, _ in res if p == "?")
print(f"oracle accuracy {ok}/{len(res)} = {ok/len(res):.3f}; unparsable {un}")
print("wrong:", [i for i, p, k in res if p != k and p != "?"]); print("unparsable:", [i for i, p, _ in res if p == "?"])
