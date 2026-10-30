"""Oracle reading check for ViEduHop-Multi: Qwen3-32B (:8004, T=0, short chain of thought, no thinking mode) answers each item with ALL gold spans in the prompt.
Shows that the keys are consistent with the evidence and measures the headroom when retrieval is perfect."""
import json, glob, re, urllib.request, concurrent.futures as cf, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
T = {}
for f in glob.glob(str(ROOT / "data/textbooks/parsed/nxbgd_*/spans.jsonl")):
    for l in open(f):
        s = json.loads(l); T[s["span_id"]] = s["text"]
items = [json.loads(l) for l in open(ROOT / "benchmarks/otter_hard/multihop/multihop_all.jsonl")]
def ask(it):
    ctx = "\n".join(f"[{i+1}] {T[s]}" for i, s in enumerate(it["gold_span_ids"]))
    opts = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
    p = f"Đoạn trích từ sách giáo khoa:\n{ctx}\n\nCâu hỏi: {it['question']}\n{opts}\n\nHãy suy luận ngắn gọn từng bước dựa trên các đoạn trích, rồi kết thúc bằng một dòng \"Đáp án: X\" với X là A, B, C hoặc D."
    body = {"model": "Qwen/Qwen3-32B", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 1200, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8004/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"].strip()
    m = re.findall(r"Đáp án\s*[:：]?\s*\**([ABCD])", out)
    return it["id"], it["hop_count"], (m[-1] if m else "?"), it["answer"]
with cf.ThreadPoolExecutor(3) as ex:
    res = list(ex.map(ask, items))
by = collections.defaultdict(list)
for i, h, p, k in res: by[int(h)].append(p == k)
tot = [p == k for _, _, p, k in res]
print("oracle accuracy all", round(sum(tot) / len(tot), 3), {h: (round(sum(v) / len(v), 3), len(v)) for h, v in sorted(by.items())})
print("wrong ids:", [i for i, h, p, k in res if p != k])
json.dump(res, open(ROOT / "runs/vieduhop_multihop/oracle_qwen3_32b_cot.json", "w"))
