"""ViEduHop-Para generation + acceptance pipeline (resumable).
  python pipeline.py gen   --src <items.jsonl> --ids <ids.json|-> --out <dir>
  python pipeline.py check --src <items.jsonl> --out <dir>
Writes <out>/gen.jsonl (raw candidates), <out>/checks.jsonl (all check details, raw judge/answer outputs), <out>/final.jsonl (chosen candidate per item or null).
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from para_lib import *  # noqa
import prompts as PR
from llm_pool import chat
from otter.llm.interface import parse_json_loose

THR_ABS, THR_REL = 0.35, 0.70   # fixed in docs/VIEDUHOP_PARA.md
_lock = threading.Lock()


def append(path, rec):
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_keys(path, keyf):
    p = Path(path)
    return {keyf(r) for r in jl(p)} if p.exists() else set()


def fmt_opts(o):
    return {k: o[k] for k in LETTERS}


# ------------------------------------------------------------------ generation
def gen_one(it, cand, out):
    import os
    gt = "(không cung cấp)" if os.environ.get("PARA_NOSPAN") == "1" else gold_text(it["gold_span_ids"])
    u = PR.GEN_USER.format(question=it["question"], key=it["answer"], gold=gt, **fmt_opts(it["options"]))
    r = chat([{"role": "system", "content": PR.GEN_SYSTEM}, {"role": "user", "content": u}], temperature=0.7, max_tokens=1200)
    parsed, err = None, None
    try:
        parsed = parse_json_loose(r.text)
    except Exception as e:  # noqa
        err = str(e)[:200]
    append(out / "gen.jsonl", {"id": it["id"], "cand": cand, "raw": r.text, "parsed": parsed, "parse_error": err,
                               "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms})


def cmd_gen(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = {r["id"]: r for r in jl(a.src)}
    ids = json.load(open(a.ids))["ids"] if a.ids != "-" else list(items)
    have = done_keys(out / "gen.jsonl", lambda r: (r["id"], r["cand"]))
    todo = [(items[i], c) for i in ids for c in (1, 2) if (i, c) not in have]
    print(f"gen: {len(todo)} calls", flush=True)
    with ThreadPoolExecutor(3) as ex:
        for k, _ in enumerate(ex.map(lambda t: gen_one(t[0], t[1], out), todo), 1):
            if k % 40 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


# ------------------------------------------------------------------ checks
def protected_strings(text):
    """Standalone numbers (digits only, separators/decimal marks removed; the grade mention 'lớp N' is boilerplate and exempt) and formula-like tokens (letters+digits, case-sensitive)."""
    t = re.sub(r"lớp\s*\d+", " ", nfc(text), flags=re.I)
    nums = {re.sub(r"[^\d]", "", m) for m in re.findall(r"(?<!\w)\d+(?:[.,]\d+|\s\d{3}(?!\d))*(?!\w)", t)}
    forms = {re.sub(r"\s", "", w) for w in re.findall(r"\b(?=\w*[^\W\d_])(?=\w*\d)\w+\b", t)}
    return nums | forms


def unprotected(text):
    return [t for t, prot in entity_mask(text) if not prot and t not in VI_STOPWORDS]


def check_a(it, p):
    if not isinstance(p, dict) or not isinstance(p.get("options"), dict) or not isinstance(p.get("question"), str):
        return False, "bad_json_shape"
    o = p["options"]
    if set(o) != set(LETTERS) or any(not isinstance(o[k], str) or not o[k].strip() for k in LETTERS) or not p["question"].strip():
        return False, "options_not_ABCD_or_empty"
    norm = lambda s: " ".join(syllables(s))
    if norm(p["question"]) == norm(it["question"]):
        return False, "unchanged_stem"
    if any(norm(o[k]) == norm(it["options"][k]) and len(unprotected(it["options"][k])) >= 2 for k in LETTERS):
        return False, "unchanged_option"
    if len({norm(o[k]) for k in LETTERS}) < 4:
        return False, "duplicate_options"
    protected = protected_strings(it["question"] + " " + " ".join(it["options"][k] for k in LETTERS))
    ptxt = p["question"] + " " + " ".join(o[k] for k in LETTERS)
    have = protected_strings(ptxt)
    missing = [x for x in protected if x not in have]
    if missing:
        return False, f"numbers_missing:{sorted(set(missing))[:3]}"
    gt = gold_text(it["gold_span_ids"])
    cp = copied_trigrams(p["question"], gt)
    for k in LETTERS:
        cp += copied_trigrams(o[k], gt)
    if cp:
        return False, f"copied_trigram:{' '.join(cp[0])}"
    return True, "ok"


def check_b(it, p):
    gt = gold_text(it["gold_span_ids"])
    ob = overlap(mcq_text(it["question"], it["options"]), gt)
    pb = overlap(mcq_text(p["question"], p["options"]), gt)
    ok = pb <= THR_ABS and pb <= THR_REL * ob
    return ok, {"overlap_orig": ob, "overlap_para": pb}


def check_c(it, p):
    gt = gold_text(it["gold_span_ids"])
    o, po = it["options"], p["options"]
    u = PR.JUDGE_USER.format(key=it["answer"], gold=gt, oq=it["question"], pq=p["question"],
                             **{f"o{k}": o[k] for k in LETTERS}, **{f"p{k}": po[k] for k in LETTERS})
    r = chat([{"role": "system", "content": PR.JUDGE_SYSTEM}, {"role": "user", "content": u}], temperature=0.0, max_tokens=500)
    try:
        j = parse_json_loose(r.text)
    except Exception:
        return False, {"raw": r.text, "error": "unparsable"}
    dsc = j.get("distractor_same_claim")
    dsc_ok = (all(v is True for v in dsc.values()) and len(dsc) >= 3) if isinstance(dsc, dict) else dsc is True
    ok = (j.get("stem_equivalent") is True and j.get("key_equivalent") is True and dsc_ok
          and j.get("distractors_still_wrong") is True and j.get("adds_information") is False)
    return ok, {"raw": r.text, "judge": j}


def shuffled(it_q, opts, key, seed):
    """Return (options dict in new order, new key letter)."""
    rng = random.Random(seed)
    perm = list(LETTERS); rng.shuffle(perm)          # perm[i] = original letter shown at position i
    new = {LETTERS[i]: opts[perm[i]] for i in range(4)}
    return new, LETTERS[perm.index(key)]


def pick(text):
    m = re.search(r"\b([ABCD])\b", text or "")
    return m.group(1) if m else None


def mc_probe(question, opts, key, seed_base, n, gold=None):
    res = []
    for s in range(n):
        seed = int(hashlib.sha256(f"{seed_base}|{s}".encode()).hexdigest()[:8], 16)
        so, sk = shuffled(question, opts, key, seed)
        if gold is not None:
            u, sysm = PR.ANS_USER.format(gold=gold, question=question, **so), PR.ANS_SYSTEM
        else:
            u, sysm = PR.CB_USER.format(question=question, **so), PR.CB_SYSTEM
        r = chat([{"role": "system", "content": sysm}, {"role": "user", "content": u}], temperature=0.0, max_tokens=8)
        res.append({"seed": seed, "key_shuffled": sk, "raw": r.text, "pick": pick(r.text), "correct": pick(r.text) == sk})
    return res


def check_d(it, p):
    gt = gold_text(it["gold_span_ids"])
    res = mc_probe(p["question"], p["options"], it["answer"], f"ans|{it['id']}|{p['question']}", 3, gold=gt)
    n = sum(x["correct"] for x in res)
    return n >= 2, {"samples": res, "n_correct": n}


def check_item(it, cands, out):
    rec = {"id": it["id"], "cands": []}
    chosen = None
    for c in cands:
        p = c["parsed"]
        cr = {"cand": c["cand"], "stage_reached": "gen"}
        ok, info = (False, "parse_failed:" + str(c.get("parse_error"))) if p is None else check_a(it, p)
        cr["a"] = {"pass": ok, "info": info}
        if ok:
            cr["stage_reached"] = "a"
            ok, info = check_b(it, p); cr["b"] = {"pass": ok, "info": info}
            if ok:
                cr["stage_reached"] = "b"
                ok, info = check_c(it, p); cr["c"] = {"pass": ok, "info": info}
                if ok:
                    cr["stage_reached"] = "c"
                    ok, info = check_d(it, p); cr["d"] = {"pass": ok, "info": info}
                    if ok:
                        cr["stage_reached"] = "d"
        cr["passed_all"] = cr["stage_reached"] == "d"
        rec["cands"].append(cr)
        if cr["passed_all"]:
            chosen = c["cand"]
            break
    rec["chosen"] = chosen
    if chosen is not None:
        p = next(c["parsed"] for c in cands if c["cand"] == chosen)
        cb = mc_probe(p["question"], p["options"], it["answer"], f"cb|{it['id']}|{p['question']}", 4)
        rec["closed_book_probe"] = {"n": 4, "correct": sum(x["correct"] for x in cb), "samples": cb}
    append(out / "checks.jsonl", rec)


def cmd_check(a):
    out = Path(a.out)
    items = {r["id"]: r for r in jl(a.src)}
    gens = {}
    for r in jl(out / "gen.jsonl"):
        gens.setdefault(r["id"], []).append(r)
    have = done_keys(out / "checks.jsonl", lambda r: r["id"])
    todo = [i for i in gens if i not in have and len(gens[i]) >= 2]
    print(f"check: {len(todo)} items", flush=True)
    def one(i):
        check_item(items[i], sorted(gens[i], key=lambda c: c["cand"]), out)
    with ThreadPoolExecutor(3) as ex:
        for k, _ in enumerate(ex.map(one, todo), 1):
            if k % 20 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "check"]); ap.add_argument("--src", required=True); ap.add_argument("--ids", default="-"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"gen": cmd_gen, "check": cmd_check}[a.cmd](a)
