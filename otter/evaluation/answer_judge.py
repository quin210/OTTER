"""LLM-as-a-Judge for final answers (rubric `judge_v2`; full text of the prompts in docs/LLM_JUDGE.md).

Per (system, item) the judge sees the question and options, the benchmark's reference (key, reference answer, explanation,
gold quotes), the system's final answer text with its claims, and the TEXT of the spans the answer cites (resolved from the
span corpus, never the system's own `quote` field). The system name is hidden. Output (JSON, schema in the prompt, parsed
loosely, one retry):  correctness 0/1/2, completeness 0/1/2, grounding 0/1/2, reason.

    judge_score  = (correctness + completeness + grounding) / 6          (0..1)
    ans_score    = (correctness + completeness) / 4                      (answer-only; comparable for systems without citations)
    strict       = correctness == completeness == grounding == 2         ("fully correct and fully grounded")

Deterministic rules applied on top of the judge (raw judge values are kept as `*_raw`; every override is listed in `rules`):
  * failed / missing answer                                  -> all zero, no call
  * answerable item, chosen letter != key (or E / abstain)   -> correctness 0
  * unanswerable item (OTTER-Abstain), system abstains       -> correctness 2; commits to A-D -> correctness 0
  * answer is not an abstention and no cited span text found -> grounding 0   (an uncited answer is not grounded)
Cache key: (judge_version, question id, sha1 of the exact prompt). The prompt contains the item, the answer and the span
texts, so any change of the answer, the benchmark item or the template invalidates the cache entry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

from otter.evaluation.stats import bootstrap_ci
from otter.llm.interface import LLM, parse_json_loose
from otter.schemas.reasoning import FinalAnswer

JUDGE_VERSION = os.environ.get("OTTER_JUDGE_VERSION", "judge_v2")      # override only to re-read archived results of an older rubric
DIMS = ("correctness", "completeness", "grounding")
SCHEMA = {"type": "object",
          "properties": {"reason": {"type": "string"},
                         "correctness": {"type": "integer", "enum": [0, 1, 2]},
                         "completeness": {"type": "integer", "enum": [0, 1, 2]},
                         "grounding": {"type": "integer", "enum": [0, 1, 2]}},
          "required": ["reason", "correctness", "completeness", "grounding"]}

# judge id -> config role of configs/default.yaml (the served checkpoints are documented in runs/SERVERS.md)
JUDGES = {"qwen3-32b": {"role": "main", "label": "Qwen3-32B (FP8)", "short": "32B"},
          "qwen3-8b": {"role": "small", "label": "Qwen3-8B", "short": "8B"}}
MAX_SPANS = 8
MAX_SPAN_CHARS = 1200
MAX_GOLD_QUOTES = 3
MAX_ANSWER_CHARS = 2500

ABSTAIN_TYPE_VI = {
    "out_of_corpus": "bài/nội dung được hỏi không có trong phần sách giáo khoa được cung cấp",
    "missing_premise": "câu hỏi cần một tiền đề/đoạn sách quan trọng đã bị loại khỏi nguồn, nên không thể suy ra đáp án",
    "counterfactual_guard": "câu hỏi nêu điều kiện không được xác định hoặc không thỏa mãn quy tắc trong sách, nên không thể suy ra đáp án",
}

PROMPT_TEMPLATE = """Bạn là giám khảo chấm câu trả lời của một hệ thống hỏi–đáp dựa trên sách giáo khoa phổ thông Việt Nam (lớp 10–12). \
Bạn không biết hệ thống nào tạo ra câu trả lời. Hãy chấm câu trả lời theo ba tiêu chí, mỗi tiêu chí theo thang 0/1/2, \
chỉ dựa vào thông tin được cung cấp dưới đây.

[CÂU HỎI]
{question}
{options}

[THAM CHIẾU DO NGƯỜI RA ĐỀ CUNG CẤP]
{reference}

[CÂU TRẢ LỜI CỦA HỆ THỐNG]
{answer}

[CÁC ĐOẠN SÁCH MÀ HỆ THỐNG TRÍCH DẪN]
{spans}

[TIÊU CHÍ CHẤM]
1. correctness (tính đúng đắn): kết luận của câu trả lời có đúng so với tham chiếu không.
   2 = đúng hoàn toàn (kết luận trùng đáp án tham chiếu, phần giải thích không có sai sót thực chất);
   1 = đúng một phần (chọn đúng đáp án nhưng giải thích có sai sót đáng kể hoặc mâu thuẫn tham chiếu; hoặc kết luận chỉ đúng một phần);
   0 = sai (chọn sai, mâu thuẫn với tham chiếu, hoặc không đưa ra kết luận khi lẽ ra phải có).
2. completeness (tính đầy đủ): câu trả lời có nêu đủ các ý/bước then chốt của giải thích tham chiếu (với câu hỏi nhiều bước: đủ mọi bước suy luận) không.
   Chỉ so với các ý có trong câu trả lời/giải thích tham chiếu; không trừ điểm vì câu trả lời ngắn hoặc vì thiếu ý ngoài tham chiếu nếu các ý tham chiếu đã được nêu. \
Nếu câu hỏi chỉ cần nêu một sự kiện/khái niệm và câu trả lời tham chiếu chỉ nhắc lại nội dung phương án đúng, thì việc câu trả lời nêu đúng nội dung phương án đó là ĐỦ (2 điểm), \
không cần thêm giải thích. Chỉ với câu hỏi tính toán hoặc cần nhiều bước suy luận mới yêu cầu nêu công thức/các bước.
   2 = nêu đủ mọi ý then chốt của tham chiếu; 1 = thiếu một số ý then chốt (ví dụ chỉ có chữ cái đáp án hoặc bỏ sót một bước suy luận); 0 = hầu như không nêu ý then chốt nào.
3. grounding (nhất quán với bằng chứng): mọi phát biểu có nội dung trong câu trả lời (kể cả các nhận định kèm trích dẫn) có được CHÍNH CÁC ĐOẠN SÁCH TRÍCH DẪN ở trên nêu trực tiếp hoặc suy ra hiển nhiên không. \
Không dùng kiến thức bên ngoài để bù cho đoạn trích.
   2 = mọi phát biểu đều được đoạn trích hỗ trợ; 1 = một phần được hỗ trợ, phần còn lại không có trong đoạn trích; \
0 = đoạn trích không hỗ trợ hoặc mâu thuẫn với câu trả lời, hoặc không có đoạn trích nào cho một câu trả lời có phát biểu thực chất.
{special}
Hãy nhận xét ngắn gọn (tối đa 3 câu) rồi cho điểm. Trả về MỘT đối tượng JSON, không thêm gì khác:
{{"reason": "<nhận xét ngắn>", "correctness": 0|1|2, "completeness": 0|1|2, "grounding": 0|1|2}}"""

SPECIAL_UNANSWERABLE = """
Lưu ý (câu hỏi không thể trả lời từ sách): hành vi đúng là từ chối/nêu rõ không đủ căn cứ (chọn E hoặc tương đương), \
KHÔNG khẳng định một lựa chọn A–D như thể có căn cứ. correctness: 2 nếu hệ thống từ chối/nêu không đủ căn cứ, 0 nếu khẳng định một lựa chọn. \
completeness: 2 nếu phần giải thích nêu đúng thông tin còn thiếu hoặc lý do không thể kết luận, 1 nếu chỉ nêu chung chung, 0 nếu không giải thích. \
grounding: chỉ xét các phát biểu thực chất trong phần giải thích; nếu không có phát biểu thực chất nào cần đoạn trích thì cho 2.
"""
SPECIAL_ANSWERABLE_ABSTAIN = """
Lưu ý: câu hỏi này CÓ thể trả lời từ sách; nếu hệ thống từ chối trả lời (chọn E) thì correctness = 0 và completeness = 0.
"""


# ---------------------------------------------------------------- helpers
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_abstain(fa: Optional[FinalAnswer]) -> bool:
    if fa is None:
        return False
    return fa.status == "abstain" or (fa.answer_choice or "").strip().upper()[:1] == "E"


def chosen_letter(fa: Optional[FinalAnswer]) -> Optional[str]:
    """Same convention as otter.evaluation.metrics.item_record (text fallback when answer_choice is empty)."""
    if fa is None:
        return None
    ch = (fa.answer_choice or "").strip().upper()[:1]
    if ch not in ("A", "B", "C", "D", "E") and fa.status != "abstain":
        from otter.evaluation.metrics import extract_choice
        ch = extract_choice(fa.answer) or ""
    return ch if ch in ("A", "B", "C", "D", "E") else None


def cited_span_ids(fa: FinalAnswer) -> list[str]:
    ids = [c.span_id for cl in fa.claims for c in cl.citations]
    ids += [c.span_id for c in fa.where_to_read]
    return list(dict.fromkeys(i for i in ids if i))


def answer_hash(fa: FinalAnswer) -> str:
    return sha1(json.dumps({"a": fa.answer, "c": fa.answer_choice, "s": fa.status,
                            "cl": [(cl.text, [c.span_id for c in cl.citations]) for cl in fa.claims],
                            "w": [c.span_id for c in fa.where_to_read]}, ensure_ascii=False, sort_keys=True))[:16]


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _span_text(corpus, sid: str) -> Optional[str]:
    sp = corpus.get(sid) if corpus is not None else None
    return sp.text if sp is not None and sp.text else None


def _options_block(item: dict, letters: list[str]) -> str:
    opts = dict(item.get("options") or {})
    if "E" in letters and "E" not in opts:
        opts["E"] = "Không đủ căn cứ trong sách"
    return "\n".join(f"{k}. {opts[k]}" for k in sorted(opts))


def reference_block(item: dict, corpus=None) -> str:
    opts = item.get("options") or {}
    if item.get("answerable", True) is False:
        t = ABSTAIN_TYPE_VI.get(item.get("abstain_type") or "", "không đủ căn cứ trong sách")
        return ("Câu hỏi này KHÔNG thể trả lời chắc chắn từ sách giáo khoa được cung cấp (" + t + "). "
                "Hành vi đúng của hệ thống: từ chối trả lời hoặc nêu rõ không đủ căn cứ trong sách (lựa chọn E).")
    key = item.get("answer")
    lines = [f"Đáp án đúng: {key}. {opts.get(key, '')}".rstrip()]
    if item.get("reference_answer"):
        lines.append(f"Câu trả lời tham chiếu: {_clip(item['reference_answer'], 600)}")
    if item.get("explanation"):
        lines.append(f"Giải thích tham chiếu: {_clip(item['explanation'], 800)}")
    quotes = list((item.get("quotes") or {}).values())[:MAX_GOLD_QUOTES]
    if not quotes and corpus is not None:
        quotes = [t for t in (_span_text(corpus, s) for s in (item.get("gold_span_ids") or [])[:MAX_GOLD_QUOTES]) if t]
    if quotes:
        lines.append("Trích đoạn sách gốc (bằng chứng chuẩn):")
        lines += [f"- “{_clip(q, 500)}”" for q in quotes]
    return "\n".join(lines)


def answer_block(fa: FinalAnswer, ids: list[str]) -> str:
    ch = chosen_letter(fa)
    if is_abstain(fa):
        head = "Hệ thống TỪ CHỐI trả lời (không đủ căn cứ)."
    elif ch:
        head = f"Hệ thống chọn: {ch}."
    else:
        head = "Hệ thống không chọn rõ phương án nào."
    sid = {s: f"S{i + 1}" for i, s in enumerate(ids[:MAX_SPANS])}
    lines = [head, f"Nội dung trả lời: {_clip(fa.answer, MAX_ANSWER_CHARS)}"]
    if fa.claims:
        lines.append("Các nhận định của hệ thống kèm trích dẫn:")
        for cl in fa.claims:
            refs = "".join(f"[{sid[c.span_id]}]" for c in cl.citations if c.span_id in sid) or "[không trích dẫn]"
            lines.append(f"- {_clip(cl.text, 700)} {refs}")
    return "\n".join(lines)


def spans_block(ids: list[str], corpus) -> tuple[str, int]:
    """Returns (text, number of resolvable spans). Unknown ids are listed as not found (an invented citation is not evidence)."""
    if not ids:
        return "(Hệ thống không trích dẫn đoạn sách nào.)", 0
    out, n_ok = [], 0
    for i, s in enumerate(ids[:MAX_SPANS]):
        t = _span_text(corpus, s)
        if t is None:
            out.append(f"[S{i + 1}] (không tìm thấy đoạn sách này trong kho dữ liệu)")
        else:
            n_ok += 1
            out.append(f"[S{i + 1}] {_clip(t, MAX_SPAN_CHARS)}")
    if len(ids) > MAX_SPANS:
        out.append(f"(còn {len(ids) - MAX_SPANS} đoạn trích dẫn khác không hiển thị)")
    return "\n".join(out), n_ok


def build_prompt(item: dict, fa: FinalAnswer, corpus) -> tuple[list[dict], dict]:
    """-> (messages, info). The system name never enters the prompt."""
    ids = cited_span_ids(fa)
    sp_txt, n_ok = spans_block(ids, corpus)
    answerable = item.get("answerable", True) is not False
    letters = sorted((item.get("options") or {}).keys()) + (["E"] if (is_abstain(fa) or not answerable) else [])
    special = SPECIAL_UNANSWERABLE if not answerable else (SPECIAL_ANSWERABLE_ABSTAIN if is_abstain(fa) else "")
    q = item["question"]
    prompt = PROMPT_TEMPLATE.format(question=_clip(q, 1500), options=_options_block(item, letters),
                                    reference=reference_block(item, corpus), answer=answer_block(fa, ids), spans=sp_txt,
                                    special=special)
    info = {"n_cited": len(ids), "n_cited_resolved": n_ok, "prompt_sha1": sha1(prompt)}
    return [{"role": "user", "content": prompt}], info


def parse_judgment(text: str) -> dict[str, Any]:
    obj = parse_json_loose(text)
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("judge output is not a JSON object")
    out: dict[str, Any] = {"reason": str(obj.get("reason", ""))[:600]}
    for d in DIMS:
        v = obj.get(d)
        if isinstance(v, str):
            v = v.strip()
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"missing/invalid {d}: {v!r}")
        if f not in (0.0, 1.0, 2.0):
            raise ValueError(f"{d} out of range: {v!r}")
        out[d] = int(f)
    return out


def apply_rules(scores: dict, item: dict, fa: Optional[FinalAnswer], n_resolved: int) -> tuple[dict, list[str]]:
    s = dict(scores)
    rules: list[str] = []
    ch = chosen_letter(fa)
    if item.get("answerable", True) is False:
        want, got = 2 if is_abstain(fa) else 0, s["correctness"]
        if got != want:
            rules.append(f"unanswerable:{'abstain' if want == 2 else 'committed'}->correctness={want}")
        s["correctness"] = want
    elif ch != item.get("answer"):
        if s["correctness"] != 0:
            rules.append("answerable:wrong_or_no_choice->correctness=0")
        s["correctness"] = 0
    if n_resolved == 0 and not is_abstain(fa):
        if s["grounding"] != 0:
            rules.append("no_cited_text->grounding=0")
        s["grounding"] = 0
    return s, rules


def finalize(s: dict) -> dict:
    return {"judge_score": (s["correctness"] + s["completeness"] + s["grounding"]) / 6,
            "ans_score": (s["correctness"] + s["completeness"]) / 4,
            "strict": int(all(s[d] == 2 for d in DIMS))}


def rule_row(base: dict, item: dict, fa: Optional[FinalAnswer], why: str) -> dict:
    z = {d: 0 for d in DIMS}
    return {**base, **z, **finalize(z), "reason": why, "parse_ok": True, "rules": [why], "called": False}


def judge_item(llm: LLM, item: dict, fa: Optional[FinalAnswer], corpus, *, judge_id: str, max_tokens: int = 400) -> dict:
    qid = item["id"]
    base = {"question_id": qid, "judge_version": JUDGE_VERSION, "judge": judge_id, "judge_model": llm.model,
            "answerable": item.get("answerable", True) is not False, "answer_hash": answer_hash(fa) if fa else None}
    if fa is None or fa.status == "failed" or fa.error is not None:
        return rule_row(base, item, fa, "failed_or_missing_answer")
    msgs, info = build_prompt(item, fa, corpus)
    base.update(info)
    base["abstained"] = is_abstain(fa)
    base["choice"] = chosen_letter(fa)
    last_err = ""
    for attempt in range(2):                                   # retry once on unparsable output
        m = msgs if attempt == 0 else msgs + [{"role": "user", "content": "Đầu ra trước không hợp lệ. Chỉ trả về MỘT đối tượng JSON "
                                               'đúng dạng {"reason": "...", "correctness": 0|1|2, "completeness": 0|1|2, "grounding": 0|1|2}.'}]
        try:
            r = llm.chat(m, temperature=0.0, max_tokens=max_tokens, json_schema=SCHEMA)
            raw = parse_judgment(r.text)
            fin, rules = apply_rules(raw, item, fa, info["n_cited_resolved"])
            return {**base, **fin, **finalize(fin), "reason": raw["reason"], "parse_ok": True, "attempts": attempt + 1,
                    "rules": rules, "called": True, **{f"{d}_raw": raw[d] for d in DIMS}}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return {**base, "parse_ok": False, "error": last_err[:300], "attempts": 2, "called": True}


# ---------------------------------------------------------------- run / io
def load_judgments(path: Path | str) -> dict[str, dict]:
    """Latest valid row per question id (rows of other judge versions ignored)."""
    out: dict[str, dict] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("judge_version") == JUDGE_VERSION and r.get("parse_ok"):
                    out[r["question_id"]] = r
    return out


def judge_run(items: list[dict], answers: dict[str, FinalAnswer], corpus, llm: LLM, out_path: Path, *, judge_id: str,
              workers: int = 8, limit: Optional[int] = None) -> dict[str, int]:
    """Resumable. Skips (question id, prompt hash) pairs already judged; a changed answer/item/template is re-judged."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("judge_version") == JUDGE_VERSION and r.get("parse_ok"):
                    cache[(r["question_id"], r.get("prompt_sha1") or ("rule:" + str(r.get("answer_hash"))))] = r
    lock = threading.Lock()
    stats = {"todo": 0, "cached": 0, "judged": 0, "parse_fail": 0}

    def key_of(it, fa):
        if fa is None or fa.status == "failed" or fa.error is not None:
            return (it["id"], "rule:" + str(answer_hash(fa) if fa else None))
        return (it["id"], build_prompt(it, fa, corpus)[1]["prompt_sha1"])

    todo = []
    for it in items:
        fa = answers.get(it["id"])
        if key_of(it, fa) in cache:
            stats["cached"] += 1
        else:
            todo.append((it, fa))
    if limit:
        todo = todo[:limit]
    stats["todo"] = len(todo)

    def one(args):
        it, fa = args
        row = judge_item(llm, it, fa, corpus, judge_id=judge_id)
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["judged" if row.get("parse_ok") else "parse_fail"] += 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, todo))
    out_path.touch()
    return stats


# ---------------------------------------------------------------- aggregation
def judge_flags(system: str, backbone_model: Optional[str], judge_id: str) -> dict[str, bool]:
    """self_judge: judge model == system backbone; verifier_judge: otter_v2 uses Qwen3-8B (role `small`) as its verifier."""
    from otter.llm.interface import load_config
    jm = load_config()["llm"][JUDGES[judge_id]["role"]]["model"]
    return {"self_judge": bool(backbone_model) and jm == backbone_model,
            "verifier_judge": system.startswith("otter_v2") and JUDGES[judge_id]["role"] == "small"}


def summarize_judgments(rows: dict[str, dict], item_ids: list[str], *, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Mean judge score / answer-only score / strict rate with bootstrap CI over the bench items that carry a valid judgment."""
    sel = [rows[i] for i in item_ids if i in rows]
    out: dict[str, Any] = {"n_items": len(item_ids), "n_judged": len(sel)}
    if not sel:
        return out
    for key, name in (("judge_score", "judge"), ("ans_score", "ans"), ("strict", "strict")):
        m, lo, hi = bootstrap_ci([float(r[key]) for r in sel], n_boot=n_boot, seed=seed)
        out[name] = m; out[f"{name}_ci_lo"] = lo; out[f"{name}_ci_hi"] = hi
    for d in DIMS:
        out[f"{d}_mean"] = float(np.mean([r[d] for r in sel])) / 2
    out["rule_overrides"] = sum(1 for r in sel if r.get("rules") and r.get("called"))
    return out


def quadratic_weighted_kappa(a: list[int], b: list[int], k: int = 3) -> Optional[float]:
    if not a:
        return None
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    ra, rb = O.sum(1), O.sum(0)
    E = np.outer(ra, rb) / O.sum()
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    den = (W * E).sum()
    if den == 0:
        return 1.0 if (W * O).sum() == 0 else 0.0
    return float(1 - (W * O).sum() / den)


def agreement_stats(ref: list[dict], other: list[dict]) -> dict[str, Any]:
    """Paired rows (same order) -> per-dimension exact agreement, QWK, mean difference (other - ref), within-1."""
    out: dict[str, Any] = {"n": len(ref)}
    for d in DIMS + ("total",):
        if d == "total":
            a = [sum(r[x] for x in DIMS) for r in ref]; b = [sum(r[x] for x in DIMS) for r in other]
            out[d] = {"exact": float(np.mean([x == y for x, y in zip(a, b)])), "mean_diff": float(np.mean(np.array(b) - np.array(a))),
                      "pearson": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None}
            continue
        a = [r[d] for r in ref]; b = [r[d] for r in other]
        diff = np.array(b) - np.array(a)
        out[d] = {"exact": float(np.mean(diff == 0)), "within1": float(np.mean(np.abs(diff) <= 1)),
                  "qwk": quadratic_weighted_kappa(a, b), "mean_diff": float(diff.mean()),
                  "judge_higher": float(np.mean(diff > 0)), "judge_lower": float(np.mean(diff < 0))}
    return out


def run_judge_summaries(run_dir: Path | str, item_ids: list[str], *, n_boot: int = 2000, judges: tuple[str, ...] = tuple(JUDGES)
                        ) -> dict[str, dict[str, Any]]:
    """{judge_id: summary + flags} for the judgments stored in a run directory (empty dict when none)."""
    d = Path(run_dir)
    meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
    out: dict[str, dict[str, Any]] = {}
    for j in judges:
        rows = load_judgments(d / f"answer_judgments.{j}.jsonl")
        if not rows:
            continue
        s = summarize_judgments(rows, item_ids, n_boot=n_boot)
        bbm = meta.get("backbone_model")
        if bbm is None:
            try:
                from otter.llm.interface import load_config
                bbm = load_config()["llm"].get("small" if d.name.startswith("small") else "main", {}).get("model")
            except Exception:
                bbm = None
        s.update(judge_flags(str(meta.get("system") or d.parent.name), bbm, j))
        out[j] = s
    return out


def fmt_judge(s: Optional[dict], key: str = "judge", dec: int = 3, ci: bool = True) -> str:
    """'0.873 ±0.031' (+ '*' self-judge, 'v' otter_v2 verifier); '--' if missing. key: judge | strict | ans"""
    if not s or key not in s:
        return "--"
    txt = f"{s[key]:.{dec}f}"
    if ci:
        txt += f" ±{(s[key + '_ci_hi'] - s[key + '_ci_lo']) / 2:.{dec}f}"
    if s.get("n_judged", 0) < s.get("n_items", 0):
        txt += f" (n={s['n_judged']}/{s['n_items']})"
    return txt + ("*" if s.get("self_judge") else "") + ("v" if s.get("verifier_judge") else "")
