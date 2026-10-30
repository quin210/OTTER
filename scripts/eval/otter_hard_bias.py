"""Answer-only / heuristic baselines on OTTER-Hard (artifact check). Writes benchmarks/otter_hard/bias_report.json.

  - heuristics: longest option, option most similar to the other options, option most similar to the question
  - LLM answer-only: the model sees ONLY the four options (no question, no passage); 2 option permutations
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation.hard_gen import chat_json, permute, fmt_options, _letter, CLOSED_SCHEMA  # noqa: E402
from otter.evaluation.mcq import LETTERS  # noqa: E402
from otter.evaluation.text import char_ngrams, jaccard  # noqa: E402
from otter.llm.interface import get_llm  # noqa: E402

SYS = "Bạn là trợ lý trả lời câu hỏi trắc nghiệm. Chỉ trả về JSON hợp lệ."
P = ("Dưới đây là bốn phương án của một câu hỏi trắc nghiệm, nhưng câu hỏi đã bị ẩn. Hãy đoán phương án có nhiều khả năng là "
     "đáp án đúng nhất.\n{opts}\n\nTrả về JSON: {{\"answer\": \"<A|B|C|D>\"}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "benchmarks/otter_hard/otter_hard_all.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks/otter_hard/bias_report.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--roles", default="small,main")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.bench) if l.strip()]
    n = len(rows)
    rep = {"n": n, "chance": 0.25}
    L = lambda r: {k: len(r["options"][k]) for k in LETTERS}
    rep["longest_option_acc"] = sum(max(L(r), key=L(r).get) == r["answer"] for r in rows) / n
    rep["correct_unique_longest_rate"] = sum(L(r)[r["answer"]] == max(L(r).values()) and list(L(r).values()).count(max(L(r).values())) == 1 for r in rows) / n
    rep["correct_unique_shortest_rate"] = sum(L(r)[r["answer"]] == min(L(r).values()) and list(L(r).values()).count(min(L(r).values())) == 1 for r in rows) / n
    def centroid(r):
        g = {k: char_ngrams(r["options"][k]) for k in LETTERS}
        return max(LETTERS, key=lambda k: sum(jaccard(g[k], g[j]) for j in LETTERS if j != k))
    rep["most_similar_to_other_options_acc"] = sum(centroid(r) == r["answer"] for r in rows) / n
    def qsim(r):
        q = char_ngrams(r["question"])
        return max(LETTERS, key=lambda k: jaccard(q, char_ngrams(r["options"][k])))
    rep["most_similar_to_question_acc"] = sum(qsim(r) == r["answer"] for r in rows) / n
    rep["by_length_ratio"] = {}
    for role in [x for x in a.roles.split(",") if x]:
        llm = get_llm(role)
        def one(job):
            r, p = job
            o, ans, back = permute(r["options"], r["answer"], r["id"] + f":aonly:{p}")
            try:
                obj, _ = chat_json(llm, [{"role": "system", "content": SYS}, {"role": "user", "content": P.format(opts=fmt_options(o))}],
                                   CLOSED_SCHEMA, max_tokens=40, validate=lambda x: _letter(x.get("answer")) is not None)
                return r["id"], p, back.get(_letter(obj["answer"])) == r["answer"]
            except Exception:
                return r["id"], p, None
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, [(r, p) for r in rows for p in range(4)]))
        ok = [x for x in res if x[2] is not None]
        rep[f"answer_only_{role}_acc"] = sum(x[2] for x in ok) / max(1, len(ok))
        rep[f"answer_only_{role}_n_calls"] = len(ok)
        dif = {r["id"]: r["difficulty"] for r in rows}
        ne = [x for x in ok if dif[x[0]] != "easy"]
        rep[f"answer_only_{role}_acc_non_easy"] = sum(x[2] for x in ne) / max(1, len(ne))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
