"""OTTER v2 (`otter_v2`): span-of-record, option-wise proof obligations, cascade router, cross-model
citation verification and calibrated abstention. Copy-then-extend of the OTTER idea; does not touch
`otter_full` (master.py) code paths.

Pipeline (see docs/OTTER_V2_DESIGN.md)
  1. retrieve   hybrid BM25+bge-m3 spans for the question and for every (question, option) pair, fused by RRF;
                ontology facts/rules (supported only) add *candidate spans* and short hints (navigation layer,
                never evidence, never citable).                              [0 LLM calls]
  2. obligate   ONE generator call: for each option decide entailed / refuted / unsupported by the shown spans,
                pick the answer and emit claims {text, span_ids, quote}.     [1 call, backbone]
  3. cascade    if the grounding level is < 2 (router) a gap round retrieves with the generator's own `missing`
                note + unsupported options and re-runs step 2 once.          [0-1 extra call]
  4. verify     deterministic quote-in-span check, then ONE call to a different model (role `small`) that checks
                each claim against ONLY its cited spans; unsupported claims are dropped. [1 call, verifier]
  5. calibrate  grounding level 0..3 -> status supported/qualified; optional abstention below a threshold.
Calculation items are delegated to the otter_full master (deterministic calculator / rule engine).
"""
from __future__ import annotations

import ast
import logging
import re
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from otter.agents import prompts as P
from otter.agents.runtime import (LRU, Meter, MeteredLLM, QuestionInput, detect_question_type, extract_letter,
                                  safe_json)
from otter.llm.interface import LLM
from otter.reasoning.citation_resolver import CitationResolver
from otter.reasoning.text import truncate
from otter.schemas.reasoning import AnswerClaim, ChoiceVerdict, FinalAnswer

log = logging.getLogger("otter.v2")

ABSTAIN_TEXT = "Không đủ căn cứ trong sách giáo khoa để trả lời câu hỏi này."
V2_ABLATIONS = ("no_option_obligations", "no_router", "no_gap_round", "no_verified_citation", "no_abstention",
                "no_ontology_hints", "no_calc_delegate", "no_option_queries")


@dataclass
class V2Config:
    top_k_question: int = 6
    top_k_option: int = 3
    max_spans: int = 10
    span_chars: int = 700
    hint_facts: int = 4
    ontology_spans: int = 2          # extra candidate spans coming from ontology hits
    option_obligations: bool = True   # prompt-level per-option verdicts (verification, not retrieval)
    option_queries: bool = True       # retrieval-level per-option queries (retrieval, not verification)
    router: bool = True              # cascade: gap round only when grounding level < gap_level
    gap_round: bool = True
    gap_level: int = 2
    verify_trigger: bool = True      # v2.1: router also fires on conflicting `entailed` options / verifier rejection
    verified_citation: bool = True
    abstention: bool = False         # calibrated abstention on/off (operating point)
    abstain_below: int = 1           # abstain when grounding level < abstain_below (0..3)
    ontology_hints: bool = True
    calc_delegate: bool = True
    verifier_role: str = "small"
    max_claims: int = 3
    parallel_retrieval: int = 4


# --------------------------------------------------------------------------- prompts
SYSTEM = ("Bạn là trợ lý kiểm chứng bằng chứng cho câu hỏi trắc nghiệm theo sách giáo khoa phổ thông Việt Nam. "
          "Văn bản sách giáo khoa (các đoạn có mã) là bằng chứng duy nhất được phép trích dẫn. "
          "Trả lời bằng đúng MỘT đối tượng JSON, không thêm chữ nào khác.")


def gen_user(q: QuestionInput, spans: list[dict], hints: list[str], obligations: bool, max_claims: int) -> str:
    ev = "\n\n".join(f"[{s['span_id']}] {s['text']}" for s in spans) or "(không tìm thấy đoạn văn nào)"
    h = ("\nGợi ý từ ontology (tự động, có thể SAI, chỉ để định hướng, KHÔNG phải bằng chứng, không trích dẫn):\n"
         + "\n".join(f"- {x}" for x in hints) + "\n") if hints else ""
    head = f"{P._scope(q.subject, q.grade)}\nCác đoạn trích sách giáo khoa:\n{ev}\n{h}\nCâu hỏi: {q.question}\n"
    if not q.choices:
        return head + 'Trả về JSON: {"answer": "...", "claims": [{"text": "...", "span_ids": ["..."], "quote": "..."}]}'
    keys = list(q.choices)
    opt_schema = ('"options": {' + ", ".join(f'"{k}": "entailed|refuted|unsupported"'
                                              for k in keys) + "}, ") if obligations else ""
    ab = [k for k, v in q.choices.items() if _is_abstain(k, v)]
    ab_note = (f"Lựa chọn {ab[0]} nghĩa là sách không đủ căn cứ: CHỌN {ab[0]} khi không lựa chọn nào khác được các "
               "đoạn trích chứng minh (entailed); khi đó để claims rỗng.\n") if ab else ""
    task = ab_note + (
        "Nhiệm vụ: với MỖI lựa chọn, kiểm tra xem các đoạn trích có đủ để chứng minh lựa chọn đó là đáp án ĐÚNG của "
        "câu hỏi hay không (nghĩa vụ chứng minh).\n"
        '- "entailed": các đoạn trích (có thể ghép 2 đoạn) cho thấy lựa chọn đúng và đầy đủ.\n'
        '- "refuted": các đoạn trích cho thấy lựa chọn sai (mâu thuẫn).\n'
        '- "unsupported": đoạn trích không đủ để kết luận.\n'
        if obligations else "")
    return (head + f"Các lựa chọn:\n{P.fmt_choices(q.choices)}\n\n" + task
            + "Chọn đáp án tốt nhất. Nếu có lựa chọn entailed thì chọn nó. Nếu KHÔNG lựa chọn nào entailed, vẫn chọn "
              'đáp án khả dĩ nhất nhưng để "claims" rỗng và nêu trong "missing" thông tin còn thiếu trong sách.\n'
              f'"claims": tối đa {max_claims} khẳng định ngắn, mỗi khẳng định là một mệnh đề đầy đủ (kèm chủ thể) rút '
              'từ các đoạn trích, có "span_ids" (chỉ mã đã hiển thị) và "quote" là một câu NGUYÊN VĂN ngắn từ đoạn đó.\n'
              'Trả về JSON gọn (một dòng, không xuống dòng thừa): {' + opt_schema + '"answer": "<chữ cái>", "claims": [{"text": "...", "span_ids": ["..."], '
              '"quote": "..."}], "missing": "<thiếu gì, hoặc rỗng>"}')


VERIFY_SYSTEM = ("Bạn là người kiểm định bằng chứng độc lập. Chỉ dựa vào các đoạn trích được đưa, không dùng kiến thức "
                 "bên ngoài. Trả lời bằng đúng MỘT đối tượng JSON.")


def verify_user(question: str, answer_text: Optional[str], claims: list[dict]) -> str:
    blocks = []
    for c in claims:
        sp = "\n".join(f"[{s['span_id']}] {s['text']}" for s in c["spans"])
        blocks.append(f"MỆNH ĐỀ {c['id']}: {c['text']}\nĐOẠN TRÍCH ĐƯỢC TRÍCH DẪN:\n{sp}")
    ans = f"\nĐáp án được đề xuất: {answer_text}\n" if answer_text else ""
    return (f"Câu hỏi: {question}{ans}\n" + "\n\n".join(blocks)
            + '\n\nVới mỗi mệnh đề: "supported" = đoạn trích nêu trực tiếp hoặc suy ra hiển nhiên mọi ý quan trọng; '
              '"partial" = chỉ một phần; "unsupported" = không hỗ trợ/mâu thuẫn. Thêm "answer_supported": '
              '"yes|no" = các mệnh đề được hỗ trợ có đủ để kết luận đáp án đề xuất là đúng cho câu hỏi không.\n'
              'Trả về JSON: {"claims": [{"id": "<id>", "verdict": "supported|partial|unsupported"}], '
              '"answer_supported": "yes|no"}')


# --------------------------------------------------------------------------- helpers
def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower(), flags=re.U).strip()


def quote_in_span(quote: str, text: str, min_ratio: float = 0.8) -> bool:
    """Deterministic: normalised quote occurs in span (or >=min_ratio of its tokens occur as a contiguous run)."""
    nq, nt = _norm(quote or ""), _norm(text or "")
    if not nq:
        return False
    if nq in nt:
        return True
    toks = nq.split()
    if len(toks) < 4:
        return False
    # longest contiguous run of quote tokens found in text
    best = 0
    for i in range(len(toks)):
        for j in range(len(toks), i + best, -1):
            if " ".join(toks[i:j]) in nt:
                best = max(best, j - i)
                break
    return best / len(toks) >= min_ratio


def _rrf(lists: list[list[str]], k: int = 60) -> list[str]:
    sc: dict[str, float] = {}
    for lst in lists:
        for r, sid in enumerate(lst):
            sc[sid] = sc.get(sid, 0.0) + 1.0 / (k + r + 1)
    return sorted(sc, key=lambda s: -sc[s])


def snippet(text: str, quote: str = "", max_sent: int = 2, max_chars: int = 320) -> str:
    """<=2 sentences of the span: those around the quote when it is found, else the opening ones."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?;:])\s+|\n+", text or "") if x.strip()]
    if not sents:
        return (text or "")[:max_chars]
    start = 0
    nq = _norm(quote or "")
    if nq:
        for i, x in enumerate(sents):
            if _norm(x) and (_norm(x) in nq or nq[:40] in _norm(x)):
                start = i
                break
    return " ".join(sents[start:start + max_sent])[:max_chars]


def _is_abstain(k: str, v: str) -> bool:
    """OTTER-Abstain explicit option E: 'Không đủ căn cứ trong sách'."""
    return str(k).upper() == "E" and str(v).strip().lower().startswith("không đủ căn cứ")


def _ids(v: Any) -> list[str]:
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            v = [v]
    return [str(x) for x in (v or [])]


# --------------------------------------------------------------------------- system
class OtterV2:
    name = "otter_v2"

    def __init__(self, llm: LLM, verifier: LLM, store: Any, cfg: V2Config, *, name: str = "otter_v2",
                 llm_cache: Optional[LRU] = None, calc_system: Any = None):
        self.llm, self.verifier, self.store, self.cfg = llm, verifier, store, cfg
        self.name, self.cache = name, llm_cache
        self.resolver = CitationResolver(store)
        self.calc_system = calc_system
        self.corpus = None

    # ---- retrieval (no LLM) ------------------------------------------------------------------------
    def _search(self, query: str, q: QuestionInput, k: int) -> list[str]:
        try:
            return [s.span_id for s, _ in self.store.search_spans(query, subject=q.subject, grade=q.grade, top_k=k)]
        except Exception as e:
            log.debug("search failed: %s", e)
            return []

    def _hints(self, q: QuestionInput, query: str) -> tuple[list[str], list[str]]:
        try:
            hits = self.store.search_knowledge(query, subject=q.subject, grade=q.grade,
                                               kinds=("fact", "rule"), top_k=self.cfg.hint_facts + 4)
        except Exception:
            return [], []
        hits = [h for h in hits if h.get("source_support") == "supported"][: self.cfg.hint_facts]
        texts = [truncate(str(h.get("label_or_text")), 220) for h in hits]
        sids = [s for h in hits for s in _ids(h.get("source_span_ids"))[:1]]
        return texts, sids

    def retrieve(self, q: QuestionInput, extra_queries: Optional[list[str]] = None, exclude: Optional[set] = None,
                 only_extra: bool = False) -> tuple[list[dict], list[str], dict]:
        """only_extra (gap round, v2.1): search ONLY with the gap queries. Fusing them with the original queries let the first-round
        queries dominate the RRF top-`max_spans`, so the gap round returned the same spans (it added a new span on 2 of 12 escalated
        dev items)."""
        c = self.cfg
        queries: list[tuple[str, int]] = [] if (only_extra and extra_queries) else [
            (q.question + " " + " ".join((q.choices or {}).values()), c.top_k_question), (q.question, c.top_k_question)]
        if q.choices and c.option_queries and not (only_extra and extra_queries):
            queries += [(f"{q.question} {v}", c.top_k_option) for k, v in q.choices.items() if not _is_abstain(k, v)]
        for e in extra_queries or []:
            queries.append((e, c.top_k_question if only_extra else c.top_k_option + 1))
        with ThreadPoolExecutor(max_workers=c.parallel_retrieval) as ex:
            lists = list(ex.map(lambda t: self._search(t[0], q, t[1]), queries))
            hint_texts, hint_sids = (self._hints(q, q.question + " " + " ".join((q.choices or {}).values()))
                                     if (c.ontology_hints and not only_extra) else ([], []))
        ranked = _rrf(lists)
        if hint_sids:        # ontology = navigation: its source spans become candidates, ranked below text hits
            ranked += [s for s in dict.fromkeys(hint_sids) if s not in ranked][: c.ontology_spans]
        if exclude:
            ranked = [s for s in ranked if s not in exclude]
        ranked = ranked[: c.max_spans]
        spans = []
        try:
            got = {s.span_id: s for s in self.store.get_spans(ranked)}
        except Exception:
            got = {}
        for sid in ranked:
            if sid in got:
                spans.append({"span_id": sid, "text": truncate(got[sid].text, c.span_chars)})
        return spans, hint_texts, {"n_queries": len(queries) + (1 if (c.ontology_hints and not only_extra) else 0),
                                   "hint_span_ids": hint_sids, "span_ids": [x["span_id"] for x in spans]}

    # ---- generator ---------------------------------------------------------------------------------
    def _generate(self, llm, q: QuestionInput, spans, hints) -> dict:
        obj, raw = safe_json(llm, [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": gen_user(q, spans, hints, self.cfg.option_obligations,
                                                                        self.cfg.max_claims)}],
                             max_tokens=1100, retry_msg=P.JSON_RETRY)
        obj = obj if isinstance(obj, dict) else {}
        shown = {s["span_id"] for s in spans}
        letter = extract_letter(obj.get("answer", raw), q.choices) if q.choices else None
        opts = {}
        for k, v in (obj.get("options") or {}).items() if isinstance(obj.get("options"), dict) else []:
            vd = str(v.get("verdict", "") if isinstance(v, dict) else v).strip().lower()
            opts[str(k).upper()] = {"verdict": vd if vd in ("entailed", "refuted", "unsupported") else "unsupported"}
        claims = []
        for cl in (obj.get("claims") or [])[: self.cfg.max_claims] if isinstance(obj.get("claims"), list) else []:
            if not isinstance(cl, dict) or not str(cl.get("text") or "").strip():
                continue
            sids = [s for s in _ids(cl.get("span_ids")) if s in shown]
            if sids:
                claims.append({"text": str(cl["text"]).strip(), "span_ids": sids, "quote": str(cl.get("quote") or "")})
        return {"letter": letter, "options": opts, "claims": claims, "missing": str(obj.get("missing") or "").strip(),
                "raw": raw if not obj else ""}

    def _level(self, q: QuestionInput, g: dict) -> int:
        """Pre-verification grounding level: 0 none, 1 claims but chosen option not entailed, 2 entailed."""
        if not g["claims"] or g["letter"] is None and q.choices:
            return 0
        if self.cfg.option_obligations and q.choices:
            return 2 if g["options"].get(g["letter"], {}).get("verdict") == "entailed" else 1
        return 2

    # ---- verification ------------------------------------------------------------------------------
    def _verify(self, vllm, q: QuestionInput, g: dict, span_text: dict[str, str]) -> dict:
        claims = g["claims"]
        for c in claims:   # deterministic layer: quote must occur in a cited span
            c["quote_ok"] = any(quote_in_span(c["quote"], span_text.get(s, "")) for s in c["span_ids"])
        ans_text = f"{g['letter']}. {q.choices[g['letter']]}" if q.choices and g["letter"] else None
        if not self.cfg.verified_citation or not claims:
            for c in claims:
                c["verdict"] = "unverified"
            return {"answer_supported": None}
        payload = [{"id": f"c{i + 1}", "text": c["text"],
                    "spans": [{"span_id": s, "text": span_text[s]} for s in c["span_ids"] if s in span_text]}
                   for i, c in enumerate(claims)]
        obj, _ = safe_json(vllm, [{"role": "system", "content": VERIFY_SYSTEM},
                                  {"role": "user", "content": verify_user(q.question, ans_text, payload)}],
                           max_tokens=250, retry_msg=P.JSON_RETRY)
        obj = obj if isinstance(obj, dict) else {}
        vd = {str(x.get("id")): str(x.get("verdict", "")).lower() for x in (obj.get("claims") or [])
              if isinstance(x, dict)} if isinstance(obj.get("claims"), list) else {}
        for i, c in enumerate(claims):
            v = vd.get(f"c{i + 1}", "unsupported")
            c["verdict"] = v if v in ("supported", "partial", "unsupported") else "unsupported"
        a = str(obj.get("answer_supported", "")).lower()
        return {"answer_supported": True if a.startswith("y") else (False if a.startswith("n") else None)}

    # ---- main --------------------------------------------------------------------------------------
    def answer(self, item: dict[str, Any]) -> FinalAnswer:
        t0 = time.perf_counter()
        meter, vmeter = Meter(), Meter()
        q = QuestionInput.from_item(item)
        try:
            if (self.cfg.calc_delegate and self.calc_system is not None and q.choices
                    and detect_question_type(q.question, q.choices) == "calculation"):
                fa = self.calc_system.answer(item)
                fa.system = self.name
                fa.tool_usage["v2_route"] = "delegate_calc"
                return fa
            fa = self._answer(q, MeteredLLM(self.llm, meter, cache=self.cache),
                              MeteredLLM(self.verifier, vmeter, cache=self.cache))
        except Exception as e:
            fa = FinalAnswer(question_id=q.qid, system=self.name, answer="", status="failed",
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}")
        m, v = meter.as_dict(), vmeter.as_dict()
        fa.tool_usage.update(m)
        fa.tool_usage.update({"main_llm_calls": m["llm_calls"], "verifier_llm_calls": v["llm_calls"],
                              "verifier_tokens": v["tokens"], "main_tokens": m["tokens"]})
        fa.tool_usage["llm_calls"] = m["llm_calls"] + v["llm_calls"]
        fa.tool_usage["prompt_tokens"] = m["prompt_tokens"] + v["prompt_tokens"]
        fa.tool_usage["completion_tokens"] = m["completion_tokens"] + v["completion_tokens"]
        fa.tool_usage["tokens"] = m["tokens"] + v["tokens"]
        fa.tool_usage["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        fa.tool_usage.setdefault("subagents", 0)
        fa.tool_usage["corpus"] = self.corpus
        return fa

    def _grounded(self, q: QuestionInput, g: dict, level: int, ver: dict) -> tuple[list[dict], int]:
        """Claims kept after verification and the grounding level 0..3 for one generation round."""
        cl = g["claims"]
        keep = [x for x in cl if x["verdict"] in ("supported", "unverified")
                or (x["verdict"] == "partial" and x["quote_ok"])]
        n_sup = sum(x["verdict"] == "supported" for x in cl)
        c = self.cfg
        if not keep or g["letter"] is None and q.choices:
            return keep, 0
        grounding = level  # 1 or 2 from the generator
        if c.verified_citation:
            if ver["answer_supported"] is False and n_sup == 0:
                grounding = min(grounding, 1)
            elif level >= 2 and n_sup >= 1 and any(x["quote_ok"] for x in keep) and ver["answer_supported"] is not False:
                grounding = 3
        elif level >= 2 and any(x["quote_ok"] for x in keep):
            grounding = 3
        return keep, grounding

    def _escalate(self, q: QuestionInput, g: dict, level: int, ver: dict, keep: list[dict]) -> Optional[str]:
        """Router: why a gap round is needed (None = the obligation is closed, stay at one round).
        v2.1: the router used to look only at the generator's own level (`level < gap_level`), but the generator rated the
        chosen option `entailed` on 108/108 dev items, so the cascade never fired (gap_level=2) or always fired (gap_level=3).
        It now also reacts to evidence of an unclosed obligation that is only visible after the cheap checks:
        two or more options `entailed` at once (conflicting proofs), or the cross-model verifier rejecting the answer."""
        c = self.cfg
        if level < c.gap_level:
            return "gen_level"
        if not c.verify_trigger:
            return None
        if q.choices and sum(o["verdict"] == "entailed" for o in g["options"].values()) >= 2:
            return "conflicting_entailed"
        if c.verified_citation and (ver["answer_supported"] is False or (g["claims"] and not keep)):
            return "verifier_reject"
        return None

    def _answer(self, q: QuestionInput, llm, vllm) -> FinalAnswer:
        c = self.cfg
        spans, hints, info = self.retrieve(q)
        r_calls, by_round = info["n_queries"], [info["span_ids"]]
        g = self._generate(llm, q, spans, hints)
        level = self._level(q, g)
        rounds = 1
        span_text = {s["span_id"]: s["text"] for s in spans}
        ver = self._verify(vllm, q, g, span_text)
        keep, grounding = self._grounded(q, g, level, ver)
        why = self._escalate(q, g, level, ver, keep) if c.gap_round else None
        trigger = why or ("always" if (c.gap_round and not c.router) else None)
        # cascade router: escalate only when the first round did not close the obligation
        if trigger:
            extra = [g["missing"]] if g["missing"] else []
            if q.choices:
                extra += [f"{q.question} {q.choices[k]}" for k, o in g["options"].items()
                          if (o["verdict"] == "unsupported" or (trigger == "conflicting_entailed" and o["verdict"] == "entailed"))
                          and k in q.choices][:2]
            if not extra:
                extra = [q.question + " " + (q.choices[g["letter"]] if g["letter"] else "")]
            spans2, hints2, info2 = self.retrieve(q, extra_queries=extra, only_extra=True)
            r_calls += info2["n_queries"]
            by_round.append(info2["span_ids"])
            merged = {s["span_id"]: s for s in spans}
            for s in spans2:
                merged.setdefault(s["span_id"], s)
            new_ids = [s for s in merged if s not in {x["span_id"] for x in spans}]
            if new_ids or not c.router:
                spans_all = list(merged.values())[: c.max_spans + 4]
                g2 = self._generate(llm, q, spans_all, hints or hints2)
                l2 = self._level(q, g2)
                st2 = {s["span_id"]: s["text"] for s in spans_all}
                ver2 = self._verify(vllm, q, g2, st2)
                keep2, gr2 = self._grounded(q, g2, l2, ver2)
                if gr2 >= grounding:        # keep the better-grounded round; tie -> the round that saw more evidence
                    g, level, ver, keep, grounding, spans, span_text = g2, l2, ver2, keep2, gr2, spans_all, st2
                rounds = 2
        cl = g["claims"]
        letter = g["letter"]
        e_chosen = bool(letter and q.choices and _is_abstain(letter, q.choices.get(letter, "")))
        abstain = e_chosen or (c.abstention and grounding < c.abstain_below)
        answer_claims = []
        for i, x in enumerate(keep):
            cites, _ = self.resolver.resolve_many(x["span_ids"], with_quote=True, quote_chars=240)
            if cites:
                answer_claims.append(AnswerClaim(claim_id=f"c{i + 1}", text=x["text"], citations=cites))
        # where_to_read: decisive span(s) = verified claim spans that contain the quote (deterministic localisation),
        # else the claim's first span; fallback (flagged) = generator-cited / top-ranked retrieved span
        decisive: list[tuple[str, str]] = []
        for x in keep:
            hit = [s for s in x["span_ids"] if quote_in_span(x["quote"], span_text.get(s, ""))] or x["span_ids"][:1]
            for s in hit[:1]:
                if s not in [d[0] for d in decisive]:
                    decisive.append((s, x["quote"]))
        located = "verified" if decisive else "none"
        if not decisive and spans and not e_chosen:
            cand = [s for x in cl for s in x["span_ids"]] or [spans[0]["span_id"]]
            decisive, located = [(cand[0], "")], "unverified"
        wtr = []
        for sid, qt in decisive[:2]:
            cs, _ = self.resolver.resolve_many([sid], with_quote=False)
            if cs:
                cs[0].quote = snippet(span_text.get(sid, ""), qt)
                wtr.append(cs[0])
        if q.choices:
            text = (f"Đáp án {letter}. {q.choices[letter]}" if letter else "Không xác định được đáp án.")
        else:
            text = ""
        if abstain:
            fa = FinalAnswer(question_id=q.qid, system=self.name, answer=ABSTAIN_TEXT,
                             answer_choice=letter if e_chosen else None,
                             status="abstain", claims=[])
        else:
            status = "supported" if grounding >= 3 else "qualified"
            if not letter and q.choices:
                status = "abstain"
            fa = FinalAnswer(question_id=q.qid, system=self.name, answer=text, answer_choice=letter,
                             status=status, claims=answer_claims)
        fa.where_to_read = [] if e_chosen else wtr
        fa.tool_usage["v2_located"] = "none" if e_chosen else located
        fa.choice_verdicts = [ChoiceVerdict(choice_id=k, status={"entailed": "entailed", "refuted": "contradicted"}.get(
            o["verdict"], "undetermined")) for k, o in g["options"].items() if k in (q.choices or {})]
        fa.tool_usage.update({"calls": 0, "retrieved_span_ids": list(span_text), "v2_rounds": rounds, "retrieval_calls": r_calls,
                              "retrieved_by_round": by_round,
                              "v2_route": "cascade" if rounds == 2 else "single_round", "v2_escalation": trigger,
                              "v2_grounding": grounding, "v2_gen_level": level, "v2_answer_supported": ver["answer_supported"],
                              "v2_claims": [{"text": x["text"], "span_ids": x["span_ids"], "verdict": x["verdict"],
                                             "quote_ok": x["quote_ok"]} for x in cl],
                              "v2_n_claims_dropped": len(cl) - len(keep), "v2_ungrounded_answer": grounding < 2,
                              "v2_would_abstain_at": {str(t): grounding < t for t in (1, 2, 3)}})
        return fa


# --------------------------------------------------------------------------- frozen config (configs/otter_v2.yaml)
V2_YAML = Path(__file__).resolve().parents[2] / "configs" / "otter_v2.yaml"


def load_frozen_config(path: Optional[Path] = None) -> V2Config:
    """V2Config from configs/otter_v2.yaml, the single source of frozen hyper-parameters (code defaults only fill gaps).
    Command-line `--flag k=v` values still override (used for ablations / the threshold study on DEV)."""
    cfg = V2Config()
    p = Path(path) if path else V2_YAML
    if not p.exists():
        return cfg
    import yaml
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for k, v in (y.get("retrieval") or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, int(v))
    r = y.get("router") or {}
    cfg.router = bool(r.get("cascade", cfg.router))
    cfg.gap_level = int(r.get("gap_level", cfg.gap_level))
    cfg.verify_trigger = bool(r.get("verify_trigger", cfg.verify_trigger))
    cfg.max_claims = int((y.get("claims") or {}).get("max_claims", cfg.max_claims))
    cfg.calc_delegate = bool(y.get("calc_delegate", cfg.calc_delegate))
    cfg.verifier_role = str(y.get("verifier_role", cfg.verifier_role))
    ca = ((y.get("operating_points") or {}).get("calibrated_abstain") or {})
    if "abstain_below" in ca:        # applies only when abstention=true is requested (answer-always stays the default)
        cfg.abstain_below = int(ca["abstain_below"])
    return cfg


# --------------------------------------------------------------------------- factory (called from systems.get_system)
def build_otter_v2(llm: LLM, store: Any, flags: dict, *, name: str, cache: Optional[LRU], calc_factory=None) -> OtterV2:
    from otter.llm.interface import get_llm
    cfg = load_frozen_config()
    ab = {a: bool(flags.pop(a, False)) for a in V2_ABLATIONS}
    cfg.option_obligations = not ab["no_option_obligations"]
    cfg.option_queries = not ab["no_option_queries"]
    cfg.router = not ab["no_router"]
    cfg.gap_round = not ab["no_gap_round"]
    cfg.verified_citation = not ab["no_verified_citation"]
    cfg.ontology_hints = not ab["no_ontology_hints"]
    cfg.calc_delegate = not ab["no_calc_delegate"]
    for k in ("top_k_question", "top_k_option", "max_spans", "span_chars", "hint_facts", "ontology_spans", "gap_level",
              "abstain_below", "max_claims"):
        if k in flags:
            setattr(cfg, k, int(flags.pop(k)))
    if "verify_trigger" in flags:
        cfg.verify_trigger = bool(flags.pop("verify_trigger"))
    if "abstention" in flags:
        cfg.abstention = bool(flags.pop("abstention"))
    if ab["no_abstention"]:
        cfg.abstention = False
    if "verifier_role" in flags:
        cfg.verifier_role = str(flags.pop("verifier_role"))
    verifier = flags.pop("verifier_llm", None) or get_llm(cfg.verifier_role)
    suffix = "".join(f"-{a}" for a, on in ab.items() if on)
    return OtterV2(llm, verifier, store, cfg, name=name + suffix, llm_cache=cache,
                   calc_system=calc_factory() if (calc_factory and cfg.calc_delegate) else None)
