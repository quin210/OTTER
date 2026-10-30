"""`verify_evidence`: independent checks of a claim (or a batch of choice claims) against cited ids.

Three separately reported layers (spec 4 / 5.3):
  structural      -- every cited assertion / rule / span / packet id exists (any unknown id => fail);
  source_link     -- cited spans exist AND are linked to the cited assertions / rules
                     (pass | partial | unlinked | fail);
  semantic_support-- HEURISTIC: lexical coverage of the claim by the evidence text, optionally
                     combined with an LLM judge whose quote must occur verbatim in a real span.
Overall: supported only if structural=pass, >=1 existing evidence span and semantic=supported;
contradicted only with an LLM-judged, quote-verified opposing span; otherwise undetermined.
"""
from __future__ import annotations

from typing import Any, Optional

from otter.reasoning.text import contains_quote, coverage, tokens, truncate

MAX_JUDGE_SPANS = 8


def _as_list(x: Any) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [x]
    return [str(i) for i in x if i]


def verify_evidence(ctx: Any, claim: Optional[str] = None, claims: Optional[list[dict[str, str]]] = None,
                    assertion_ids: Any = None, rule_ids: Any = None, span_ids: Any = None, packet_ids: Any = None,
                    question: Optional[str] = None, use_judge: bool = True,
                    claim_spans: Optional[dict[str, list[str]]] = None) -> dict[str, Any]:
    store = ctx.store
    if not claim and not claims:
        return {"status": "error", "reason": "no_claim"}
    batch = [{"id": str(c.get("id")), "text": str(c.get("text", ""))} for c in (claims or [])]
    if claim:
        batch = [{"id": "claim", "text": str(claim)}] + batch
    a_ids, r_ids, s_ids, p_ids = map(_as_list, (assertion_ids, rule_ids, span_ids, packet_ids))
    for v in (claim_spans or {}).values():
        s_ids.extend(x for x in _as_list(v) if x not in s_ids)

    # ---- structural -------------------------------------------------------------------
    facts = {a: store.get_fact(a) for a in a_ids}
    rules = {r: store.get_rule(r) for r in r_ids}
    found_spans = {s.span_id: s for s in store.get_spans(s_ids)} if s_ids else {}
    unknown = ([a for a, f in facts.items() if f is None] + [r for r, x in rules.items() if x is None]
               + [s for s in s_ids if s not in found_spans])
    if p_ids and ctx.workspace is not None:
        unknown += [p for p in p_ids if ctx.workspace.get(p) is None]
    structural = {"status": "fail" if unknown else "pass", "unknown_ids": unknown}

    # ---- source links -----------------------------------------------------------------
    linked: set[str] = set()
    for obj in list(facts.values()) + list(rules.values()):
        if obj is not None:
            linked.update(obj.source_span_ids)
    linked_spans = {s.span_id: s for s in store.get_spans(sorted(linked))} if linked else {}
    dangling_links = sorted(linked - set(linked_spans))
    cited_existing = [s for s in s_ids if s in found_spans]
    if not cited_existing and not linked_spans:
        link_status = "fail"
    elif not (a_ids or r_ids):
        link_status = "unlinked"           # direct span citations without an assertion/rule
    elif cited_existing and not all(s in linked for s in cited_existing):
        link_status = "partial"
    elif dangling_links:
        link_status = "partial"
    else:
        link_status = "pass"
    source_link = {"status": link_status, "linked_span_ids": sorted(set(linked_spans)),
                   "unlinked_span_ids": [s for s in cited_existing if s not in linked],
                   "missing_span_ids": [s for s in s_ids if s not in found_spans] + dangling_links}

    pool: dict[str, Any] = {}
    for s in cited_existing:
        pool[s] = found_spans[s]
    for s, sp in linked_spans.items():
        pool.setdefault(s, sp)
    texts = {sid: sp.text for sid, sp in pool.items()}
    for f in facts.values():                  # assertion text counts as evidence text only via its spans' ids
        if f is not None and f.source_span_ids:
            k = f.source_span_ids[0]
            if k in texts:
                texts[k] = texts[k] + " " + f.text

    # ---- semantic support (heuristic) ---------------------------------------------------
    results: dict[str, dict[str, Any]] = {}
    for c in batch:
        allowed = set(_as_list((claim_spans or {}).get(c["id"]))) or set(texts)
        best_sid, best = None, 0.0
        for sid in allowed:
            if sid not in texts:
                continue
            sc = coverage(c["text"], texts[sid])
            if sc > best:
                best_sid, best = sid, sc
        results[c["id"]] = {"lexical_score": round(best, 4), "lexical_best_span_id": best_sid,
                            "semantic": "supported" if best >= ctx.support_threshold else "undetermined",
                            "method": "lexical", "span_ids": [best_sid] if best_sid and best >= ctx.support_threshold else []}

    judged = False
    if use_judge and ctx.judge_llm is not None and pool and "judge" not in ctx.disabled:
        judged = _judge(ctx, batch, pool, question, results)
    if question and not judged:
        # "is this choice the answer?" cannot be decided by lexical overlap alone
        for r in results.values():
            if r["semantic"] == "supported":
                r["semantic"], r["span_ids"], r["method"] = "undetermined", [], "lexical(choice_mode_not_decisive)"

    out_claims = []
    for c in batch:
        r = results[c["id"]]
        sem = r["semantic"]
        if structural["status"] != "pass" or not pool:
            status = "undetermined"
        elif sem == "supported":
            status = "supported"
        elif sem == "contradicted":
            status = "contradicted"
        else:
            status = "undetermined"
        out_claims.append({"id": c["id"], "text": truncate(c["text"], 200), "status": status, "semantic_support": sem,
                           "method": r["method"], "lexical_score": r["lexical_score"],
                           "evidence_span_ids": r["span_ids"], "quote": r.get("quote"),
                           "judge_note": r.get("judge_note")})
    res: dict[str, Any] = {
        "structural": structural, "source_link": source_link,
        "semantic_support_method": "lexical+llm_judge" if judged else "lexical",
        "semantic_support_is_heuristic": True,
        "verified_span_ids": sorted(pool) if structural["status"] == "pass" else [],
    }
    if claim and not claims:
        main = out_claims[0]
        res.update(status=main["status"], semantic_support=main["semantic_support"], evidence_span_ids=main["evidence_span_ids"],
                   lexical_score=main["lexical_score"], quote=main["quote"])
    else:
        res.update(status="ok", claims=out_claims)
    return res


def _judge(ctx: Any, batch: list[dict[str, str]], pool: dict[str, Any], question: Optional[str],
           results: dict[str, dict[str, Any]]) -> bool:
    from otter.agents.prompts import JUDGE_SYSTEM, judge_user   # local import: avoid package cycle
    from otter.llm.interface import parse_json_loose

    # rank spans by overlap with all claims (+ question), keep the best few
    q = " ".join([question or ""] + [c["text"] for c in batch])
    ranked = sorted(pool.items(), key=lambda kv: -coverage(q, kv[1].text))[:MAX_JUDGE_SPANS]
    spans = [{"span_id": sid, "text": truncate(sp.text, 700)} for sid, sp in ranked]
    texts = {sid: sp.text for sid, sp in ranked}
    msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_user(batch, spans, question)}]
    data = None
    for attempt in range(2):                                  # one retry on an unparsable reply
        try:
            r = ctx.judge_llm.chat(msgs, temperature=0.0, max_tokens=200 + 120 * len(batch))
            data = parse_json_loose(r.text)
            break
        except ValueError as e:
            err = e
            msgs = msgs + [{"role": "assistant", "content": r.text[:1500]},
                           {"role": "user", "content": "Phản hồi trước không phải JSON hợp lệ. Trả lời lại bằng đúng "
                                                       "MỘT đối tượng JSON theo mẫu."}]
        except Exception as e:                                # judge unavailable
            err = e
            break
    if data is None:                                          # -> lexical only (choice mode: undetermined)
        for v in results.values():
            v["judge_note"] = f"judge_error: {type(err).__name__}"
        return False
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    by_id = {str(v.get("id")): v for v in verdicts if isinstance(v, dict)}
    for c in batch:
        v = by_id.get(c["id"])
        r = results[c["id"]]
        r["method"] = "lexical+llm_judge"
        if not v:
            r["semantic"], r["span_ids"], r["judge_note"] = "undetermined", [], "no_verdict"
            continue
        verdict = str(v.get("verdict", "")).strip().lower()
        sid, quote = v.get("span_id"), (v.get("quote") or "").strip()
        if verdict not in ("supported", "contradicted"):
            r["semantic"], r["span_ids"], r["judge_note"] = "undetermined", [], "not_enough_info"
            continue
        if sid not in texts or not quote or not contains_quote(quote, texts[sid]):
            # the judge's evidence must be real: quote has to occur in the cited span
            hit = next((s for s, t in texts.items() if quote and contains_quote(quote, t)), None)
            if hit is None:
                r["semantic"], r["span_ids"], r["judge_note"] = "undetermined", [], f"{verdict}_but_quote_not_in_spans"
                continue
            sid = hit
        if verdict == "supported" and question is None and not tokens(c["text"]):
            r["semantic"], r["judge_note"] = "undetermined", "empty_claim"
            continue
        extra = []
        for m in (v.get("more") or [])[:2] if isinstance(v.get("more"), list) else []:
            if not isinstance(m, dict):
                continue
            msid, mq = m.get("span_id"), (m.get("quote") or "").strip()
            if msid in texts and msid != sid and mq and contains_quote(mq, texts[msid]):
                extra.append(msid)                            # every extra hop must also be quote-verified
        r["semantic"], r["span_ids"], r["quote"] = verdict, [sid] + extra, quote[:300]
    return True
