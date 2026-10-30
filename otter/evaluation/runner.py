"""Benchmark runner: system × backbone × bench -> runs/<exp>/<system>/<backbone>/answers.jsonl.

System specs:
  <name>                 -> otter.agents.systems.get_system(name, backbone=..., **flags)   (WS3)
  baseline:<name>        -> otter.baselines.<name> (get_system / make_system / build_system)  (WS4)
  external:<answers.jsonl> -> answers produced by a separate CLI (WS4); validated and copied
  dummy:<first|random|oracle|abstain|eval_direct> -> pipeline tests / sanity checks
Resumable: items already in answers.jsonl (without error) are skipped; `retry_failed` re-runs failed ones.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import platform
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from otter.evaluation.exclusion import excluded_spans
from otter.evaluation.mcq import MCQItem
from otter.evaluation.text import sha1
from otter.schemas.reasoning import FinalAnswer

log = logging.getLogger("otter.eval")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def system_slug(spec: str) -> str:
    if spec.startswith("external:"):
        return Path(spec.split(":", 1)[1]).parent.parent.name or "external"
    return spec.replace("dummy:", "dummy_").replace("baseline:", "").replace("/", "_").replace(":", "_")


def resolve_system(spec: str, *, backbone: str, flags: dict[str, Any], bench_index: dict, corpus=None, llm=None):
    if spec.startswith("dummy:"):
        from otter.evaluation.dummy_systems import get_dummy
        return get_dummy(spec.split(":", 1)[1], backbone=backbone, bench=bench_index, corpus=corpus, llm=llm)
    if spec == "eval_direct":
        from otter.evaluation.dummy_systems import get_dummy
        return get_dummy("eval_direct", backbone=backbone, llm=llm)
    if spec.startswith("baseline:"):
        mod = importlib.import_module(f"otter.baselines.{spec.split(':', 1)[1]}")
        for fn in ("get_system", "make_system", "build_system"):
            if hasattr(mod, fn):
                return getattr(mod, fn)(backbone=backbone, **flags)
        raise AttributeError(f"otter.baselines.{spec} exposes no get_system/make_system/build_system")
    if spec == "ircot":
        from otter.evaluation.rag_variants import get_ircot
        return get_ircot(backbone=backbone, **flags)
    if spec in ("rag_bm25", "rag_dense", "rag_hybrid", "text_rag"):
        from otter.evaluation.rag_variants import get_system as get_rag  # text_rag = alias of rag_hybrid (top_k=5)
        return get_rag("rag_hybrid" if spec == "text_rag" else spec, backbone=backbone, **flags)
    from otter.agents.systems import get_system  # WS3
    return get_system(spec, backbone=backbone, **flags)


def _read_done(path: Path) -> dict[str, FinalAnswer]:
    done: dict[str, FinalAnswer] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    fa = FinalAnswer.model_validate_json(line)
                except Exception:
                    continue
                done[fa.question_id] = fa
    return done


def _ok(fa: FinalAnswer) -> bool:
    return fa.error is None and fa.status != "failed"


def run_system(system, items: list[MCQItem], out_dir: Path, *, workers: int = 4, retry_failed: bool = False,
               meta: Optional[dict] = None, progress_every: int = 10, abstain_option: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ans_path = out_dir / "answers.jsonl"
    done = _read_done(ans_path)
    todo = [it for it in items if it.id not in done or (retry_failed and not _ok(done[it.id]))]
    log.info("%s: %d items, %d already done, %d to run (workers=%d)", out_dir, len(items), len(items) - len(todo),
             len(todo), workers)
    lock = threading.Lock()
    name = getattr(system, "name", type(system).__name__)

    def one(it: MCQItem) -> FinalAnswer:
        t = time.perf_counter()
        try:
            # OTTER-Abstain: hide the item's excluded spans from retrieval during this call (no-op for other benchmarks)
            with excluded_spans(it.excluded_span_ids):
                fa = system.answer(it.as_system_input(abstain_option=True) if abstain_option else it.as_system_input())
            if not isinstance(fa, FinalAnswer):
                fa = FinalAnswer.model_validate(fa)
        except Exception as e:
            fa = FinalAnswer(question_id=it.id, system=name, answer="", status="failed",
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}")
        fa.question_id = it.id
        fa.tool_usage = dict(fa.tool_usage or {})
        fa.tool_usage["wall_latency_ms"] = (time.perf_counter() - t) * 1000
        with lock, open(ans_path, "a", encoding="utf-8") as f:
            f.write(fa.model_dump_json() + "\n")
        return fa

    meta = dict(meta or {})
    meta.update(started_at=meta.get("started_at") or now(), n_items=len(items), n_todo=len(todo))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    n_fail = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            fa = fu.result()
            n_fail += not _ok(fa)
            if k % progress_every == 0 or k == len(futs):
                el = time.perf_counter() - t0
                log.info("%d/%d done, %d failed, %.1fs elapsed, %.2fs/item", k, len(futs), n_fail, el, el / k)
    final = _read_done(ans_path)
    meta.update(finished_at=now(), n_answers=len(final), n_failed=sum(1 for v in final.values() if not _ok(v)),
                n_missing=sum(1 for it in items if it.id not in final))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def import_external(path: Path, items: list[MCQItem], out_dir: Path, *, meta: Optional[dict] = None) -> dict:
    """Validate an externally produced answers.jsonl and copy the bench subset into out_dir."""
    ids = {it.id for it in items}
    good: dict[str, FinalAnswer] = {}
    bad = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            fa = FinalAnswer.model_validate_json(line)
        except Exception:
            bad += 1
            continue
        if fa.question_id in ids:
            good[fa.question_id] = fa
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "answers.jsonl"
    if Path(path).resolve() != dst.resolve():
        with open(dst, "w", encoding="utf-8") as f:
            for it in items:
                if it.id in good:
                    f.write(good[it.id].model_dump_json() + "\n")
    meta = dict(meta or {})
    meta.update(external_source=str(path), external_sha1=sha1(Path(path).read_text(encoding="utf-8")),
                invalid_lines=bad, n_items=len(items), n_answers=len(good),
                n_missing=len(ids - set(good)), imported_at=now())
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def base_meta(spec: str, backbone: str, bench: Path, flags: dict) -> dict:
    from otter.llm.interface import load_config
    try:
        cfg = load_config()["llm"].get(backbone, {})
    except Exception:
        cfg = {}
    return {"system_spec": spec, "system": system_slug(spec), "backbone": backbone, "backbone_model": cfg.get("model"),
            "backbone_url": cfg.get("base_url"), "flags": flags, "bench": str(bench),
            "bench_sha1": sha1(Path(bench).read_text(encoding="utf-8")), "host": platform.node(),
            "started_at": now()}
