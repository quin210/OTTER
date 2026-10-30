"""Reproducibility fingerprint stored in every run_meta.json."""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_GLOBS = ["otter/**/*.py", "scripts/eval/*.py", "scripts/baselines/*.py", "configs/*.yaml"]


def _sha(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def code_fingerprint() -> dict:
    files = sorted({p for g in CODE_GLOBS for p in ROOT.glob(g) if p.is_file() and "__pycache__" not in p.parts})
    cfg = [p for p in files if p.suffix == ".yaml"]
    return {"code_sha256": _sha(files), "configs_sha256": _sha(cfg), "n_code_files": len(files),
            "python": sys.version.split()[0], "platform": platform.platform()}


METHOD_GLOBS = ["otter/agents/**/*.py", "otter/reasoning/**/*.py", "otter/tools/**/*.py", "otter/evaluation/rag_variants.py",
                "otter/evaluation/exclusion.py", "configs/otter_v2.yaml", "configs/default.yaml"]
MARKER = ".method_sha256_v2"


def method_fingerprint() -> str:
    """Hash of the code that decides an answer (agents, reasoning, tools, RAG variants, exclusion hook, configs). Builders, storage, scoring,
    baselines (WS4), logging and scripts are excluded so unrelated edits by other workstreams do not invalidate finished runs."""
    files = sorted(p for g in METHOD_GLOBS for p in ROOT.glob(g) if p.is_file() and "__pycache__" not in p.parts)
    return _sha(files)


def guard_rundir(out: Path, *, now: str | None = None) -> dict:
    """Never silently mix or overwrite results of different code versions.

    * run_meta.json of an earlier start is appended to run_meta_history.jsonl (resuming rewrites run_meta.json).
    * `<out>/.method_sha256` records the method fingerprint of the run. If a directory with answers was produced by a different
      method version, the WHOLE directory is moved to runs/<exp>_prev_<stamp>/<system>/<backbone> (outside the exp so aggregates
      ignore it; nothing is deleted) and the run starts fresh. Legacy directories without the marker are adopted as-is.
    """
    import json, shutil, time
    out = Path(out)
    stamp = now or time.strftime("%Y%m%d_%H%M%S")
    cur = method_fingerprint()
    info = {"moved_to": None, "method_sha256": cur}
    marker, ans, meta = out / MARKER, out / "answers.jsonl", out / "run_meta.json"
    if ans.exists() and marker.exists() and marker.read_text().strip() != cur:
        try:
            exp_dir, rel = out.parents[1], out.relative_to(out.parents[1])
            dest = exp_dir.parent / f"{exp_dir.name}_prev_{stamp}" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(dest))
            info["moved_to"] = str(dest)
        except Exception:
            raise
    out.mkdir(parents=True, exist_ok=True)
    if meta.exists():
        with open(out / "run_meta_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(json.loads(meta.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
    marker.write_text(cur)
    return info
