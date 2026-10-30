import json
from pathlib import Path
from otter.evaluation import run_provenance as rp


def test_guard_moves_dir_on_method_change_and_keeps_history(tmp_path, monkeypatch):
    out = tmp_path / "runs" / "exp" / "sysA" / "main"
    out.mkdir(parents=True)
    (out / "answers.jsonl").write_text('{"x":1}\n')
    (out / "run_meta.json").write_text(json.dumps({"a": 1}))
    monkeypatch.setattr(rp, "method_fingerprint", lambda: "v1")
    assert rp.guard_rundir(out)["moved_to"] is None                      # legacy dir adopted, marker written
    assert (out / ".method_sha256_v2").read_text() == "v1"
    assert len((out / "run_meta_history.jsonl").read_text().splitlines()) == 1
    assert rp.guard_rundir(out)["moved_to"] is None                      # same method: resume keeps the dir
    monkeypatch.setattr(rp, "method_fingerprint", lambda: "v2")
    info = rp.guard_rundir(out, now="T")
    dest = Path(info["moved_to"])
    assert dest == tmp_path / "runs" / "exp_prev_T" / "sysA" / "main" and (dest / "answers.jsonl").exists()
    assert not (out / "answers.jsonl").exists() and (out / ".method_sha256_v2").read_text() == "v2"
