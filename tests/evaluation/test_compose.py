"""ViEduHop-Compose: deterministic parts (no LLM, no corpus)."""
import json
import subprocess
import sys
from pathlib import Path

from otter.agents.compose import (ComposeRequest, best_sentence, basic_checks, len_balance, parse_item)

ROOT = Path(__file__).resolve().parents[2]


def _req(**kw):
    d = dict(req_id="r1", subject="physics", grade=10, book_id="b", lesson_id="b:ch01:l01", level="thong_hieu",
             lesson_label="Bài 1: Test", book_title="Vật lí 10")
    d.update(kw)
    return ComposeRequest(**d)


def test_request_text_and_roundtrip():
    r = _req(focus="I. Mục", constraint="cần số liệu trong bảng")
    t = r.text()
    assert "Vật lí lớp 10" in t and "mục “I. Mục”" in t and "thông hiểu" in t and "cần số liệu trong bảng" in t
    assert ComposeRequest.from_dict(r.to_dict()).req_id == "r1"


def test_len_balance():
    assert len_balance({"A": "x" * 20, "B": "y" * 22, "C": "z" * 19, "D": "w" * 21})[0]
    assert not len_balance({"A": "x" * 20, "B": "y" * 60, "C": "z" * 19, "D": "w" * 21})[0]
    assert len_balance({"A": "1,0", "B": "2,5", "C": "3,0", "D": "10,5"})[0]       # short numeric options: absolute slack


def test_parse_item_places_key_and_checks():
    obj = {"question": "Đơn vị của lực trong SI là gì?", "correct": "Newton", "distractors": ["Joule", "Watt", "Pascal"],
           "explanation": "Lực đo bằng newton.", "bloom": "nhận biết"}
    it = parse_item(obj, "k")
    assert it and it["options"][it["key"]] == "Newton" and it["bloom"] == "nhan_biet"
    assert parse_item({**obj, "distractors": ["a", "b"]}, "k") is None
    chk = basic_checks(it, _req(), None)
    assert chk["schema_ok"] and chk["no_meta_ref"] and chk["basic_ok"]
    bad = dict(it, question="Theo đoạn trích trên, đơn vị của lực là gì?")
    assert not basic_checks(bad, _req(), None)["no_meta_ref"]


def test_best_sentence_is_verbatim():
    text = "Vật chuyển động thẳng đều. Gia tốc của vật ném xiên luôn bằng gia tốc trọng trường và hướng xuống dưới."
    s = best_sentence(text, "gia tốc của vật ném xiên bằng gia tốc trọng trường")
    assert s and s in text
    assert best_sentence(text, "hoàn toàn không liên quan đến văn bản") is None


