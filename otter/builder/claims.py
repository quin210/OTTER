"""Claim comparison for knowledge ingestion: does a NEW item agree with / conflict with an OLD item?

Two tiers, cheapest first (thresholds are fixed a priori, not tuned on any experiment):

  T0  exact normalised key                                           -> agrees
  T1  lexical + value check (no LLM)
        * content-token Jaccard of the two claims with numbers removed (`j`)
        * the set of numeric values (numbers, years) of each claim
        agrees     j >= AGREE_J and equal value sets
        conflicts  j >= CONFLICT_J, both claims carry numbers, same number of numbers, value sets differ on both sides
                   (same wording around, different number/year) -> "high-confidence" lexical conflict
        uncertain  plausibly the same slot (j >= SLOT_J or dense cosine >= SLOT_COS) but not decided above, or the two claims
                   differ in a capitalised entity while the rest agrees (aliases exist: "Nguyễn Huệ" = "Quang Trung")
        unrelated  everything else (different aspect of the concept) -> contributes nothing
  T2  small LLM (role `small`) for `uncertain` pairs, optionally also to CONFIRM lexical conflicts.
        Labels: same | conflict | unrelated | unclear.  `unclear` and failed calls -> ambiguous (never forced).

A new item is aggregated over its candidate old items with priority agrees > conflicts > ambiguous > new.
Two rule sets: "v1" (first release, used for the first synthetic run) and "v2" (default; added AFTER seeing the failures of that run:
containment-based slot detection for claims of unequal length, a time-point guard for statistics of different years, a negation
guard, and a judge prompt that treats different time points as unrelated).  Both are kept so the change can be reported as post hoc.

Every verdict carries the evidence (tier, slot, old/new value, j, cosine) so the review list is auditable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from otter.llm.interface import LLM, parse_json_loose
from otter.storage.textutil import VI_STOPWORDS, nfc, norm_key, syllables

log = logging.getLogger("otter.builder.claims")
JUDGE_VERSION = "claimjudge-v2"

AGREE_J = 0.70
CONFLICT_J = 0.75
SLOT_J = 0.45
SLOT_COS = 0.90
COVER_SLOT = 0.80                      # v2: containment (overlap coefficient) of the shorter claim

NUM_RE = re.compile(r"(?<![\w])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])")
CAP_RE = re.compile(r"(?<![.!?:;\n])\s((?:[A-ZÀ-Ỹ][\wà-ỹ]+)(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+)*)")


@dataclass
class PairVerdict:
    label: Literal["agrees", "conflicts", "ambiguous", "unrelated"]
    tier: str                                # exact | lexical | llm | lexical_pending
    jaccard: float = 0.0
    cosine: Optional[float] = None
    slot: str = ""
    old_value: str = ""
    new_value: str = ""
    note: str = ""
    llm_label: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


# --------------------------------------------------------------------------------------------- values
def parse_number(tok: str) -> Optional[float]:
    t = tok.strip()
    try:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):          # 1.000.000 or 1.000,5 (Vietnamese thousand sep)
            return float(t.replace(".", "").replace(",", "."))
        return float(t.replace(",", "."))
    except ValueError:
        return None


def number_values(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUM_RE.finditer(nfc(text)):
        v = parse_number(m.group(1))
        if v is not None:
            out.append((m.group(1), v))
    return out


def _entities(text: str) -> set[str]:
    return {norm_key(m.group(1)) for m in CAP_RE.finditer(". " + nfc(text)) if len(m.group(1)) > 2}


def content_tokens(text: str) -> set[str]:
    """Syllables without numbers/stopwords (the 'wording' of a claim)."""
    return {t for t in syllables(text) if t not in VI_STOPWORDS and not t.isdigit()}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _eq_values(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> bool:
    sa = sorted(round(v, 9) for _, v in a)
    sb = sorted(round(v, 9) for _, v in b)
    return sa == sb


NEG_CUES = {"không", "chưa", "chẳng", "chả"}
YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")


def _is_year(tok: str) -> bool:
    return bool(YEAR_RE.match(tok.strip()))


def containment(a: set, b: set) -> float:
    m = min(len(a), len(b))
    return len(a & b) / m if m >= 3 else 0.0


def _neg(text: str) -> bool:
    return bool(NEG_CUES & set(syllables(text)))


def formula_key(f: Optional[str]) -> str:
    return re.sub(r"\s+", "", (f or "")).replace("*", "").replace("×", "").replace("·", "").casefold()


# --------------------------------------------------------------------------------------------- lexical tier
def compare_lexical(old: str, new: str, *, cosine: Optional[float] = None, old_formula: Optional[str] = None,
                    new_formula: Optional[str] = None, rules: str = "v2") -> PairVerdict:
    v2 = rules == "v2"
    if norm_key(old) == norm_key(new):
        return PairVerdict("agrees", "exact", 1.0, cosine, note="identical_normalised_text")
    jo, jn = content_tokens(old), content_tokens(new)
    j = round(jaccard(jo, jn), 4)
    cov = round(containment(jo, jn), 4) if v2 else 0.0
    vo, vn = number_values(old), number_values(new)
    same_vals = _eq_values(vo, vn)
    eo, en = _entities(old), _entities(new)
    fo, fn = formula_key(old_formula), formula_key(new_formula)
    slot_lex = j >= SLOT_J or (v2 and cov >= COVER_SLOT)
    if fo and fn and fo != fn and slot_lex:
        return PairVerdict("ambiguous", "lexical_pending", j, cosine, slot="formula", old_value=old_formula or "",
                           new_value=new_formula or "", note="formula_differs_needs_judge")
    if v2 and _neg(old) != _neg(new) and slot_lex:
        return PairVerdict("ambiguous", "lexical_pending", j, cosine, slot="negation", note="negation_differs_needs_judge")
    if v2 and vo and vn and not same_vals:
        yo = {round(v, 9) for t, v in vo if _is_year(t)}
        yn = {round(v, 9) for t, v in vn if _is_year(t)}
        oo = {round(v, 9) for t, v in vo if not _is_year(t)}
        on = {round(v, 9) for t, v in vn if not _is_year(t)}
        if yo and yn and yo != yn and oo and on and oo != on:  # year AND value both differ: another time point of a statistic
            return PairVerdict("unrelated", "lexical", j, cosine, note="different_time_point")
    if j >= AGREE_J and same_vals and eo == en:
        return PairVerdict("agrees", "lexical", j, cosine, note="same_wording_same_values")
    if v2 and cov >= 0.9 and same_vals and eo == en and 4 <= len(jn) <= len(jo):      # new claim is contained in the old one
        return PairVerdict("agrees", "lexical", j, cosine, note="contained_same_values")
    if slot_lex and same_vals and (eo - en) and (en - eo):
        return PairVerdict("ambiguous", "lexical_pending", j, cosine, slot="entity", old_value=", ".join(sorted(eo - en)),
                           new_value=", ".join(sorted(en - eo)), note="entity_differs_needs_judge")
    if j >= CONFLICT_J and vo and vn and len(vo) == len(vn) and not same_vals:
        so = {round(v, 9) for _, v in vo}
        sn = {round(v, 9) for _, v in vn}
        ov = [t for t, v in vo if round(v, 9) not in sn]
        nv = [t for t, v in vn if round(v, 9) not in so]
        if ov and nv:
            return PairVerdict("conflicts", "lexical_pending", j, cosine, slot="number", old_value=", ".join(ov),
                               new_value=", ".join(nv), note="same_wording_different_number")
    if slot_lex or (cosine is not None and cosine >= SLOT_COS and j >= 0.25):
        return PairVerdict("ambiguous", "lexical_pending", j, cosine,
                           note="same_slot_plausible" if j >= SLOT_J else "contained_slot_plausible")
    return PairVerdict("unrelated", "lexical", j, cosine)


# --------------------------------------------------------------------------------------------- LLM tier
class _PJ(BaseModel):
    id: int
    label: Literal["same", "conflict", "unrelated", "unclear"]
    slot: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None


class _PJs(BaseModel):
    judgments: list[_PJ]


JUDGE_SYSTEM_V1 = """Bạn là người kiểm tra tính nhất quán giữa hai sách giáo khoa (hoặc hai lần xuất bản) về cùng một môn học. Với mỗi cặp (A = tri thức đã có, B = tri thức mới), chỉ dựa vào hai câu/nội dung được đưa ra (không dùng kiến thức bên ngoài), hãy gán đúng một nhãn:
- "same": A và B nêu CÙNG một ý (có thể diễn đạt khác nhau, hoặc B chi tiết hơn nhưng không trái với A), cùng giá trị nếu có số liệu/mốc thời gian/tên riêng.
- "conflict": A và B nói về CÙNG một đối tượng và CÙNG một thuộc tính/khía cạnh nhưng GIÁ TRỊ hoặc kết luận KHÁC NHAU (khác số liệu, năm, tên, định nghĩa, phân loại, công thức không tương đương) hoặc khẳng định trái ngược nhau.
- "unrelated": A và B nói về đối tượng hoặc khía cạnh KHÁC nhau (không so sánh được, không mâu thuẫn), kể cả khi có từ ngữ giống nhau.
- "unclear": không đủ thông tin để quyết định.
Lưu ý: khác đơn vị nhưng cùng giá trị (5 km và 5000 m) là "same"; hai tên gọi khác nhau của cùng một người/vật là "same"; số liệu của hai đối tượng khác nhau là "unrelated".
Nếu nhãn là "conflict", điền "slot" (thuộc tính bị so sánh, ngắn gọn), "old_value" (giá trị trong A), "new_value" (giá trị trong B).
VÍ DỤ (bịa, chỉ để minh hoạ định dạng):
1. A: "Sông Alpha dài khoảng 1 200 km." B: "Chiều dài sông Alpha là khoảng 1 350 km." -> conflict (slot: chiều dài sông Alpha; 1 200 km; 1 350 km)
2. A: "Sông Alpha dài khoảng 1 200 km." B: "Sông Beta dài khoảng 800 km." -> unrelated
3. A: "Tốc độ ánh sáng trong chân không xấp xỉ 3.10^8 m/s." B: "Ánh sáng truyền trong chân không với tốc độ khoảng 300 000 km/s." -> same
Trả về JSON: {"judgments": [{"id": <số thứ tự cặp>, "label": "same|conflict|unrelated|unclear", "slot": str|null, "old_value": str|null, "new_value": str|null}]} cho TẤT CẢ các cặp."""


JUDGE_SYSTEM = JUDGE_SYSTEM_V1.replace(
    "Lưu ý: khác đơn vị",
    "Lưu ý: số liệu của CÁC NĂM / MỐC THỜI GIAN KHÁC NHAU (ví dụ sản lượng năm 2010 và năm 2020) là các điểm dữ liệu khác nhau, nhãn là \"unrelated\" chứ không phải \"conflict\"; "
    "một câu có thêm ý nhưng không trái với câu kia là \"same\". Khác đơn vị")


class PairJudge:
    """Batched, cached small-LLM adjudication of claim pairs."""

    def __init__(self, llm: Optional[LLM], cache_dir: Optional[Path] = None, *, batch_size: int = 8, workers: int = 3,
                 max_chars: int = 700, rules: str = "v2"):
        self.system, self.version = (JUDGE_SYSTEM, "claimjudge-v2") if rules == "v2" else (JUDGE_SYSTEM_V1, "claimjudge-v1")
        self.llm, self.cache_dir, self.batch_size, self.workers, self.max_chars = llm, cache_dir, batch_size, workers, max_chars
        self.stats = {"llm_calls": 0, "cached_batches": 0, "failed_batches": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "pairs": 0, "recorded_prompt_tokens_cached": 0}
        self.schema = _PJs.model_json_schema()
        self._lock = threading.Lock()

    def _cache(self, key: str) -> Optional[Path]:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _judge_batch(self, pairs: list[tuple[str, str]]) -> list[Optional[_PJ]]:
        lines = []
        for i, (a, b) in enumerate(pairs, 1):
            lines.append(f"{i}. A: \"{a.strip()[: self.max_chars]}\"\n   B: \"{b.strip()[: self.max_chars]}\"")
        user = "CÁC CẶP:\n" + "\n".join(lines)
        model = getattr(self.llm, "model", "")
        key = hashlib.sha256(f"{self.version}\x1f{model}\x1f{user}".encode("utf-8")).hexdigest()
        cp = self._cache(key)
        if cp and cp.exists():
            try:
                d = json.loads(cp.read_text(encoding="utf-8"))
                with self._lock:
                    self.stats["cached_batches"] += 1
                    self.stats["recorded_prompt_tokens_cached"] += d.get("prompt_tokens", 0)
                return [(_PJ.model_validate(x) if x else None) for x in d["labels"]]
            except Exception:
                pass
        if self.llm is None:
            return [None] * len(pairs)
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        for attempt in range(4):
            try:
                r = self.llm.chat(msgs, temperature=0.0, max_tokens=120 * len(pairs) + 64, json_schema=self.schema)
                with self._lock:
                    self.stats["llm_calls"] += 1
                    self.stats["prompt_tokens"] += r.prompt_tokens
                    self.stats["completion_tokens"] += r.completion_tokens
                obj = _PJs.model_validate(parse_json_loose(r.text))
                out: list[Optional[_PJ]] = [None] * len(pairs)
                for j in obj.judgments:
                    if 1 <= j.id <= len(pairs):
                        out[j.id - 1] = j
                if cp:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_text(json.dumps({"labels": [x.model_dump() if x else None for x in out], "model": model,
                                              "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens},
                                             ensure_ascii=False), encoding="utf-8")
                return out
            except Exception as e:
                log.debug("pair-judge batch failed (%s): %s", attempt, e)
                time.sleep(min(30, 3 * (attempt + 1)) if not isinstance(e, ValueError) else 0.5)
        with self._lock:
            self.stats["failed_batches"] += 1
        return [None] * len(pairs)

    def judge(self, pairs: list[tuple[str, str]]) -> list[Optional[_PJ]]:
        """Judge unique (old, new) text pairs; order preserved; None = no usable answer."""
        uniq = list(dict.fromkeys(pairs))
        self.stats["pairs"] += len(uniq)
        batches = [uniq[i:i + self.batch_size] for i in range(0, len(uniq), self.batch_size)]
        res: dict[tuple[str, str], Optional[_PJ]] = {}
        if batches:
            with ThreadPoolExecutor(max_workers=max(1, min(self.workers, 6))) as ex:
                futs = {ex.submit(self._judge_batch, b): b for b in batches}
                for f in as_completed(futs):
                    for p, r in zip(futs[f], f.result()):
                        res[p] = r
        return [res.get(p) for p in pairs]


def apply_judgment(v: PairVerdict, j: Optional[_PJ]) -> PairVerdict:
    """Turn a pending lexical verdict + LLM judgment into the final verdict."""
    if j is None:
        return PairVerdict("ambiguous", "llm_failed", v.jaccard, v.cosine, v.slot, v.old_value, v.new_value,
                           note=f"{v.note}; no_llm_answer")
    lab = {"same": "agrees", "conflict": "conflicts", "unrelated": "unrelated", "unclear": "ambiguous"}[j.label]
    note = v.note
    if v.label == "conflicts" and lab != "conflicts":
        note += "; lexical_conflict_overridden_by_llm"
    return PairVerdict(lab, "llm", v.jaccard, v.cosine, j.slot or v.slot, j.old_value or v.old_value,
                       j.new_value or v.new_value, note=note, llm_label=j.label)


def finalize_pending(v: PairVerdict, *, lexical_only: bool) -> PairVerdict:
    """Without an LLM: auto-conflicts on same-wording number changes stay conflicts (flagged), everything else uncertain
    stays ambiguous (never forced)."""
    if v.tier != "lexical_pending":
        return v
    if v.label == "conflicts":
        return PairVerdict("conflicts", "lexical", v.jaccard, v.cosine, v.slot, v.old_value, v.new_value, note=v.note)
    return PairVerdict("ambiguous", "lexical", v.jaccard, v.cosine, v.slot, v.old_value, v.new_value, note=v.note)


# --------------------------------------------------------------------------------------------- aggregation
PRIORITY = {"agrees": 3, "conflicts": 2, "ambiguous": 1}


def aggregate(pairs: list[tuple[str, PairVerdict]]) -> tuple[str, Optional[str], Optional[PairVerdict]]:
    """[(old_item_id, verdict)] -> (item verdict in agrees|conflicts|ambiguous|new, matched old id, deciding pair)."""
    best: Optional[tuple[str, PairVerdict]] = None
    for oid, v in pairs:
        if v.label == "unrelated":
            continue
        if best is None or PRIORITY[v.label] > PRIORITY[best[1].label] or \
                (PRIORITY[v.label] == PRIORITY[best[1].label] and v.jaccard > best[1].jaccard):
            best = (oid, v)
    if best is None:
        return "new", None, None
    return best[1].label, best[0], best[1]
