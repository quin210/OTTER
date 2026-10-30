"""Shared helpers for the ViEduHop-Typed tier (benchmarks/typed/typed600.jsonl): answer normalisation, accepted variants for
fill-in-the-blank keys, and the fixed prompts. No LLM calls here. New file; imports nothing that is frozen."""
from __future__ import annotations

import re
import unicodedata

PROVENANCE_STATEMENT = "reviewed by 5 teachers (stated by the authors; records pending)"
AUTHORSHIP_STATEMENT = "authored by the research group (stated by the user on 27 Oct 2026)"
REVIEW_STATEMENT_ATTRIBUTION = "stated by the user on 27 Oct 2026: the questions were authored by the research group and reviewed by 5 teachers"

# ---------------------------------------------------------------- Vietnamese orthography normalisation
_TONE_PLACE = {}
# 'hoá' -> 'hóa', 'khoẻ' -> 'khỏe', 'thuỷ' -> 'thủy' (tone on the first vowel of oa/oe/uy)
for old, new in (("oá", "óa"), ("oà", "òa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
                 ("oé", "óe"), ("oè", "òe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
                 ("uý", "úy"), ("uỳ", "ùy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy")):
    _TONE_PLACE[old] = new
_Y2I = {"ý": "í", "ỳ": "ì", "ỷ": "ỉ", "ỹ": "ĩ", "ỵ": "ị"}      # kí / ký, lí / lý

# token-level canonical forms (both the key and the prediction are mapped; so 'axit' == 'acid' etc.)
CANON = {"axit": "acid", "este": "ester", "enzim": "enzyme", "ancol": "alcohol", "glucozơ": "glucose", "glucozo": "glucose",
         "hiđrocacbon": "hydrocarbon", "hidrocacbon": "hydrocarbon", "eten": "ethene", "polime": "polymer", "catot": "cathode",
         "anot": "anode", "gen": "gene", "timin": "thymine", "peptit": "peptide", "ribôxôm": "ribosome", "ribosom": "ribosome",
         "adn": "dna", "arn": "rna", "marn": "mrna", "ti": "ty", "oxy": "oxi", "hidro": "hydro", "hiđro": "hydro",
         "bazơ": "base", "baze": "base", "photpholipit": "phospholipid",
         # added 27 Oct after seeing Qwen3-8B closed-book predictions (transliterations of terms used in the keys):
         "prôtôn": "proton", "prôton": "proton", "nơtron": "neutron", "nơtrôn": "neutron", "êlectron": "electron", "êlectrôn": "electron"}
# reverse table used to *list* obvious accepted variants of a key (token-level; first listed is the canonical key spelling)
_VARIANT_TOKENS = {"acid": ["axit"], "axit": ["acid"], "ester": ["este"], "enzyme": ["enzim"], "alcohol": ["ancol"],
                   "glucose": ["glucozơ"], "hydrocarbon": ["hiđrocacbon"], "ethene": ["eten"], "polymer": ["polime"],
                   "cathode": ["catot"], "anode": ["anot"], "gene": ["gen"], "thymine": ["timin"], "peptide": ["peptit"],
                   "ribosome": ["ribôxôm"], "dna": ["adn"], "adn": ["dna"], "rna": ["arn"], "arn": ["rna"], "mrna": ["mARN"],
                   "ty": ["ti"]}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def norm_answer(s: str) -> str:
    """NFC, lower-case, tone-placement/ky-ly variants unified, punctuation stripped, token-level canonical forms."""
    t = nfc(str(s or "")).lower()
    for a, b in _TONE_PLACE.items():
        t = t.replace(a, b)
    for a, b in _Y2I.items():
        t = t.replace(a, b)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE).replace("_", " ")
    toks = [CANON.get(x, x) for x in t.split()]
    return " ".join(toks)


_LEAD = re.compile(r"^(đáp án|trả lời|answer)\s*(là)?\s*[:\-]?\s*", re.I)


def clean_prediction(p: str) -> str:
    p = (p or "").strip()
    p = _LEAD.sub("", p)
    return p.strip().strip('"“”«»\'`').strip()


def accepted_variants(gold: str) -> list[str]:
    """Obvious spelling/transliteration variants of a key (acid/axit, DNA/ADN, ...). The gold itself is not included."""
    out: list[str] = []
    toks = gold.split()
    for i, tk in enumerate(toks):
        for v in _VARIANT_TOKENS.get(tk.lower(), []):
            cand = " ".join(toks[:i] + [v] + toks[i + 1:])
            if cand not in out and cand != gold:
                out.append(cand)
    return out


def string_match(pred: str, gold: str, variants: list[str] | None = None) -> bool:
    """Strict: normalised prediction equals the normalised key or one of its accepted variants."""
    p = norm_answer(clean_prediction(pred))
    if not p:
        return False
    return p in {norm_answer(gold)} | {norm_answer(v) for v in (variants or [])}


def string_contains(pred: str, gold: str, variants: list[str] | None = None) -> bool:
    """Lenient (supplementary): the normalised key occurs as a whole-token run inside the normalised prediction."""
    p = " " + norm_answer(clean_prediction(pred)) + " "
    for g in {norm_answer(gold)} | {norm_answer(v) for v in (variants or [])}:
        if g and f" {g} " in p:
            return True
    return False


# ---------------------------------------------------------------- fill-in-the-blank wrapper (identical for every system)
FILL_INSTRUCTION = "Điền vào chỗ trống (_____) bằng một từ hoặc cụm từ ngắn; chỉ trả về phần cần điền. "


def fill_question(q: str) -> str:
    return FILL_INSTRUCTION + q


# ---------------------------------------------------------------- Qwen3-32B judge prompt (fixed BEFORE any result was looked at)
JUDGE_PROMPT_VERSION = "typed_fill_judge_v1"
JUDGE_SYSTEM = ("Bạn là giám khảo chấm câu điền khuyết môn học phổ thông Việt Nam. Bạn chỉ so sánh câu trả lời của thí sinh với "
                "đáp án chuẩn, không tự đánh giá lại kiến thức. Trả lời bằng đúng MỘT đối tượng JSON.")


def judge_user(question: str, gold: str, variants: list[str], prediction: str) -> str:
    var = ("; ".join(variants)) if variants else "(không có)"
    return (f"Câu hỏi (chỗ trống là _____): {question}\n"
            f"Đáp án chuẩn: {gold}\nCác cách viết khác được chấp nhận: {var}\n"
            f"Câu trả lời của thí sinh: {prediction or '(trống)'}\n\n"
            "Quy tắc chấm:\n"
            "- \"correct\" nếu câu trả lời cùng thuật ngữ / cùng giá trị với đáp án chuẩn: chấp nhận khác biệt viết hoa, dấu câu, "
            "dấu thanh, phiên âm (ví dụ axit = acid), và chấp nhận câu trả lời dài hơn nếu nó chứa đúng thuật ngữ chuẩn mà không kèm "
            "thuật ngữ khác mâu thuẫn.\n"
            "- \"incorrect\" nếu khác nghĩa, chỉ đúng một phần ý chính, nêu nhiều đáp án khác nhau, trống, hoặc từ chối trả lời.\n"
            'Trả về JSON: {"verdict": "correct|incorrect", "reason": "<tối đa 20 từ>"}')
