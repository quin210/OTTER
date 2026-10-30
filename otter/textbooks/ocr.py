"""VLM OCR of scanned Vietnamese textbook pages (Qwen3-VL via OpenAI-compatible vLLM servers).

Each page gets two independent reads of the printed page number:
  * ``tag``  - the full-page transcription must end with ``<<SỐ_TRANG: N>>`` (or ``không``);
  * ``crop`` - a separate request on the top+bottom margin strips only.
``otter.textbooks.pages`` keeps a label only when it was actually read from the page.

Degenerate generation (repetition loops) is detected, the looping tail is trimmed and the
page is retried once with a repetition penalty; such pages are flagged for manual review.
Nothing here invents text: on failure the page text stays empty and ``error`` is set.

(Rewritten 25 Oct 23:4x after the original WS1 file was overwritten by an external tool;
the foreign file is kept at docs/overwritten_2338_ocr.py.)
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

OCR_VERSION = "otter-vlm-ocr-0.4"

PAGE_PROMPT = """Bạn là hệ thống OCR cho sách giáo khoa tiếng Việt. Hãy chép lại TOÀN BỘ chữ trên trang ảnh này, đúng nguyên văn, đầy đủ dấu tiếng Việt, theo thứ tự đọc tự nhiên (cột trái trước cột phải, trên xuống dưới).
Quy tắc định dạng Markdown:
- Tiêu đề chương/chủ đề dùng "#", tiêu đề bài dùng "##", mục (I., II., ...) dùng "###", tiểu mục (1., 2., ...) dùng "####". Giữ nguyên số thứ tự in trên trang (ví dụ "Bài 7", "CHƯƠNG II", "I.", "1.").
- Công thức toán, vật lí viết bằng LaTeX trong $...$ hoặc $$...$$; phương trình hoá học viết dạng chữ (ví dụ 2H2 + O2 → 2H2O).
- Bảng viết thành bảng Markdown.
- Với hình ảnh, sơ đồ, bản đồ: chỉ ghi chú thích in trên trang dạng [Hình 1.2. <chú thích>]; không mô tả nội dung hình nếu không có chữ.
- Không thêm lời giải thích, không tóm tắt, không dịch, không tự bịa nội dung không có trên trang. Không chép tiêu đề chạy (header/footer lặp lại) và số trang vào phần nội dung.
Dòng cuối cùng của câu trả lời PHẢI là <<SỐ_TRANG: N>> với N là số trang in trên trang (chỉ chữ số hoặc số La Mã như in), hoặc <<SỐ_TRANG: không>> nếu trang không in số trang."""

CROP_PROMPT = ("Ảnh này gồm lề trên và lề dưới của một trang sách. Số trang được in ở đây là bao nhiêu? "
               "Chỉ trả lời đúng con số (hoặc số La Mã) như in trên ảnh; nếu không thấy số trang thì trả lời: không")

VI_DIACRITIC = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
_TAG_RE = re.compile(r"\s*<<\s*S[ỐO]_?\s*TRANG\s*:\s*([^>]*?)\s*>>\s*$", re.I)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_PAGE_NUM_RE = re.compile(r"^(?:\d{1,4}|[IVXLCDM]{1,8}|[ivxlcdm]{1,8})$")


@dataclass
class OCRResult:
    text: str = ""
    printed_page_label_tag: Optional[str] = None
    printed_page_label_crop: Optional[str] = None
    finish_reason: Optional[str] = None
    attempts: int = 0
    latency_s: float = 0.0
    completion_tokens: Optional[int] = None
    degenerate: bool = False
    trimmed_repetition: bool = False
    error: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- text helpers

def normalize_label(raw: Optional[str]) -> Optional[str]:
    """Return a printed page label if ``raw`` looks like one (digits / roman numerals), else None."""
    if raw is None:
        return None
    s = unicodedata.normalize("NFC", raw).strip().strip(".:-–—()[]* \"'")
    s = re.sub(r"^(?:trang|page|tr\.)\s*", "", s, flags=re.I).strip()
    if not s or s.lower() in {"không", "khong", "none", "null", "n/a", "-"}:
        return None
    return s if _PAGE_NUM_RE.match(s) else None


def split_page_tag(text: str) -> tuple[str, Optional[str], bool]:
    """Strip the trailing ``<<SỐ_TRANG: N>>`` tag -> (body, label or None, tag_present)."""
    m = _TAG_RE.search(text or "")
    if not m:
        return (text or "").strip(), None, False
    return text[:m.start()].strip(), normalize_label(m.group(1)), True


def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def repetition_score(text: str) -> float:
    """0..1; high when the text is dominated by repeated lines or a repeated character n-gram loop."""
    lines = [l for l in _lines(text) if len(l) >= 4 and not re.fullmatch(r"[|\-:\s]+", l)]
    line_dup = 0.0
    if len(lines) >= 6:
        seen: set[str] = set(); dups = 0
        for l in lines:
            dups += l in seen; seen.add(l)
        line_dup = dups / len(lines)
    tail = re.sub(r"\s+", " ", text[-1500:])
    chunk_dup = 0.0
    if len(tail) >= 300:
        grams = [tail[i:i + 20] for i in range(0, len(tail) - 20, 10)]
        chunk_dup = 1 - len(set(grams)) / len(grams)
    return round(max(line_dup, chunk_dup), 4)


def trim_degenerate_tail(text: str, min_repeats: int = 4) -> tuple[str, bool]:
    """Cut a looping tail (a block of 1..12 lines, or a char span, repeated >= min_repeats times
    consecutively at the end of the text), keeping one copy. Returns (text, changed)."""
    lines = text.rstrip().split("\n")
    for k in range(1, 13):
        if len(lines) < k * min_repeats:
            break
        block = [l.strip() for l in lines[-k:]]
        if not any(block):
            continue
        reps, i = 1, len(lines) - k
        while i - k >= 0 and [l.strip() for l in lines[i - k:i]] == block:
            reps += 1; i -= k
        if reps >= min_repeats:
            return "\n".join(lines[:i + k]).rstrip(), True
    m = re.search(r"(.{3,200}?)(?:\s*\1){%d,}\s*$" % (min_repeats - 1), text, re.S)
    if m and len(m.group(0)) > 60:
        return (text[:m.start()] + m.group(1)).rstrip(), True
    return text, False


def quality_metrics(text: str) -> dict[str, Any]:
    t = unicodedata.normalize("NFC", text or "")
    words = _WORD_RE.findall(t)
    alpha_words = [w for w in words if len(w) >= 2]
    vi = sum(1 for w in alpha_words if any(c in VI_DIACRITIC for c in w.lower()))
    n = len(t)
    non_space = [c for c in t if not c.isspace()]
    garbage = sum(1 for c in non_space if c in "�?" or unicodedata.category(c) in ("Co", "Cn", "Cc"))
    latex = len(re.findall(r"\$[^$]+\$", t))
    return {
        "n_chars": n,
        "n_words": len(words),
        "vi_diacritic_ratio": round(vi / len(alpha_words), 4) if alpha_words else 0.0,
        "garbage_ratio": round(garbage / len(non_space), 4) if non_space else 0.0,
        "repetition": repetition_score(t),
        "n_latex": latex,
        "n_table_rows": sum(1 for l in t.splitlines() if l.strip().startswith("|")),
        "n_headings": sum(1 for l in t.splitlines() if l.lstrip().startswith("#")),
    }


def quality_score(m: dict[str, Any], finish_reason: Optional[str] = None,
                  degenerate: bool = False) -> tuple[float, bool, list[str]]:
    """Heuristic 0..1 page quality + manual-review flag + reasons."""
    score, reasons = 1.0, []
    if m["n_words"] >= 25:
        r = m["vi_diacritic_ratio"]
        if r < 0.15:
            score -= 0.45; reasons.append(f"very_low_diacritics:{r}")
        elif r < 0.3 and m["n_latex"] < 3:
            score -= 0.2; reasons.append(f"low_diacritics:{r}")
    if m["garbage_ratio"] > 0.02:
        score -= min(0.5, 5 * m["garbage_ratio"]); reasons.append(f"garbage:{m['garbage_ratio']}")
    if m["repetition"] > 0.5:
        score -= 0.4; reasons.append(f"repetition:{m['repetition']}")
    elif m["repetition"] > 0.3:
        score -= 0.15; reasons.append(f"repetition:{m['repetition']}")
    if finish_reason == "length":
        score -= 0.3; reasons.append("truncated_max_tokens")
    if degenerate:
        score -= 0.3; reasons.append("degenerate_generation")
    score = round(max(0.0, min(1.0, score)), 4)
    review = score < 0.7 or degenerate or finish_reason == "length"
    return score, review, reasons


# ----------------------------------------------------------------------------- adapters

def _data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode("ascii")


MAX_IMAGE_PIXELS = 2_400_000   # ~1.3k x 1.8k; larger rasters overflow the 16k-token context (text stays legible)


def _page_image_bytes(path: Path) -> tuple[bytes, str]:
    """Original file bytes, or a downscaled PNG when the raster is very large."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if w * h <= MAX_IMAGE_PIXELS:
            return path.read_bytes(), ("image/png" if path.suffix.lower() == ".png" else "image/jpeg")
        f = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        small = im.convert("RGB").resize((int(w * f), int(h * f)), Image.LANCZOS)
        buf = io.BytesIO(); small.save(buf, format="PNG")
        return buf.getvalue(), "image/png"


def margin_strip_png(path: Path, frac: float = 0.09) -> bytes:
    """Top and bottom margin strips of the page stacked into one PNG (for the page-number read)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        sh = max(40, int(h * frac))
        top, bot = im.crop((0, 0, w, sh)), im.crop((0, h - sh, w, h))
        out = Image.new("RGB", (w, 2 * sh + 12), "white")
        out.paste(top, (0, 0)); out.paste(bot, (0, sh + 12))
        buf = io.BytesIO(); out.save(buf, format="PNG")
        return buf.getvalue()


class CallableOCRAdapter:
    """Test adapter: ``fn(path) -> str`` (full-page markdown incl. optional page tag)."""

    def __init__(self, fn: Callable[[Path], str], model: str = "callable"):
        self.fn, self.model = fn, model

    async def ocr_page(self, path: Path) -> OCRResult:
        t0 = time.time()
        try:
            raw = self.fn(Path(path))
        except Exception as e:  # pragma: no cover - surfaced in record
            return OCRResult(error=f"{type(e).__name__}: {e}", attempts=1, model=self.model)
        body, label, present = split_page_tag(raw)
        body2, trimmed = trim_degenerate_tail(body)
        return OCRResult(text=body2, printed_page_label_tag=label, finish_reason="stop", attempts=1,
                         latency_s=round(time.time() - t0, 3), trimmed_repetition=trimmed, model=self.model,
                         notes=[] if present else ["page_tag_missing"])


class VisionLLMOCRAdapter:
    """Async OCR over one or more OpenAI-compatible vLLM endpoints (least-loaded dispatch)."""

    def __init__(self, base_urls: list[str] | str, model: str = "Qwen/Qwen3-VL-8B-Instruct",
                 per_endpoint_concurrency: int = 48, max_tokens: int = 6144, timeout_s: float = 600.0,
                 crop_read: bool = True, max_attempts: int = 3, first_attempt_max_tokens: int = 2500):
        from openai import AsyncOpenAI
        if isinstance(base_urls, str):
            base_urls = [u for u in base_urls.split(",") if u]
        self.base_urls, self.model, self.max_tokens = list(base_urls), model, max_tokens
        self.clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=timeout_s, max_retries=0) for u in self.base_urls]
        self.sems = [asyncio.Semaphore(per_endpoint_concurrency) for _ in self.base_urls]
        # short margin reads get their own slots so they never queue behind full-page reads
        self.crop_sems = [asyncio.Semaphore(max(4, per_endpoint_concurrency // 3)) for _ in self.base_urls]
        self.load = [0] * len(self.base_urls)
        self.crop_read, self.max_attempts = crop_read, max_attempts
        self.first_attempt_max_tokens = first_attempt_max_tokens

    async def _chat(self, img_url: str, prompt: str, max_tokens: int, extra: dict | None = None, crop: bool = False):
        i = min(range(len(self.clients)), key=lambda k: self.load[k])
        self.load[i] += 1
        try:
            async with (self.crop_sems if crop else self.sems)[i]:
                resp = await self.clients[i].chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=max_tokens, seed=0,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": prompt}]}],
                    extra_body=extra or {})
            return resp, self.base_urls[i]
        finally:
            self.load[i] -= 1

    async def _read_label_crop(self, path: Path) -> tuple[Optional[str], Optional[str]]:
        try:
            png = await asyncio.to_thread(margin_strip_png, path)
            resp, _ = await self._chat(_data_url(png), CROP_PROMPT, 16, crop=True)
            return normalize_label(resp.choices[0].message.content or ""), None
        except Exception as e:
            return None, f"crop_read_error:{type(e).__name__}"

    async def ocr_page(self, path: Path) -> OCRResult:
        path = Path(path)
        res = OCRResult(model=self.model)
        t0 = time.time()
        img_url = _data_url(*(await asyncio.to_thread(_page_image_bytes, path)))
        crop_task = asyncio.create_task(self._read_label_crop(path)) if self.crop_read else None
        extra: dict[str, Any] = {}
        last_err = None
        best: Optional[tuple[str, Optional[str], bool, str, Optional[int]]] = None
        for attempt in range(1, self.max_attempts + 1):
            res.attempts = attempt
            # legit pages need <= ~1.3k tokens (p99 1068); a first attempt that hits the cap is a repetition loop,
            # so keep it short and only give retries (with a repetition penalty) the full budget
            budget = min(self.max_tokens, self.first_attempt_max_tokens) if attempt == 1 else self.max_tokens
            try:
                resp, ep = await self._chat(img_url, PAGE_PROMPT, budget, extra)
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:300]}"
                res.notes.append(f"attempt{attempt}_error:{type(e).__name__}:{str(e)[:120]}")
                await asyncio.sleep(min(30, 5 * attempt))
                continue
            choice = resp.choices[0]
            raw = choice.message.content or ""
            res.endpoint = ep
            fin = choice.finish_reason
            toks = getattr(resp.usage, "completion_tokens", None) if resp.usage else None
            body, label, present = split_page_tag(raw)
            if not present:
                res.notes.append(f"attempt{attempt}_page_tag_missing")
            degenerate = fin == "length" or repetition_score(body) > 0.5
            best = (body, label, degenerate, fin, toks)
            if not degenerate:
                break
            res.notes.append(f"attempt{attempt}_degenerate(finish={fin})")
            extra = {"repetition_penalty": 1.1 if attempt == 1 else 1.2}
        if best is None:
            res.error = last_err or "no response"
        else:
            body, label, degenerate, fin, toks = best
            if degenerate:
                body, trimmed = trim_degenerate_tail(body)
                res.trimmed_repetition = trimmed
            res.text, res.printed_page_label_tag, res.degenerate = body, label, degenerate
            res.finish_reason, res.completion_tokens = fin, toks
        if crop_task is not None:
            res.printed_page_label_crop, note = await crop_task
            if note:
                res.notes.append(note)
        res.latency_s = round(time.time() - t0, 2)
        return res
