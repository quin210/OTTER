"""Builder step 3: schema-constrained LLM extraction of candidate knowledge per chunk.

* Output schema = `ExtractionOutput` (pydantic) -> JSON schema passed as `guided_json` to vLLM.
* Spans are shown as short local ids S1..Sn; every item must cite them. Mapping back to real
  span ids and all checks happen in normalize/validate (the extractor never sets support status).
* Content-hash cache (`<cache_dir>/<h[:2]>/<h>.json`) keyed on prompt version + model + chunk span
  ids/texts, so re-runs are incremental. Thread pool for high concurrency against vLLM.
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, ValidationError

from otter.builder.segment import Chunk
from otter.llm.interface import LLM, parse_json_loose
from otter.schemas.ontology import ConceptType, KnowledgeStatus, RelationType
from otter.storage.textutil import slugify

log = logging.getLogger("otter.builder.extract")
PROMPT_VERSION = "extract-v6"

PED_VI = {"objective": "mục tiêu", "definition": "định nghĩa", "concept_explanation": "giải thích khái niệm",
          "principle": "nguyên lí", "law": "định luật", "formula": "công thức", "condition": "điều kiện",
          "exception": "ngoại lệ", "process": "quá trình", "example": "ví dụ", "worked_example": "ví dụ giải",
          "summary": "tóm tắt", "figure_caption": "chú thích hình", "table": "bảng", "reading_text": "bài đọc"}

SUBJECT_VI = {"physics": "Vật lí", "chemistry": "Hoá học", "biology": "Sinh học", "history": "Lịch sử",
              "geography": "Địa lí", "literature": "Ngữ văn", "other": "Khác"}


# ----------------------------------------------------------------- LLM-facing schema
PRED_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"


class XAtom(BaseModel):
    predicate: str = Field(pattern=PRED_PATTERN,
                           description="snake_case tiếng Anh không dấu, mô tả mệnh đề KHẳNG ĐỊNH, vd. light_intensity_increases")
    args: list[str] = Field(default_factory=list, description="nhãn khái niệm (đúng như trong danh sách concepts) hoặc biến ?x")
    text: str = Field(description="mệnh đề tiếng Việt, bám sát câu chữ trong sách")
    negated: bool = False


class XVariable(BaseModel):
    symbol: str
    meaning: str
    unit: Optional[str] = None


class XConcept(BaseModel):
    label: str
    aliases: list[str] = Field(default_factory=list)
    concept_type: ConceptType = "Concept"
    definition: Optional[str] = None
    spans: list[str] = Field(min_length=1)


class XRelation(BaseModel):
    subject: str
    relation_type: RelationType
    object: str
    status: KnowledgeStatus = "explicit_fact"
    qualifier: Optional[str] = None
    spans: list[str] = Field(min_length=1)


class XFact(BaseModel):
    text: str
    concepts: list[str] = Field(default_factory=list)
    status: KnowledgeStatus = "explicit_fact"
    spans: list[str] = Field(min_length=1)


class XRule(BaseModel):
    rule_type: Literal["conditional", "definitional", "causal", "formula", "classification", "temporal"] = "conditional"
    premises: list[XAtom] = Field(min_length=1)
    guards: list[XAtom] = Field(default_factory=list)
    conclusions: list[XAtom] = Field(min_length=1)
    exceptions: list[XAtom] = Field(default_factory=list)
    formula: Optional[str] = None
    variables: list[XVariable] = Field(default_factory=list)
    status: KnowledgeStatus = "explicit_rule"
    spans: list[str] = Field(min_length=1)


class XObjective(BaseModel):
    text: str
    spans: list[str] = Field(min_length=1)


class ExtractionOutput(BaseModel):          # key order = generation order under guided decoding
    concepts: list[XConcept] = Field(default_factory=list)
    rules: list[XRule] = Field(default_factory=list)
    facts: list[XFact] = Field(default_factory=list)
    relations: list[XRelation] = Field(default_factory=list)
    objectives: list[XObjective] = Field(default_factory=list)


TOP_MAX_ITEMS = {"concepts": 25, "rules": 16, "facts": 30, "relations": 20, "objectives": 10}
NESTED_MAX_ITEMS = {"spans": 6, "aliases": 5, "args": 6, "concepts": 8, "premises": 6, "guards": 6,
                    "conclusions": 4, "exceptions": 4, "variables": 10}
STR_MAX = {"label": 120, "predicate": 64, "symbol": 24, "unit": 24, "formula": 200, "meaning": 160,
           "qualifier": 240, "definition": 500, "text": 500}


def output_schema() -> dict:
    """JSON schema for guided decoding: the pydantic schema plus size bounds (maxItems / maxLength) so
    the constrained decoder cannot loop until max_tokens. Parsing stays lenient (see parse_output)."""
    sch = ExtractionOutput.model_json_schema()

    def bound(props: dict, top: bool) -> None:
        for name, p in props.items():
            target = p
            if "anyOf" in p:                        # Optional[...] -> constrain the non-null branch
                target = next((x for x in p["anyOf"] if x.get("type") != "null"), p)
            if target.get("type") == "array":
                target["maxItems"] = (TOP_MAX_ITEMS if top else NESTED_MAX_ITEMS).get(name, 8)
                it = target.get("items", {})
                if it.get("type") == "string":
                    if name == "spans":
                        it["pattern"] = "^S[0-9]{1,3}$"
                    else:
                        it["maxLength"] = 60 if name == "aliases" else 120
            elif target.get("type") == "string" and "enum" not in target and "pattern" not in target:
                target["maxLength"] = STR_MAX.get(name, 300)
    bound(sch["properties"], True)
    for d in sch.get("$defs", {}).values():
        bound(d.get("properties", {}), False)
    return sch


def _close_stack(text: str) -> list[tuple[int, str]]:
    """Positions just after each '}' (outside strings) with the closers needed to finish the JSON there."""
    out, stack, in_str, esc = [], [], False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if ch == "}":
                out.append((i + 1, "".join(reversed(stack))))
    return out


def salvage_json(text: str) -> Any:
    """Recover the complete items of a truncated JSON object (e.g. output cut at max_tokens)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object to salvage")
    text = text[i:]
    for n, (pos, closers) in enumerate(reversed(_close_stack(text))):
        if n > 400:
            break
        try:
            obj = json.loads(text[:pos] + closers)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("could not salvage truncated JSON")


SYSTEM_PROMPT = """Bạn là chuyên gia xây dựng ontology tri thức từ sách giáo khoa Việt Nam. Nhiệm vụ: đọc các đoạn văn (mỗi đoạn có mã [S1], [S2], ...) và trích xuất tri thức có cấu trúc dưới dạng JSON đúng lược đồ.

QUY TẮC BẮT BUỘC:
1. CHỈ trích xuất những gì được nêu trong các đoạn văn đã cho. TUYỆT ĐỐI không bịa, không thêm kiến thức bên ngoài, không suy đoán số liệu, tên riêng, mốc thời gian, công thức.
2. Mỗi mục PHẢI có "spans" liệt kê mã đoạn (S1, S2, ...) chứa bằng chứng trực tiếp; chỉ dùng mã có trong bài. Không có bằng chứng thì không trích xuất.
3. "status": "explicit_fact" (sách phát biểu trực tiếp một sự kiện/quan hệ); "explicit_rule" (sách phát biểu trực tiếp một quy tắc/nguyên lí/định luật/công thức dạng điều kiện -> kết luận); "derived_conclusion" (kết luận sách tự rút ra từ các ý trong đoạn); "hypothesized_relation" (chỉ được gợi ý, hạn chế dùng).
4. QUY TẮC (rules) — chỉ khi sách nêu rõ quan hệ điều kiện/nhân quả/định luật/công thức/phân loại:
   - premises: các tiền đề chính ("nếu/khi X thì ..."); conclusions: kết luận; exceptions: ngoại lệ sách nêu ("trừ...", "ngoại trừ...").
   - guards: MỌI điều kiện áp dụng/phạm vi mà sách nêu kèm ("trong một giới hạn nhất định", "khi các yếu tố khác không đổi", "ở điều kiện thường", "với vật chuyển động thẳng đều", "chưa đạt điểm bão hoà"...). Không bỏ sót điều kiện; không đưa điều kiện phạm vi vào premises.
   - Mỗi mệnh đề (atom): "predicate" là snake_case TIẾNG ANH không dấu mô tả mệnh đề ở dạng KHẲNG ĐỊNH (cùng một ý thì dùng cùng predicate); "negated": true nếu sách nói mệnh đề đó KHÔNG xảy ra (khi đó predicate vẫn ở dạng khẳng định, vd. "không tăng thêm" -> predicate "photosynthesis_rate_increases", negated true); "text": mệnh đề tiếng Việt bám sát câu chữ; "args": nhãn khái niệm liên quan (đúng như trong concepts) hoặc biến "?x".
   - rule_type: "conditional" (nếu... thì...), "causal" (nguyên nhân -> kết quả), "formula" (CHỈ khi có công thức toán học), "definitional" (điều kiện cần và đủ để là X), "classification" (thuộc loại nào), "temporal" (trình tự thời gian).
5. CÔNG THỨC: rule_type="formula", "formula" viết ASCII chính xác như sách (vd. "v = s / t", "Q = m * c * delta_t"); liệt kê MỌI biến trong "variables" (symbol, meaning, unit theo sách; sách không nêu đơn vị thì unit=null). Premises của công thức mô tả điều kiện áp dụng/đại lượng đã biết, conclusions mô tả đại lượng được tính.
6. KHÁI NIỆM (concepts): thuật ngữ chuyên môn của môn học (không lấy từ thông thường), nhãn tiếng Việt chuẩn như trong sách (giữ nguyên dấu, viết thường trừ tên riêng), concept_type phù hợp (Process quá trình, Quantity đại lượng, Unit đơn vị, Law định luật, Structure cấu trúc/bộ phận, Substance chất, Organism sinh vật, Phenomenon hiện tượng, Event sự kiện, Person nhân vật, Place địa danh, Period thời kì, Work tác phẩm, LiteraryDevice biện pháp/thể loại văn học, Entity thực thể khác, Concept khái niệm chung). "definition" chỉ khi sách định nghĩa/mô tả (chép sát câu chữ, không dài quá 2 câu); "aliases": tên gọi khác, kí hiệu, viết tắt sách nêu.
7. QUAN HỆ (relations): subject/object là nhãn khái niệm (có trong concepts); relation_type đúng nghĩa: is_a (là một loại), part_of (là bộ phận của), has_part, has_property, involves, affects, causes, caused_by, occurs_before, occurs_after, occurs_in (diễn ra ở/trong), located_in, measured_in (đơn vị đo), defined_as, example_of, contrasts_with, synonym_of, authored_by (tác phẩm -> tác giả), belongs_to_period, related_to (chỉ khi không loại nào phù hợp). Điều kiện/phạm vi ghi vào "qualifier".
8. SỰ KIỆN (facts): mệnh đề nguyên tử, đủ nghĩa khi đứng một mình (thay đại từ "nó", "đây" bằng tên cụ thể), trung thành với sách; đặc biệt với Lịch sử, Địa lí, Ngữ văn: sự kiện, mốc thời gian, địa danh, số liệu, tác giả, tác phẩm, đặc điểm nghệ thuật. "concepts": nhãn các khái niệm được nhắc tới.
9. MỤC TIÊU (objectives): chỉ khi đoạn văn nêu yêu cầu cần đạt ("Sau bài học này, em sẽ...", "Học xong bài này..."); mỗi yêu cầu một mục.
10. Bỏ qua câu hỏi, bài tập, hoạt động, lời dẫn, chú thích hình không chứa tri thức. Phần bị lỗi OCR nặng thì bỏ qua.
11. Ưu tiên CHẤT LƯỢNG hơn số lượng: chỉ lấy khái niệm trọng tâm của bài (thường 5–20), không tạo quan hệ "involves"/"related_to" chung chung chỉ vì hai từ cùng xuất hiện. Trích xuất đầy đủ nhưng không lặp lại: mỗi ý chỉ một lần ở dạng phù hợp nhất (định nghĩa -> concept.definition; quan hệ hai khái niệm -> relation; điều kiện -> kết luận -> rule; còn lại -> fact).

VÍ DỤ ĐỊNH DẠNG (chỉ minh hoạ, KHÔNG chép nội dung ví dụ vào kết quả):
[S1] Điện trở là đại lượng đặc trưng cho mức độ cản trở dòng điện của vật dẫn.
[S2] Ở nhiệt độ không đổi, cường độ dòng điện chạy qua một dây dẫn kim loại tỉ lệ thuận với hiệu điện thế đặt vào hai đầu dây: I = U / R, trong đó I là cường độ dòng điện (A), U là hiệu điện thế (V), R là điện trở (Ω).
-> {"objectives":[],"rules":[{"rule_type":"formula","premises":[{"predicate":"voltage_applied_to_conductor","args":["hiệu điện thế"],"text":"đặt hiệu điện thế U vào hai đầu dây dẫn kim loại có điện trở R","negated":false}],"guards":[{"predicate":"temperature_constant","args":[],"text":"nhiệt độ không đổi","negated":false}],"conclusions":[{"predicate":"current_equals_voltage_over_resistance","args":["cường độ dòng điện"],"text":"cường độ dòng điện I = U / R","negated":false}],"formula":"I = U / R","variables":[{"symbol":"I","meaning":"cường độ dòng điện","unit":"A"},{"symbol":"U","meaning":"hiệu điện thế","unit":"V"},{"symbol":"R","meaning":"điện trở","unit":"Ω"}],"status":"explicit_rule","spans":["S2"]}],"facts":[],"relations":[{"subject":"hiệu điện thế","relation_type":"affects","object":"cường độ dòng điện","qualifier":"ở nhiệt độ không đổi, với dây dẫn kim loại","spans":["S2"]}],"concepts":[{"label":"điện trở","aliases":["R"],"concept_type":"Quantity","definition":"đại lượng đặc trưng cho mức độ cản trở dòng điện của vật dẫn","spans":["S1"]},{"label":"cường độ dòng điện","aliases":["I"],"concept_type":"Quantity","spans":["S2"]},{"label":"hiệu điện thế","aliases":["U"],"concept_type":"Quantity","spans":["S2"]}]}

LƯỢC ĐỒ JSON (đúng tên trường, ĐÚNG THỨ TỰ KHOÁ sau; trường không có thì bỏ qua hoặc để null/[]):
{"objectives":[{"text":str,"spans":["S1",...]}],
 "rules":[{"rule_type":conditional|causal|formula|definitional|classification|temporal,"premises":[ATOM],"guards":[ATOM],"conclusions":[ATOM],"exceptions":[ATOM],"formula":str|null,"variables":[{"symbol":str,"meaning":str,"unit":str|null}],"status":explicit_rule|derived_conclusion|hypothesized_relation,"spans":[...]}],
 "facts":[{"text":str,"concepts":[str],"status":explicit_fact|derived_conclusion|hypothesized_relation,"spans":[...]}],
 "relations":[{"subject":str,"relation_type":str,"object":str,"status":explicit_fact|derived_conclusion|hypothesized_relation,"qualifier":str|null,"spans":[...]}],
 "concepts":[{"label":str,"aliases":[str],"concept_type":một trong Concept|Process|Quantity|Unit|Law|Event|Person|Place|Period|State|Entity|Organism|Structure|Substance|Phenomenon|Work|LiteraryDevice|Other,"definition":str|null,"spans":[...]}]}
ATOM = {"predicate":"snake_case_ascii","args":[str],"text":str,"negated":false}
GIỚI HẠN CỨNG cho mỗi lần trả lời: tối đa 10 objectives, 16 rules, 30 facts, 20 relations, 25 concepts (chọn những mục quan trọng nhất; KHÔNG liệt kê mọi danh từ). Viết xong concepts thì đóng JSON ngay.

Chỉ trả về MỘT đối tượng JSON gọn trên một dòng (không markdown, không giải thích) với các khoá theo thứ tự: objectives, rules, facts, relations, concepts."""


def render_chunk(chunk: Chunk) -> str:
    lines = [f"Môn: {SUBJECT_VI.get(chunk.subject, chunk.subject)}; Lớp: {chunk.grade if chunk.grade else 'không rõ'}; "
             f"Sách: {chunk.book_title}"]
    if chunk.heading_paths:
        lines.append("Mục: " + " | ".join(" > ".join(h) for h in chunk.heading_paths))
    lines.append("")
    lines.append("CÁC ĐOẠN VĂN:")
    for lid, s in zip(chunk.local_ids(), chunk.spans):
        tag = PED_VI.get(s.pedagogical_type)
        lines.append(f"[{lid}]" + (f" ({tag})" if tag else "") + f" {s.text.strip()}")
    lines.append("")
    lines.append("Hãy trích xuất tri thức theo đúng các quy tắc. Mọi mục phải trích dẫn mã đoạn trong trường spans.")
    return "\n".join(lines)


def chunk_hash(chunk: Chunk, model: str) -> str:
    h = hashlib.sha256()
    h.update(f"{PROMPT_VERSION}\x1f{model}\x1f{chunk.subject}\x1f{chunk.grade}\x1f{chunk.book_title}".encode())
    for s in chunk.spans:
        h.update(f"\x1e{s.span_id}\x1f{s.text}".encode("utf-8"))
    return h.hexdigest()


@dataclass
class ExtractionResult:
    chunk: Chunk
    output: ExtractionOutput
    item_errors: list[dict] = field(default_factory=list)   # per-item schema failures (rejected)
    status: str = "ok"                                       # ok | cached | parse_failed | llm_error
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


_CONCEPT_TYPES = set(get_args(ConceptType))
_REL_TYPES = set(get_args(RelationType))
_STATUSES = set(get_args(KnowledgeStatus))
_RULE_TYPES = {"conditional", "definitional", "causal", "formula", "classification", "temporal"}
_REL_ALIASES = {"is_a_type_of": "is_a", "isa": "is_a", "type_of": "is_a", "subclass_of": "is_a", "partof": "part_of",
                "component_of": "part_of", "has_component": "has_part", "property": "has_property",
                "cause": "causes", "leads_to": "causes", "results_in": "causes", "located_at": "located_in",
                "unit": "measured_in", "has_unit": "measured_in", "written_by": "authored_by",
                "author": "authored_by", "before": "occurs_before", "after": "occurs_after", "related": "related_to"}


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _coerce(key: str, raw: dict, notes: list[str]) -> dict:
    """Lenient repairs for unguided JSON: scalar->list, enum normalisation, predicate slugs, size caps."""
    raw = dict(raw)
    raw["spans"] = [str(x).strip().strip("[]") for x in _as_list(raw.get("spans")) if str(x).strip()]
    for f in ("aliases", "concepts", "args"):
        if f in raw:
            raw[f] = [str(x) for x in _as_list(raw[f]) if x is not None and str(x).strip()]
    if "status" in raw and raw["status"] not in _STATUSES:
        raw.pop("status")
        notes.append("status_defaulted")
    if key == "concepts" and raw.get("concept_type") not in _CONCEPT_TYPES:
        ct = str(raw.get("concept_type") or "")
        match = next((t for t in _CONCEPT_TYPES if t.lower() == ct.lower()), None)
        raw["concept_type"] = match or "Concept"
    if key == "relations":
        rt = slugify(str(raw.get("relation_type") or ""), 40)
        rt = _REL_ALIASES.get(rt, rt)
        if rt not in _REL_TYPES:            # keep the item, generic type, original verb kept as qualifier
            q = raw.get("qualifier")
            raw["qualifier"] = f"{raw.get('relation_type')}" + (f"; {q}" if q else "")
            rt = "related_to"
            notes.append("relation_type_generalised")
        raw["relation_type"] = rt
    if key == "rules":
        if raw.get("rule_type") not in _RULE_TYPES:
            raw["rule_type"] = "conditional"
        for f in ("premises", "guards", "conclusions", "exceptions"):
            atoms = []
            for a in _as_list(raw.get(f)):
                if isinstance(a, str):
                    a = {"text": a}
                if not isinstance(a, dict) or not str(a.get("text") or a.get("predicate") or "").strip():
                    continue
                a = dict(a)
                a.setdefault("text", str(a.get("predicate")))
                a["args"] = [str(x) for x in _as_list(a.get("args")) if x is not None]
                a["negated"] = bool(a.get("negated", False))
                if not re.match(PRED_PATTERN, str(a.get("predicate") or "")):
                    a["predicate"] = slugify(str(a.get("predicate") or a.get("text") or "p"), 63)
                    if not re.match(PRED_PATTERN, a["predicate"]):
                        a["predicate"] = "p_" + a["predicate"]
                atoms.append(a)
            raw[f] = atoms
        if not raw["premises"] and raw["guards"]:
            raw["premises"], raw["guards"] = raw["guards"], []
        vs = raw.get("variables")
        if isinstance(vs, dict):    # {"v": "vận tốc (m/s)"} -> [{"symbol","meaning"}]
            raw["variables"] = [{"symbol": k, "meaning": str(v)} for k, v in vs.items()]
        elif vs is not None:
            raw["variables"] = [v for v in _as_list(vs) if isinstance(v, dict) and v.get("symbol")]
    return raw


def parse_output(obj: Any) -> tuple[ExtractionOutput, list[dict]]:
    """Validate item by item so one malformed item does not drop the whole chunk."""
    if not isinstance(obj, dict):
        raise ValueError("extraction output is not a JSON object")
    errs: list[dict] = []
    kept: dict[str, list] = {}
    for key, model in (("concepts", XConcept), ("relations", XRelation), ("facts", XFact), ("rules", XRule),
                       ("objectives", XObjective)):
        kept[key] = []
        items = _as_list(obj.get(key))
        cap = 2 * TOP_MAX_ITEMS.get(key, 20)
        if len(items) > cap:
            errs.append({"kind": key[:-1], "reason": "over_cap_dropped", "item": None,
                         "detail": f"{len(items) - cap} items beyond {cap}"})
            items = items[:cap]
        for raw in items:
            if not isinstance(raw, dict):
                errs.append({"kind": key[:-1], "reason": "schema_invalid", "detail": "not an object", "item": raw})
                continue
            notes: list[str] = []
            try:
                fixed = _coerce(key, raw, notes)
                if key == "rules" and not fixed.get("premises") and fixed.get("conclusions") and fixed.get("spans"):
                    # premise-less "rule" (plain statement / formula) -> keep it as a fact, never as a rule
                    txt = "; ".join(a["text"] for a in fixed["conclusions"])
                    if fixed.get("formula"):
                        txt += f" ({fixed['formula']})"
                    kept["facts"].append(XFact(text=txt, spans=fixed["spans"],
                                               status="explicit_fact"))
                    errs.append({"kind": "rule", "reason": "converted_to_fact_no_premises", "item": None})
                    continue
                kept[key].append(model.model_validate(fixed))
            except ValidationError as e:
                errs.append({"kind": key[:-1], "reason": "schema_invalid", "detail": str(e.errors()[:2])[:300],
                             "item": raw})
    return ExtractionOutput(**kept), errs


class Extractor:
    def __init__(self, llm: LLM, cache_dir: Optional[Path] = None, *, max_tokens: int = 6144,
                 use_guided: bool = False, retries: int = 1, net_retries: int = 6):
        self.llm, self.cache_dir, self.max_tokens = llm, cache_dir, max_tokens
        self.net_retries = net_retries
        self.use_guided, self.retries = use_guided, retries
        self.schema = output_schema()
        self._lock = threading.Lock()

    def _cache_path(self, h: str) -> Optional[Path]:
        return self.cache_dir / h[:2] / f"{h}.json" if self.cache_dir else None

    def extract(self, chunk: Chunk) -> ExtractionResult:
        model = getattr(self.llm, "model", "") or "unknown"
        h = chunk_hash(chunk, model)
        cp = self._cache_path(h)
        if cp and cp.exists():
            try:
                d = json.loads(cp.read_text(encoding="utf-8"))
                out, errs = parse_output(d["parsed"])
                return ExtractionResult(chunk, out, errs, "cached", None, d.get("prompt_tokens", 0),
                                        d.get("completion_tokens", 0), d.get("latency_ms", 0.0), cached=True)
            except Exception:
                pass
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": render_chunk(chunk)}]
        last_err = None
        pt = ct = 0
        lat = 0.0
        net_failures = parse_failures = 0
        while parse_failures <= self.retries and net_failures <= self.net_retries:
            try:
                r = self.llm.chat(msgs, temperature=0.0 if parse_failures == 0 else 0.3, max_tokens=self.max_tokens,
                                  json_schema=self.schema if self.use_guided else None)
                pt += r.prompt_tokens
                ct += r.completion_tokens
                lat += r.latency_ms
                truncated = False
                try:
                    obj = parse_json_loose(r.text)
                except ValueError:
                    obj, truncated = salvage_json(r.text), True     # keep the complete items (ValueError -> retry)
                out, errs = parse_output(obj)
                if truncated:
                    errs.append({"kind": "chunk", "reason": "output_truncated_salvaged", "item": None,
                                 "detail": f"completion_tokens={r.completion_tokens}"})
                if cp:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cp.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"chunk_id": chunk.chunk_id, "model": model,
                                               "prompt_version": PROMPT_VERSION, "parsed": obj, "raw": r.text,
                                               "prompt_tokens": pt, "completion_tokens": ct, "latency_ms": lat,
                                               "span_ids": [s.span_id for s in chunk.spans]},
                                              ensure_ascii=False), encoding="utf-8")
                    tmp.replace(cp)
                return ExtractionResult(chunk, out, errs, "ok", None, pt, ct, lat)
            except ValueError as e:
                last_err = f"parse_failed: {e}"
                parse_failures += 1
            except Exception as e:  # network / server errors: wait and retry (server restarts, overload)
                last_err = f"llm_error: {type(e).__name__}: {e}"
                net_failures += 1
                if net_failures <= self.net_retries:
                    time.sleep(min(120, 10 * net_failures))
        status = "parse_failed" if last_err and last_err.startswith("parse_failed") else "llm_error"
        return ExtractionResult(chunk, ExtractionOutput(), [], status, last_err, pt, ct, lat)

    def run(self, chunks: list[Chunk], workers: int = 32, progress_every: int = 20) -> list[ExtractionResult]:
        results: list[Optional[ExtractionResult]] = [None] * len(chunks)
        t0 = time.time()
        done = toks = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(self.extract, c): i for i, c in enumerate(chunks)}
            for f in as_completed(futs):
                i = futs[f]
                results[i] = f.result()
                done += 1
                toks += results[i].completion_tokens if not results[i].cached else 0
                if done % progress_every == 0 or done == len(chunks):
                    dt = time.time() - t0
                    log.info("extract %d/%d chunks (%.1f chunks/min, %.0f gen tok/s, last=%s)", done, len(chunks),
                             60 * done / max(dt, 1e-6), toks / max(dt, 1e-6), results[i].status)
        return [r for r in results if r is not None]
