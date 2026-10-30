"""All LLM prompts used by OTTER agents, tools and internal baselines.

Content-facing prompts are in Vietnamese; JSON keys / tool names stay in English.
Every prompt asks for exactly ONE JSON object so that it works without native tool calling.
"""
from __future__ import annotations

import json
from typing import Any, Optional

PROMPT_VERSION = "otter-prompts-v2"

SUBJECT_VI = {"physics": "Vật lí", "chemistry": "Hóa học", "biology": "Sinh học", "history": "Lịch sử",
              "geography": "Địa lí", "literature": "Ngữ văn", "other": "khác"}


def fmt_choices(choices: Optional[dict[str, str]]) -> str:
    if not choices:
        return ""
    return "\n".join(f"{k}. {v}" for k, v in choices.items())


def _scope(subject: Optional[str], grade: Optional[int]) -> str:
    s = SUBJECT_VI.get(subject or "", subject or "chưa xác định")
    return f"Môn: {s}" + (f"; lớp {grade}" if grade else "")


# --------------------------------------------------------------------------- direct baseline
DIRECT_SYSTEM = ("Bạn là trợ lý trả lời câu hỏi trắc nghiệm theo chương trình sách giáo khoa phổ thông Việt Nam. "
                 "Trả lời bằng đúng MỘT đối tượng JSON, không thêm chữ nào khác.")


def direct_user(question: str, choices: Optional[dict[str, str]], subject: Optional[str], grade: Optional[int]) -> str:
    if choices:
        return (f"{_scope(subject, grade)}\nCâu hỏi: {question}\nCác lựa chọn:\n{fmt_choices(choices)}\n\n"
                'Trả về JSON: {"answer": "<một chữ cái trong ' + ", ".join(choices) + '>", '
                '"explanation": "<giải thích ngắn bằng tiếng Việt>"}')
    return (f"{_scope(subject, grade)}\nCâu hỏi: {question}\n\n"
            'Trả về JSON: {"answer": "<câu trả lời ngắn gọn bằng tiếng Việt>"}')


# --------------------------------------------------------------------------- text RAG baseline
def rag_user(question: str, choices: Optional[dict[str, str]], subject: Optional[str], grade: Optional[int],
             spans: list[dict[str, Any]]) -> str:
    ev = "\n\n".join(f"[{s['span_id']}] {s['text']}" for s in spans) or "(không tìm thấy đoạn văn nào)"
    head = (f"{_scope(subject, grade)}\nCác đoạn trích từ sách giáo khoa (mã trong ngoặc vuông):\n{ev}\n\n"
            f"Câu hỏi: {question}\n")
    if choices:
        return (head + f"Các lựa chọn:\n{fmt_choices(choices)}\n\n"
                "Dựa vào các đoạn trích (nếu chúng liên quan) để chọn đáp án. Chỉ trích dẫn mã đoạn có trong danh "
                "sách trên.\n"
                'Trả về JSON: {"answer": "<chữ cái>", "cited_span_ids": ["<mã đoạn>", ...], '
                '"explanation": "<giải thích ngắn>"}')
    return (head + "Trả lời dựa trên các đoạn trích; nếu không đủ thông tin hãy nói rõ.\n"
            'Trả về JSON: {"answer": "<câu trả lời>", "cited_span_ids": ["<mã đoạn>", ...]}')


# --------------------------------------------------------------------------- master planner
PLANNER_SYSTEM = (
    "Bạn là Tác tử Chủ (Master) của hệ thống OTTER. Nhiệm vụ: phân tích câu hỏi và lập kế hoạch các mục tiêu con "
    "(subgoal) dạng đồ thị phụ thuộc không chu trình (DAG). Trả lời bằng đúng MỘT đối tượng JSON.")


def planner_user(question: str, choices: Optional[dict[str, str]], subject: Optional[str], grade: Optional[int],
                 max_subgoals: int) -> str:
    return (
        f"{_scope(subject, grade)}\nCâu hỏi: {question}\n"
        + (f"Các lựa chọn:\n{fmt_choices(choices)}\n" if choices else "")
        + "\nQuy tắc lập kế hoạch:\n"
        "- Câu hỏi đơn giản (nhận biết khái niệm, định nghĩa, một sự kiện): CHỈ 1 subgoal.\n"
        f"- Câu hỏi nhiều bước (suy luận nhân quả, điều kiện, tính toán, so sánh): 2 đến {max_subgoals} subgoal, "
        "mỗi subgoal là một câu hỏi con cụ thể cần tra cứu trong sách giáo khoa.\n"
        "- \"requires\": danh sách id subgoal phải xong trước (quan hệ VÀ). \"requires_any\": danh sách các nhóm id, "
        "chỉ cần một id trong nhóm (quan hệ HOẶC). Không được tạo chu trình.\n"
        "- Không cần thêm bước đánh giá các lựa chọn; hệ thống tự thêm bước đó.\n"
        "- type thuộc: fact_lookup, definition, condition_check, rule_application, calculation, comparison, "
        "causal_explanation, other.\n"
        "- hints: vài thuật ngữ khóa tiếng Việt để tra cứu.\n"
        'Trả về JSON: {"subject": "physics|chemistry|biology|history|geography|literature|other", '
        '"question_type": "direct|multi_hop|conditional|calculation|choice_check", '
        '"subgoals": [{"id": "sg1", "question": "...", "requires": [], "requires_any": [], "type": "fact_lookup", '
        '"hints": ["..."]}], "rationale": "<một câu>"}')


# --------------------------------------------------------------------------- homogeneous subagent
TOOL_SPECS = {
    "search_knowledge": ('Tìm khái niệm/sự kiện (fact)/quy tắc (rule) trong ontology.',
                         '{"query": "<từ khóa tiếng Việt>", "top_k": 8, "kinds": ["concept","fact","rule"]}'),
    "expand_graph": ('Mở rộng lân cận đồ thị từ các id (quan hệ có kiểu, quy tắc liên quan và điều kiện).',
                     '{"node_ids": ["<id>"], "relation_types": null, "max_hops": 1}'),
    "retrieve_source": ('Lấy nguyên văn đoạn sách giáo khoa theo mã đoạn, hoặc tìm theo truy vấn văn bản.',
                        '{"span_ids": ["<span_id>"]} hoặc {"query": "<truy vấn>", "top_k": 5}'),
    "execute_rule": ('Áp dụng chặt một quy tắc: chỉ "entailed" khi MỌI tiền đề và điều kiện được xác lập; '
                     '"blocked" khi điều kiện không thỏa; "undetermined" khi thiếu. Hoặc chứng minh một goal.',
                     '{"rule_id": "<id>", "facts": [{"predicate": "<vị từ>", "args": [], "value": true, '
                     '"evidence_ids": ["<span/fact id>"]}]} hoặc {"goal": "<vị từ>(...)", "facts": [...]}'),
    "calculate": ('Máy tính an toàn có đơn vị. Có thể dùng công thức của quy tắc (rule_id) hoặc biểu thức.',
                  '{"expression": "F/m", "variables": {"F": "6 N", "m": "2 kg"}, "target_unit": "m/s^2", '
                  '"compare_with": {"A": "4 m/s^2"}} hoặc {"rule_id": "<id>", "variables": {...}, "target": "<biến>"}'),
    "verify_evidence": ('Kiểm tra một khẳng định với các id bằng chứng (cấu trúc, liên kết nguồn, mức hỗ trợ ngữ nghĩa).',
                        '{"claim": "<khẳng định>", "assertion_ids": [], "rule_ids": [], "span_ids": []}'),
}


def subagent_system(enabled_tools: list[str], max_calls: int) -> str:
    tools = "\n".join(f"- {t}: {TOOL_SPECS[t][0]}\n  arguments: {TOOL_SPECS[t][1]}" for t in enabled_tools)
    return (
        "Bạn là một Tác tử con (Subagent) của hệ thống OTTER, trả lời câu hỏi dựa trên sách giáo khoa phổ thông Việt "
        "Nam. Mọi tác tử con đều giống nhau, chỉ khác mục tiêu con được giao.\n"
        f"Công cụ được phép dùng (tối đa {max_calls} lần gọi):\n{tools}\n\n"
        "Cách làm việc: tra cứu ontology (search_knowledge/expand_graph) khi cần khái niệm, quan hệ, quy tắc; dùng "
        "retrieve_source để đọc nguyên văn đoạn sách làm bằng chứng; chỉ dùng calculate khi cần tính toán; dùng "
        "execute_rule khi một quy tắc có điều kiện liên quan.\n"
        "Quy tắc bắt buộc:\n"
        "- KHÔNG bịa mã (id). Chỉ dùng id đã xuất hiện trong kết quả công cụ. span_ids = mã đoạn sách (span_id); "
        "assertion_ids = chỉ id của fact (dạng <môn>:a:...); rule_ids = chỉ id của rule (dạng <môn>:rule:...).\n"
        "- Khi finish, hệ thống sẽ TỰ ĐỘNG kiểm chứng kết quả cuối bằng verify_evidence; không cần tự gọi "
        "verify_evidence cho kết quả cuối. Nếu các đoạn đã có đủ thông tin thì finish ngay, không tra cứu thừa.\n"
        "- Thiếu bằng chứng thì nói là chưa xác định (undetermined); không suy ra phủ định khi điều kiện không thỏa.\n"
        "- Mỗi lượt trả lời đúng MỘT đối tượng JSON:\n"
        '  {"thought": "<tối đa 15 từ>", "action": "<tên công cụ>", "arguments": {...}}\n'
        "  hoặc khi đã đủ bằng chứng / hết lượt:\n"
        '  {"thought": "...", "action": "finish", "arguments": <kết quả như mô tả trong yêu cầu>}')


FINISH_SPEC_CLAIM = (
    '{"claim": "<khẳng định trả lời mục tiêu con, tiếng Việt>", "canonical_predicate": null, '
    '"assertion_ids": ["<fact id đã dùng>"], "rule_ids": ["<rule id đã dùng>"], "span_ids": ["<span id làm bằng chứng>"], '
    '"proposed_status": "supported|contradicted|undetermined", "missing_premises": ["<điều còn thiếu để chứng minh>"]}')

FINISH_SPEC_CHOICES = (
    '{"claim": "<kết luận ngắn>", "assertion_ids": [], "rule_ids": [], "span_ids": ["<span id>"], '
    '"choices": {"A": {"verdict": "entailed|contradicted|undetermined", "span_ids": ["<span id>"]}, ...}, '
    '"best_choice": "<chữ cái hợp lý nhất>", "missing_premises": ["<bằng chứng còn thiếu>"]}\n'
    '(verdict "entailed" = bằng chứng cho thấy lựa chọn này là đáp án đúng của câu hỏi; "contradicted" = bằng chứng '
    'cho thấy nó không phải đáp án đúng; "undetermined" = chưa đủ bằng chứng.)')


def subagent_user(subgoal_question: str, claim_type: str, question: str, choices: Optional[dict[str, str]],
                  subject: Optional[str], grade: Optional[int], dependencies: list[dict[str, Any]],
                  hints: list[str], is_choice_node: bool, notes: list[str]) -> str:
    deps = ""
    if dependencies:
        lines = []
        for d in dependencies:
            ev = "; ".join(f"[{s}] {t}" for s, t in d.get("evidence", [])[:3])
            lines.append(f"- ({d['packet_id']}, {d['status']}) {d['claim']}" + (f"\n  Bằng chứng: {ev}" if ev else ""))
        deps = "Kết quả đã kiểm chứng của các mục tiêu con phụ thuộc:\n" + "\n".join(lines) + "\n"
    extra = ("Ghi chú: " + " ".join(notes) + "\n") if notes else ""
    spec = FINISH_SPEC_CHOICES if is_choice_node else FINISH_SPEC_CLAIM
    return (
        f"{_scope(subject, grade)}\nCâu hỏi gốc: {question}\n"
        + (f"Các lựa chọn:\n{fmt_choices(choices)}\n" if choices else "")
        + f"Mục tiêu con được giao ({claim_type}): {subgoal_question}\n"
        + (f"Gợi ý tra cứu: {', '.join(hints)}\n" if hints else "")
        + deps + extra
        + f"\nKhi kết thúc, dùng action \"finish\" với arguments dạng:\n{spec}\nHãy bắt đầu.")


def observation_msg(call_id: str, tool: str, result_json: str, remaining: int) -> str:
    return (f"Kết quả {tool} ({call_id}):\n{result_json}\n"
            f"Còn {remaining} lượt gọi công cụ. Trả lời bằng MỘT đối tượng JSON (gọi công cụ tiếp hoặc finish).")


FORCE_FINISH = ("Đã hết lượt gọi công cụ. Hãy trả lời ngay bằng action \"finish\" dựa trên các bằng chứng đã có; "
                "phần nào thiếu bằng chứng thì đánh dấu undetermined.")

INVALID_ACTION = ("Phản hồi trước không phải JSON hợp lệ hoặc công cụ không tồn tại. Trả lời lại bằng đúng MỘT "
                  'đối tượng JSON {"thought": ..., "action": ..., "arguments": {...}}.')


# --------------------------------------------------------------------------- evidence judge (verify_evidence)
JUDGE_SYSTEM = ("Bạn là bộ kiểm tra bằng chứng. Chỉ dựa vào các đoạn văn được cung cấp, không dùng hiểu biết bên "
                "ngoài. Trả lời bằng đúng MỘT đối tượng JSON.")


def judge_user(claims: list[dict[str, str]], spans: list[dict[str, str]], question: Optional[str]) -> str:
    ev = "\n\n".join(f"[{s['span_id']}] {s['text']}" for s in spans)
    if question:
        cl = "\n".join(f"{c['id']}. {c['text']}" for c in claims)
        task = (f"Câu hỏi: {question}\nCác lựa chọn:\n{cl}\n\n"
                "Với MỖI lựa chọn, xác định các đoạn văn có cho thấy lựa chọn đó là đáp án đúng của câu hỏi không:\n"
                '- "supported": có đoạn văn cho thấy lựa chọn là đáp án đúng;\n'
                '- "contradicted": có đoạn văn cho thấy lựa chọn KHÔNG phải đáp án đúng;\n'
                '- "not_enough_info": các đoạn văn không đủ để kết luận.\n')
    else:
        cl = "\n".join(f"{c['id']}. {c['text']}" for c in claims)
        task = (f"Các khẳng định:\n{cl}\n\nVới MỖI khẳng định, xác định các đoạn văn ủng hộ (supported), mâu thuẫn "
                "(contradicted) hay không đủ thông tin (not_enough_info).\n")
    return (f"Các đoạn văn:\n{ev}\n\n{task}"
            "Với supported/contradicted phải trích NGUYÊN VĂN một câu ngắn (quote) từ đúng đoạn văn (span_id). Nếu "
            "kết luận cần ghép nhiều đoạn (suy luận nhiều bước), thêm các trích dẫn còn lại vào \"more\" (tối đa 2).\n"
            'Trả về JSON: {"verdicts": [{"id": "<id>", "verdict": "supported|contradicted|not_enough_info", '
            '"span_id": "<mã đoạn hoặc null>", "quote": "<trích nguyên văn hoặc null>", '
            '"more": [{"span_id": "<mã đoạn>", "quote": "<trích nguyên văn>"}]}]}')


# --------------------------------------------------------------------------- flat aggregation (multi_agent_no_workspace)
def flat_aggregate_user(question: str, choices: Optional[dict[str, str]], subject: Optional[str], grade: Optional[int],
                        findings: list[str]) -> str:
    f = "\n".join(f"- {x}" for x in findings) or "- (không có kết quả)"
    if choices:
        return (f"{_scope(subject, grade)}\nCâu hỏi: {question}\nCác lựa chọn:\n{fmt_choices(choices)}\n\n"
                f"Kết quả của các tác tử con:\n{f}\n\nTổng hợp và chọn đáp án.\n"
                'Trả về JSON: {"answer": "<chữ cái>", "explanation": "<giải thích ngắn>"}')
    return (f"{_scope(subject, grade)}\nCâu hỏi: {question}\n\nKết quả của các tác tử con:\n{f}\n\n"
            'Tổng hợp câu trả lời. Trả về JSON: {"answer": "<câu trả lời tiếng Việt>"}')


# --------------------------------------------------------------------------- composer verbalization
VERBALIZE_SYSTEM = ("Bạn diễn đạt câu trả lời tiếng Việt CHỈ từ các khẳng định đã được kiểm chứng được cung cấp. "
                    "Không thêm thông tin mới, không thêm số trang hay tên sách. Trả lời bằng MỘT đối tượng JSON.")


def verbalize_user(question: str, supported: list[str], unresolved: list[str], status: str) -> str:
    s = "\n".join(f"- {x}" for x in supported) or "- (không có)"
    u = "\n".join(f"- {x}" for x in unresolved) or "- (không có)"
    return (f"Câu hỏi: {question}\nCác khẳng định đã kiểm chứng:\n{s}\nCác điểm chưa xác định:\n{u}\n"
            f"Trạng thái: {status}. Nếu còn điểm chưa xác định, nói rõ là chưa đủ bằng chứng.\n"
            'Trả về JSON: {"answer": "<đoạn trả lời ngắn gọn>"}')


def to_json(x: Any, limit: int = 6000) -> str:
    s = json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    return s if len(s) <= limit else s[: limit - 20] + "...<truncated>"


JSON_RETRY = "Phản hồi trước không phải JSON hợp lệ. Hãy trả lời lại bằng đúng MỘT đối tượng JSON theo mẫu đã cho."
