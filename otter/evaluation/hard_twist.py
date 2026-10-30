"""ViEduHop-Twist: shared helpers (stem/span overlap, stop-words). Prompts and generators are appended below in later sections.

Overlap metric (fixed BEFORE generation, 27 Oct 2026): content-word recall of the question STEM against the union of the gold spans,
  overlap(stem, spans) = |C(stem) & C(spans)| / |C(stem)|,   C(x) = set of lower-cased syllable tokens of x minus TWIST_STOP.
Thresholds come from the same measure on the existing ViEduHop test split (benchmarks/otter_hard/test.jsonl, gold_span_ids of each item):
all types except lexical_gap: stem overlap must be BELOW the median; lexical_gap: BELOW the 25th percentile.
"""
from __future__ import annotations

from otter.evaluation.hard_seeds import _STOP
from otter.evaluation.text import tokens

OVERLAP_VERSION = "twist_stem_overlap_content_recall_v1"
TWIST_STOP = set(_STOP) | set(
    "nào gì sau đây đúng sai câu hỏi phát biểu đâu bao nhiêu thế sao tại về qua bởi nếu thường mỗi cả tất khác nhau cùng "
    "trường hợp điều sách giáo khoa lớp nói theo trong là".split())


def content_set(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in TWIST_STOP and len(t) > 0}


def stem_overlap(stem: str, span_texts: list[str]) -> float:
    q = content_set(stem)
    if not q:
        return 0.0
    s: set[str] = set()
    for t in span_texts:
        s |= content_set(t)
    return len(q & s) / len(q)


# =====================================================================================================================
# Generation prompts (versioned). Prompt texts are exported to benchmarks/otter_hard/twist/prompts.json by the builder.
# =====================================================================================================================
import json
import random
import re
from collections import defaultdict
from typing import Any, Optional

from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation import book_specific as bs
from otter.evaluation.hard_gen import (VERIFY_SCHEMA, _bool, _label, _letter, assemble_options, chat_json, fmt_options, permute,
                                       span_block, structural_checks)
from otter.evaluation.hard_seeds import Seed, eligible
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.text import char_ngrams, jaccard, norm, sha1, tokens
from otter.llm.interface import LLM

TWIST_GEN_VERSION = "otter_hard_twist_gen_v1"
TWIST_GEN_VERSION_R1 = "otter_hard_twist_gen_v1.1"      # rounds r1+: v1 + the 3 extra quality rules in ADDENDUM_V11 (round r0 used v1)
TWIST_STRICT_VERSION = "otter_hard_twist_strict_v1"
TWIST_AE_GEN_VERSION = "otter_hard_twist_almost_enough_gen_v1"
TWIST_AE_GEN_VERSION_R1 = "otter_hard_twist_almost_enough_gen_v1.1"     # rounds r1+: v1 + AE_ADDENDUM_V11 and the giveaway/option-prefix filters
TWIST_SEEDS_VERSION = "otter_hard_twist_seeds_v1"
BENCHMARK_VERSION = "vieduhop_twist_v1"

TWIST_TYPES = ["lexical_gap", "negation_exception", "hidden_condition", "numeric", "misconception", "unnamed_bridge", "near_miss_lesson"]
TYPE_CODE = {"lexical_gap": "LEX", "negation_exception": "NEG", "hidden_condition": "CND", "numeric": "NUM", "misconception": "MIS",
             "unnamed_bridge": "BRG", "near_miss_lesson": "NML", "almost_enough": "AEN"}
TYPE_HOP = {"lexical_gap": "single", "negation_exception": "single", "hidden_condition": "conditional_rule", "numeric": "calculation",
            "misconception": "single", "unnamed_bridge": "bridge", "near_miss_lesson": "single"}
OVERLAP_TYPES_STRICT = {"lexical_gap"}          # stem overlap below the 25th percentile; all other types below the median
CLOSED_BOOK_REQUIRED = {t for t in TWIST_TYPES if t != "negation_exception"}

_SYS = ("Bạn là giáo viên giàu kinh nghiệm, đang ra đề trắc nghiệm KHÓ nhưng CÔNG BẰNG bám sát sách giáo khoa Việt Nam (chương trình GDPT 2018). "
        "Chỉ trả về JSON hợp lệ.")

_COMMON = """Môn: {subject}. Lớp: {grades}.
Dưới đây là {k} đoạn trích từ sách giáo khoa (mã [S1], [S2], ...).

{blocks}

NHIỆM VỤ: viết MỘT câu hỏi trắc nghiệm 4 phương án có “cú lắt” (twist) thuộc kiểu sau.

{spec}

YÊU CẦU CHUNG (bắt buộc):
1. Câu hỏi tự đứng được như một câu hỏi kiểm tra cho học sinh: TUYỆT ĐỐI không nhắc tới “đoạn trích”, “đoạn văn”, “văn bản trên”, “S1”, “theo thông tin trên”; không chép nguyên văn cả câu trong đoạn trích.
2. Đáp án đúng DUY NHẤT, suy ra chắc chắn từ các đoạn trích (cộng phép tính số học nếu cần); không cần kiến thức ngoài sách và không được sai về khoa học/lịch sử/văn học trong thực tế.
3. Hãy viết NĂM phương án nhiễu hợp lí, nghe có vẻ đúng với người chưa đọc kĩ sách, CHẮC CHẮN SAI theo đoạn trích, cùng loại và cùng cấu trúc câu với đáp án đúng, độ dài (số ký tự) mỗi nhiễu trong khoảng ±15% so với đáp án đúng; hệ thống sẽ chọn ra 3 nhiễu. Không dùng “tất cả đều đúng/sai”, “cả A và B”.
4. Số liệu, tên riêng, đơn vị lấy CHÍNH XÁC từ đoạn trích. Đáp án đúng không xuất hiện nguyên văn trong câu hỏi.
5. “evidence”: với MỖI đoạn trong gold_spans, trích nguyên văn (đúng từng chữ) một cụm 8–30 âm tiết làm căn cứ.
6. “gold_spans”: chỉ liệt kê các đoạn THỰC SỰ cần thiết để trả lời đúng (bỏ một đoạn mà vẫn trả lời được thì không liệt kê đoạn đó).
7. Nếu các đoạn trích không cho phép tạo câu hỏi đúng kiểu “cú lắt” này một cách công bằng và rõ ràng thì đặt "feasible": false (các trường khác để rỗng).{extra}
{variant}
Trả về JSON: {{"feasible": true|false, "twist_note": "<một câu: cú lắt nằm ở đâu>", "question": "...", "correct_answer": "...", \
"distractors": ["...", "...", "...", "...", "..."], "gold_spans": ["S1"], "evidence": {{"S1": "<cụm nguyên văn>"}}{schema_extra}, "twist_variant": "none"}}"""

SPECS = {
    "lexical_gap": """KIỂU 1 — lexical_gap (khoảng cách từ vựng): phần dẫn của câu hỏi VÀ các phương án dùng cách nói ĐỜI THƯỜNG (mô tả hiện tượng quan sát được, công dụng, ví dụ hằng ngày, diễn đạt lại bằng từ đồng nghĩa), TRÁNH chép các thuật ngữ/cụm từ khóa của đoạn trích. Học sinh phải nhận ra khái niệm mà sách nói tới thì mới chọn đúng. Ví dụ minh họa kiểu (KHÔNG phải nội dung đoạn): thay vì hỏi “Quang hợp là gì?” thì hỏi “Quá trình nào giúp cây xanh dùng ánh nắng để tự làm ra chất dinh dưỡng từ nước và khí có trong không khí?”.
- Phần dẫn dùng ít từ trùng với đoạn trích nhất có thể (đồng nghĩa, mô tả vòng); không dùng thuật ngữ khóa nào trong "avoided_terms".
- Liệt kê vào "avoided_terms" 2–6 thuật ngữ/cụm từ khóa của đoạn mà phần dẫn đã tránh dùng.
- Đáp án đúng phải đúng theo đoạn trích; nhiễu là mô tả đời thường của khái niệm lân cận dễ nhầm.""",
    "negation_exception": """KIỂU 2 — negation_exception (phủ định / ngoại lệ): phần dẫn dạng PHỦ ĐỊNH hoặc NGOẠI TRỪ, ví dụ “Phát biểu nào sau đây KHÔNG đúng theo sách giáo khoa …?”, “Nội dung nào sau đây KHÔNG thuộc …?”, “Tất cả các ý sau đều … NGOẠI TRỪ …”, hoặc câu có “chỉ khi”. Viết chữ KHÔNG / NGOẠI TRỪ bằng chữ in hoa. Ba phương án nhiễu là ba mục/phát biểu ĐÚNG lấy từ một danh sách hoặc từ các ý của đoạn trích; đáp án đúng là MỘT mục/phát biểu hợp lí nghe như đúng nhưng SAI hoặc KHÔNG nằm trong danh sách mà sách nêu — đoạn trích phải cho phép loại trừ nó chắc chắn (sách liệt kê đầy đủ “gồm …”, “chỉ có …”, hoặc nói điều ngược lại). Đáp án phải sai/không thuộc theo đoạn trích VÀ không thể coi là đúng theo hiểu biết thực tế. Câu hỏi nêu rõ chủ đề khoa học/lịch sử cụ thể (không hỏi về mục lục, tên sách hay cấu trúc sách) để đáp án là duy nhất. Với kiểu này các phương án nhiễu là các phát biểu ĐÚNG (không phải sai).""",
    "hidden_condition": """KIỂU 3 — hidden_condition (điều kiện ẩn): cùng một khái niệm/tính chất/quy tắc ĐÚNG trong điều kiện A nhưng SAI hoặc KHÁC trong điều kiện B (điều kiện tiêu chuẩn so với điều kiện khác, giai đoạn lịch sử, trước/sau một mốc thời gian, định nghĩa theo cấp học, loại đối tượng, môi trường…), cả hai điều kiện đều có căn cứ trong đoạn trích. Câu hỏi nêu một bối cảnh/điều kiện cụ thể (bằng lời tự nhiên, không nêu tên ngoại lệ) và hỏi kết luận đúng trong bối cảnh đó. Một phương án nhiễu là kết luận ĐÚNG ở điều kiện KHÁC mà học sinh hay áp dụng nhầm (bẫy điều kiện). Ghi rõ trong "condition_true" điều kiện làm đáp án đúng, và trong "condition_trap" điều kiện mà nhiễu bẫy tới.""",
    "numeric": """KIỂU 4 — numeric (số liệu): lấy một SỐ LIỆU từ một đoạn (số liệu KHÔNG được nêu trong câu hỏi), rồi đổi đơn vị hoặc so sánh/tính với số liệu ở đoạn khác (hoặc với số trong câu hỏi) rồi mới chọn đáp án. Đáp án là một con số kèm đơn vị, hoặc kết luận so sánh (lớn hơn/nhỏ hơn/gấp bao nhiêu lần) có căn cứ số. Nhiễu: quên đổi đơn vị, đổi sai hệ số lũy thừa của 10, so sánh ngược, dùng số liệu sai đoạn, tính sai một bước — mỗi nhiễu là kết quả của MỘT lỗi như vậy. Bắt buộc trường "calc": {"expression": "<biểu thức số học thuần túy chỉ gồm số và + - * / ** ( ), không dùng tên biến, cho ĐÚNG giá trị số của đáp án đúng>", "unit": "<đơn vị kết quả>"}; nếu đáp án là kết luận so sánh không phải số, đặt "calc": null nhưng vẫn phải dựa trên con số trong các đoạn. Chép chính xác các chữ số trong đoạn trích.""",
    "misconception": """KIỂU 5 — misconception (quan niệm sai lầm): đoạn trích nêu (hoặc ngầm sửa) một hiểu lầm phổ biến của học sinh. Viết một tình huống/khẳng định và hỏi cách hiểu ĐÚNG; đáp án đúng là cách hiểu đúng theo sách; ba phương án nhiễu là các QUAN NIỆM SAI PHỔ BIẾN mà học sinh thường mắc và sách đính chính (mỗi nhiễu một quan niệm sai khác nhau, nghe rất hợp lí). Ghi trong "misconceptions" mô tả ngắn từng quan niệm sai (niềm tin của học sinh) và vì sao đoạn trích bác bỏ. Không chọn đề tài mà quan niệm sai đó thực ra không bị đoạn trích bác bỏ rõ ràng.""",
    "unnamed_bridge": """KIỂU 6 — unnamed_bridge (cầu nối không gọi tên): các đoạn thuộc môn/bài khác nhau nhưng cùng chạm tới một đối tượng/khái niệm/hiện tượng TRUNG GIAN. Câu hỏi dẫn tới mục tiêu qua đối tượng trung gian đó mà KHÔNG gọi tên nó và không dùng thuật ngữ nhận diện nó: mô tả nó gián tiếp bằng đặc điểm lấy từ đoạn này (S1), rồi hỏi một thuộc tính/hệ quả/giá trị chỉ có trong đoạn kia (S2, S3). Phải dùng đủ các đoạn trong gold_spans; chỉ đọc một đoạn thì không trả lời được. Ghi tên đối tượng trung gian vào "bridge_concept" (tên này KHÔNG được xuất hiện trong câu hỏi). Đối tượng trung gian phải THẬT SỰ xuất hiện (hoặc được nói tới) trong từng đoạn; KHÔNG dùng phép ví von, ẩn dụ hay “tương tự giữa hai môn”, không ghép vì hai đoạn chỉ cùng từ khóa lỏng lẻo. Nếu các đoạn không thật sự có đối tượng trung gian chung để ghép một cách tự nhiên và chính xác thì feasible=false.""",
    "near_miss_lesson": """KIỂU 7 — near_miss_lesson (bài lân cận nói khác): [S1] và [S2] bàn về cùng một chủ đề nhưng ở bài/lớp/sách khác nhau và NÓI KHÁC NHAU (số liệu, cách định nghĩa, phạm vi, cách phân loại, mức độ). Câu hỏi nhắm tới nội dung của S1: nêu bối cảnh nhận dạng bài của S1 bằng lời tự nhiên (ví dụ tên sách/môn/lớp hoặc phạm vi khái niệm đúng như tiêu đề ghi ở đầu đoạn S1) sao cho CHỈ S1 cho đáp án đúng. Một phương án nhiễu là câu trả lời đúng theo S2 (bài lân cận), các nhiễu khác hợp lí. gold_spans CHỈ gồm S1; ghi S2 vào "near_miss_span": "S2". Nếu hai đoạn không thật sự nói khác nhau về điều có thể hỏi thì feasible=false.""",
}

_SCHEMA_EXTRA = {
    "lexical_gap": ', "avoided_terms": ["..."]',
    "negation_exception": "",
    "hidden_condition": ', "condition_true": "...", "condition_trap": "..."',
    "numeric": ', "calc": {"expression": "...", "unit": "..."} | null',
    "misconception": ', "misconceptions": ["..."]',
    "unnamed_bridge": ', "bridge_concept": "..."',
    "near_miss_lesson": ', "near_miss_span": "S2"',
}
_LIT_NOTE = ("\n8. LƯU Ý NGỮ VĂN: chỉ hỏi NỘI DUNG được nêu rõ trong đoạn (tác giả, hoàn cảnh sáng tác, thể loại, nhân vật, sự việc, chi tiết); không yêu cầu "
             "cảm thụ hay diễn giải riêng; nêu TÊN văn bản/tác phẩm cụ thể (lấy từ tiêu đề trong ngoặc ở đầu đoạn) khi cần xác định nguồn.")

VARIANT_HINTS = {
    "unit_chain": "GỢI Ý BIẾN THỂ (chỉ dùng nếu tự nhiên với các đoạn trích, nếu không thì bỏ qua và đặt twist_variant = \"none\"): chuỗi đổi đơn vị — cần đổi qua ít nhất hai bước đơn vị hoặc đại lượng (ví dụ đổi km/h sang m/s; khối lượng chất kèm độ tinh khiết/phần trăm rồi mới đổi sang số mol) bằng các số liệu và quan hệ có trong đoạn trích. Nếu dùng, đặt \"twist_variant\": \"unit_chain\".",
    "asked_differs_intermediate": "GỢI Ý BIẾN THỂ (chỉ dùng nếu tự nhiên với các đoạn trích, nếu không thì đặt twist_variant = \"none\"): đại lượng được hỏi KHÁC đại lượng trung gian tính được đầu tiên (ví dụ tính một phần rồi phải hỏi tổng, hoặc tính một giá trị trung gian rồi hỏi đại lượng suy ra từ nó); nhiễu gồm chính giá trị trung gian. Nếu dùng, đặt \"twist_variant\": \"asked_differs_intermediate\".",
    "superlative_stem": "GỢI Ý BIẾN THỂ (chỉ dùng nếu tự nhiên với các đoạn trích, nếu không thì đặt twist_variant = \"none\"): phần dẫn dạng cấp cao nhất (“… lớn nhất / nhiều nhất / cao nhất / quan trọng nhất / duy nhất …”) với các phương án đều có phần đúng nhưng chỉ MỘT phương án là cực trị theo đoạn trích (các nhiễu là phương án đúng một phần hoặc cực trị của bài lân cận). Nếu dùng, đặt \"twist_variant\": \"superlative_stem\".",
}
VARIANTS_BY_TYPE = {"numeric": ["unit_chain", "asked_differs_intermediate"], "misconception": ["superlative_stem"], "near_miss_lesson": ["superlative_stem"]}
SUPERLATIVE = re.compile(r"((?<!thứ )(?<!thống )(?<!hợp )(?<!duy )\bnhất\b|duy nhất|hàng đầu)", re.I)

AE_ADDENDUM_V11 = """
QUY TẮC BỔ SUNG: (i) "question" phải đọc như một câu hỏi kiểm tra BÌNH THƯỜNG, trả lời được trên mặt chữ — TUYỆT ĐỐI không nói rằng thiếu thông tin hoặc có điều chưa biết (không dùng “nếu không biết”, “chưa biết”, “nếu biết thêm”, “không đủ thông tin”, “không thể khẳng định”); (ii) mỗi phương án là một khẳng định/số liệu cụ thể, KHÔNG có tiền tố “A.”, “B.”; (iii) bốn phương án chỉ khác nhau ở giá trị/chi tiết phụ thuộc dữ kiện thiếu; (iv) dữ kiện thiếu phải là thứ sách giáo khoa nào cũng không nêu (ví dụ số liệu của một địa phương/thí nghiệm/năm cụ thể mà đoạn không đề cập)."""

AE_SPEC = """Dưới đây là một đoạn trích từ sách giáo khoa (mã [S1]).

{blocks}

NHIỆM VỤ: viết một cặp câu hỏi trắc nghiệm “trông có vẻ trả lời được nhưng sách THIẾU ĐÚNG MỘT dữ kiện quyết định”:
- "question": câu hỏi về đúng tình huống/đối tượng/quy tắc của đoạn S1, nghe như có thể trả lời từ sách, nhưng muốn chọn đáp án thì phải biết MỘT dữ kiện cụ thể (một con số, một giá trị, một điều kiện, một tên, một mốc thời gian) mà đoạn S1 KHÔNG cho và sách giáo khoa cũng không nêu; thiếu dữ kiện đó thì KHÔNG THỂ khẳng định phương án nào đúng (cũng không suy ra phủ định). Không được thiếu nhiều dữ kiện: chỉ thiếu MỘT.
- Bốn phương án A–D là bốn kết luận khả dĩ, cùng loại và độ dài tương đương; mỗi phương án sẽ là đáp án đúng nếu dữ kiện thiếu nhận một giá trị khác nhau; không phương án nào đúng được xác định từ S1.
- "missing_datum": mô tả ngắn dữ kiện thiếu (bằng tiếng Việt).
- "question_with_datum": cùng câu hỏi (cùng bốn phương án) nhưng NÊU RÕ dữ kiện đó bằng MỘT giá trị cụ thể, sao cho từ S1 suy ra chắc chắn đúng MỘT phương án; "key_with_datum": nội dung phương án đúng đó (phải trùng nguyên văn với một trong correct_answer).
- "correct_answer": phương án đúng khi có dữ kiện; "distractors": đúng 3 phương án còn lại (khi dữ kiện đã nêu thì chắc chắn sai theo S1, nhưng khi thiếu dữ kiện thì chúng khả dĩ).
YÊU CẦU: câu hỏi tự đứng được, không nhắc “đoạn trích”, “sách”, “S1”, “theo thông tin trên”; không hỏi phủ định; đáp án đúng không xuất hiện nguyên văn trong câu hỏi; câu hỏi gốc KHÔNG chứa dữ kiện thiếu; dữ kiện thiếu không được là kiến thức phổ thông mà học sinh có thể suy ra từ sách. Nếu đoạn không cho phép dựng cặp câu hỏi như vậy thì "feasible": false.{extra}

Trả về JSON: {{"feasible": true|false, "missing_datum": "...", "question": "...", "question_with_datum": "...", "correct_answer": "...", "distractors": ["...", "...", "..."]}}"""

ADDENDUM_V11 = """
8. Chỉ hỏi điều mà đoạn trích NÊU RÕ bằng lời; không hỏi ý kiến chủ quan, “yếu tố nào quan trọng nhất / chủ yếu nhất” hay đánh giá chung chung nếu đoạn không xếp hạng rõ ràng (trừ khi gợi ý biến thể cấp cao nhất bên dưới áp dụng và đoạn nói rõ cực trị).
9. Mỗi phương án nhiễu phải SAI RÕ RÀNG theo đoạn trích (không phải “đúng một phần”), và không phương án nào đúng theo hiểu biết thực tế; đáp án đúng phải đúng cả theo đoạn trích lẫn theo thực tế.
10. Bám vào chi tiết ĐẶC THÙ của đoạn (số liệu, ví dụ, điều kiện, tên riêng, cách trình bày của sách) để người chỉ biết kiến thức phổ thông dễ chọn nhầm; tránh câu hỏi mà kiến thức phổ thông đã đủ để trả lời."""

PROMPT_TEXTS = {"onto_hint": "see build_prompt: ontology relation-chain hint appended to the unnamed_bridge spec for r3 seeds", "addendum_v1.1": ADDENDUM_V11, "almost_enough_addendum_v1.1": AE_ADDENDUM_V11, "common": _COMMON, "specs": SPECS, "schema_extra": _SCHEMA_EXTRA, "literature_note": _LIT_NOTE, "system": _SYS,
                "almost_enough": AE_SPEC}


# =====================================================================================================================
# Seeds
# =====================================================================================================================
_DEF = re.compile(r"(\bđược gọi là\b|\bgọi là\b|\bđược hiểu là\b|\bcó nghĩa là\b|\bkhái niệm\b|\bđịnh nghĩa\b|\bnghĩa là\b|(?<![a-zà-ỹ])là (?:một|quá trình|sự|hiện tượng|tập hợp|các|những))", re.I)
_MISC = re.compile(r"(không phải|chứ không|không đúng|nhầm lẫn|hiểu lầm|quan niệm|không có nghĩa|không đồng nghĩa|không đồng nhất|trái với|"
                   r"không chỉ|không thể|ngược lại|không nên|khác với|chỉ là)", re.I)
_CONTRAST = re.compile(r"(còn |ngược lại|trái lại|tuy nhiên|trong khi|nếu |khi |ở điều kiện|điều kiện|tùy thuộc|tuỳ thuộc|trước năm|sau năm|"
                       r"giai đoạn|thời kì|thời kỳ|chỉ khi|trừ |ngoại trừ|đối với|nhiệt độ|áp suất|theo hướng)", re.I)
_UNITS = re.compile(r"\d\s*(?:m/s|km/h|km|kg|mol|°C|K\b|Pa|atm|J\b|N\b|W\b|V\b|A\b|Hz|cm|mm|nm|%|g\b|lít|L\b|mL|kJ|eV|MeV|tấn|triệu|nghìn|tỉ|ha\b|"
                    r"giờ|phút|giây|s\b|năm|thế kỉ|người|km2|km²|m2|m²|m3|m³|USD|đồng)", re.I)
_LIST = re.compile(r"(gồm|bao gồm|chia thành|phân (?:chia|loại|thành)|gồm có|có (?:hai|ba|bốn|năm|sáu|bảy|\d) )", re.I)


_META = re.compile(r"(Chuyên đề học tập|Bộ sách|Cánh Diều|Kết nối tri thức|Chân trời sáng tạo|Nhà xuất bản|Bản quyền|Hội đồng quốc gia thẩm định|"
                   r"Lời nói đầu|Mục lục|Hướng dẫn sử dụng sách)", re.I)


def load_page_quality(root) -> dict:
    import glob
    pq = {}
    for p in glob.glob(str(root / "data/textbooks/parsed/nxbgd_*/pages.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            pq[(r["book_id"], r["pdf_page_index"])] = r.get("quality_score")
    return pq


def twist_pool(corpus: SpanCorpus, pageq: dict) -> list[Span]:
    out = []
    for s in corpus.spans:
        if not (s.domain and eligible(s, min_chars=120, max_chars=1900)):
            continue
        q = pageq.get((s.book_id, s.pdf_page_index))
        if (q is not None and q < 0.9) or bs._OCR_SUSPECT.search(s.text) or s.text.count("$") > 8:
            continue
        if _META.search(s.text) or _META.search(" ".join(s.heading_path[-2:])):      # publisher / book-listing pages, not subject content
            continue
        out.append(s)
    return out


def _seed(strategy: str, spans: list[Span], link: dict, tag: str) -> Seed:
    hop = "bridge" if len(spans) > 1 else "single"
    return Seed(seed_id="tws_" + sha1("|".join([strategy, tag] + [s.span_id for s in spans]))[:12], strategy=strategy, span_ids=[s.span_id for s in spans],
                domain=spans[0].domain, grade=max((s.grade for s in spans if s.grade is not None), default=None), grades=[s.grade for s in spans],
                book_ids=list(dict.fromkeys(s.book_id for s in spans)), lesson_keys=list(dict.fromkeys(s.lesson_key for s in spans)),
                target_hop_type=hop, compatible_hop_types=[], link=link, version=TWIST_SEEDS_VERSION)


def _balanced(cands: list[tuple[float, Any]], key, n: int, rng: random.Random, domains: list[str], per_domain_cap: Optional[dict] = None) -> list[Any]:
    """Round-robin over subjects, best score first (score + noise), at most per_domain_cap[domain]."""
    by: dict[str, list] = defaultdict(list)
    for sc, x in sorted(cands, key=lambda z: -(z[0] + rng.random() * 1.5)):
        by[key(x)].append(x)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        d = domains[i % len(domains)]
        i += 1
        if by.get(d) and (not per_domain_cap or sum(1 for o in out if key(o) == d) < per_domain_cap.get(d, 10 ** 9)):
            out.append(by[d].pop(0))
        elif i > 50 * len(domains) + n * len(domains):
            break
    return out


def build_seeds(corpus: SpanCorpus, pool: list[Span], dense: "Any", pos: dict, used_pairs: set, anchors: list[dict], *, n: dict[str, int],
                seed: int = 71, exclude_by_type: Optional[dict] = None, round_tag: str = "r0") -> list[Seed]:
    """Seed spans per twist type from cue statistics of the corpus (no external text). Types use at most 3 seeds per span."""
    import numpy as np
    rng = random.Random(seed)
    ex = exclude_by_type or {}
    P = lambda t: [s for s in pool if s.span_id not in ex.get(t, set())]
    doms = ["physics", "chemistry", "biology", "geography", "history", "literature"]
    out: list[Seed] = []
    d_of = lambda s: s.domain

    # 1 lexical_gap: definitional/explanatory spans of moderate length
    c = []
    for s in P("lexical_gap"):
        t = s.text
        if 150 <= len(t) <= 900 and _DEF.search(t) and s.pedagogical_type in ("definition", "concept_explanation", "law", "principle", "process"):
            c.append((len(_DEF.findall(t)) + (1 if s.pedagogical_type == "definition" else 0), s))
    out += [_seed("lexical_gap", [s], {}, round_tag) for s in _balanced(c, d_of, n.get("lexical_gap", 0), rng, doms, {"literature": max(2, n.get("lexical_gap", 0) // 12)})]

    # 2 negation_exception: enumerations / closed lists
    c = []
    for s in P("negation_exception"):
        t = s.text
        nl = len(bs._LISTLINE.findall(t))
        k = len(_LIST.findall(t))
        if (nl >= 3 or k >= 1) and t.count(",") + t.count(";") + nl >= 3 and len(t) >= 200 and s.domain != "literature":
            c.append((nl + k * 1.5 + min(t.count(","), 8) / 4, s))
    out += [_seed("negation_exception", [s], {}, round_tag) for s in _balanced(c, d_of, n.get("negation_exception", 0), rng, doms)]

    # 3 hidden_condition: spans with contrasting conditions
    c = []
    for s in P("hidden_condition"):
        t = s.text
        k = len(_CONTRAST.findall(t))
        if k >= (3 if round_tag == "r0" else 2) and len(t) >= 200 and s.domain != "literature":
            c.append((min(k, 8) + (2 if re.search(r"(ở điều kiện|điều kiện|trước năm|sau năm|giai đoạn|thời kì)", t, re.I) else 0), s))
    n_hc = n.get("hidden_condition", 0)
    n_hc_pairs = int(n_hc * 0.3) if round_tag != "r0" else 0           # r1+: 30% two-span items (same concept stated under different conditions/levels)
    out += [_seed("hidden_condition", [s], {}, round_tag) for s in _balanced(c, d_of, n_hc - n_hc_pairs, rng, doms)]

    # 4 numeric: pairs of numeric spans from the same subject (topic-related by bge-m3) + single spans with unit-bearing figures
    num = [s for s in P("numeric") if s.domain != "literature" and len(_UNITS.findall(s.text)) >= 1 and bs.span_nums(s.text) >= 2]
    idx = {s.span_id: i for i, s in enumerate(num)}
    M = dense[[pos[s.span_id] for s in num]]
    sim = M @ M.T
    pairs = []
    for i, a in enumerate(num):
        order = np.argsort(-sim[i])[:12]
        for j in order:
            b = num[int(j)]
            if b.span_id != a.span_id and a.domain == b.domain and a.lesson_key != b.lesson_key and sim[i, int(j)] >= 0.5:
                pairs.append((float(sim[i, int(j)]) * 4 + len(_UNITS.findall(a.text)) / 4, (a, b)))
    pairs = [(sc, p) for sc, p in pairs if p[0].span_id < p[1].span_id or True]
    n_num = n.get("numeric", 0)
    sel = _balanced(pairs, lambda p: p[0].domain, int(n_num * 0.65), rng, doms)
    out += [_seed("numeric", list(p), {"kind": "pair"}, round_tag) for p in sel]
    singles = [(len(_UNITS.findall(s.text)) + bs.span_nums(s.text) / 4, s) for s in num]
    out += [_seed("numeric", [s], {"kind": "single"}, round_tag) for s in _balanced(singles, d_of, n_num - int(n_num * 0.65), rng, doms)]

    # 5 misconception: spans that correct/negate a belief
    c = []
    for s in P("misconception"):
        t = s.text
        k = len(_MISC.findall(t))
        strong = bool(re.search(r"(nhầm lẫn|hiểu lầm|quan niệm|không phải|chứ không|không có nghĩa|không đồng nghĩa)", t, re.I))
        if (k >= 1 or strong) and len(t) >= 180:
            c.append((min(k, 6) + (2 if strong else 0), s))
    out += [_seed("misconception", [s], {}, round_tag) for s in _balanced(c, d_of, n.get("misconception", 0), rng, doms,
                                                                          {"literature": max(2, n.get("misconception", 0) // 10)})]

    # 6 unnamed_bridge: cross-subject anchor pairs (ontology concept / topic-bridge / LLM-judged in the existing cross-subject build)
    pset = {s.span_id for s in P("unnamed_bridge")}
    ab = [(1.0, r) for r in anchors if r["a"] in pset and r["b"] in pset and frozenset((r["a"], r["b"])) not in used_pairs]
    pairs_ = _balanced(ab, lambda r: "+".join(sorted((corpus.get(r["a"]).domain, corpus.get(r["b"]).domain))),
                       n.get("unnamed_bridge", 0), rng, sorted({"+".join(sorted((corpus.get(r["a"]).domain, corpus.get(r["b"]).domain))) for _, r in ab}))
    for r in pairs_:
        a, b = (corpus.get(r["a"]), corpus.get(r["b"]))
        if rng.random() < 0.5:
            a, b = b, a
        out.append(_seed("unnamed_bridge", [a, b], {"sources": r["sources"], "pair_id": r.get("pair_id")}, round_tag))

    # 7 near_miss_lesson: same-subject spans of different lessons with high bge-m3 similarity (prefer other grade/book)
    non_lit = [s for s in P("near_miss_lesson") if s.domain != "literature"]
    M2 = dense[[pos[s.span_id] for s in non_lit]]
    sim2 = M2 @ M2.T
    cand = []
    for i, a in enumerate(non_lit):
        for j in np.argsort(-sim2[i])[:6]:
            b = non_lit[int(j)]
            sv = float(sim2[i, int(j)])
            if a.span_id == b.span_id or a.domain != b.domain or a.lesson_key == b.lesson_key or sv < 0.68 or a.span_id > b.span_id:
                continue
            if jaccard(char_ngrams(a.text), char_ngrams(b.text)) > 0.6:
                continue
            cand.append((sv * 4 + (1.5 if a.grade != b.grade else 0) + (0.5 if a.book_id != b.book_id else 0), (a, b)))
    sel = _balanced(cand, lambda p: p[0].domain, n.get("near_miss_lesson", 0), rng, doms)
    for a, b in sel:
        a, b = (a, b) if rng.random() < 0.5 else (b, a)
        out.append(_seed("near_miss_lesson", [a, b], {"near_miss_span_id": b.span_id}, round_tag))
    if n_hc_pairs:
        chosen = {frozenset((a.span_id, b.span_id)) for a, b in sel}
        rest = [(sc, p) for sc, p in cand if frozenset((p[0].span_id, p[1].span_id)) not in chosen and p[0].grade != p[1].grade]
        for a, b in _balanced(rest, lambda p: p[0].domain, n_hc_pairs, rng, doms):
            out.append(_seed("hidden_condition", [a, b], {"kind": "pair"}, round_tag))
    for i, sd in enumerate(x for x in out if x.strategy in VARIANTS_BY_TYPE):
        opts = VARIANTS_BY_TYPE[sd.strategy]
        if sd.strategy == "numeric":
            h = opts[i % 3] if i % 3 < len(opts) else None           # 1/3 unit_chain, 1/3 asked_differs_intermediate, 1/3 no hint
        else:
            h = opts[0] if i % 3 == 0 else None                      # 1/3 superlative hint
        sd.link = dict(sd.link or {}, variant_hint=h)
    return out


def onto_bridge_seeds(corpus: SpanCorpus, onto, used_pairs: set, pool_ids: set, *, n: int, seed: int = 73, round_tag: str = "r3") -> list[Seed]:
    """unnamed_bridge seeds from the ontology relation chains (A -rel-> B -rel-> C, evidence of A->B in S1 and of B->C in S2); B is the
    never-named intermediate concept. Same-subject chains; chains already used by an existing ViEduHop item are skipped."""
    from otter.evaluation import hard_v2 as v2
    rng = random.Random(seed)
    cand = [sd for sd in v2.relation_chain_seeds(corpus, onto, seed=seed, max_per_subject=400, max_span_uses=2, min_chars=100)
            if all(x in pool_ids for x in sd.span_ids) and frozenset(sd.span_ids) not in used_pairs]
    rng.shuffle(cand)
    by: dict[str, list] = defaultdict(list)
    for sd in cand:
        by[sd.domain].append(sd)
    out, doms = [], sorted(by)
    i = 0
    while len(out) < n and any(by.values()):
        d = doms[i % len(doms)]
        i += 1
        if by[d]:
            sd = by[d].pop(0)
            sp = [corpus.get(x) for x in sd.span_ids]
            out.append(_seed("unnamed_bridge", sp, {"onto": sd.link["onto"], "round": round_tag, "kind": "onto_chain"}, round_tag))
    return out


def almost_enough_seeds(pool: list[Span], n: int, seed: int = 72, round_tag: str = "r0", exclude_spans: Optional[set] = None) -> list[Seed]:
    rng = random.Random(seed)
    doms = ["physics", "chemistry", "biology", "geography", "history"]
    c = []
    for s in pool:
        if s.domain == "literature" or s.span_id in (exclude_spans or set()):
            continue
        t = s.text
        k = len(_CONTRAST.findall(t)) + min(bs.span_nums(t), 6) / 2 + (2 if re.search(r"(nếu|khi) .{3,80} thì", t, re.I) else 0)
        if k >= 3 and 200 <= len(t) <= 1400:
            c.append((k, s))
    return [_seed("almost_enough", [s], {}, round_tag) for s in _balanced(c, lambda s: s.domain, n, rng, doms)]


# =====================================================================================================================
# Generation
# =====================================================================================================================
def gen_schema() -> dict:
    return {"type": "object", "properties": {"feasible": {"type": "boolean"}, "twist_note": {"type": "string"}, "question": {"type": "string"},
                                             "correct_answer": {"type": "string"}, "distractors": {"type": "array", "items": {"type": "string"}},
                                             "gold_spans": {"type": "array", "items": {"type": "string"}}, "evidence": {"type": "object"}},
            "required": ["feasible", "question", "correct_answer", "distractors", "gold_spans", "evidence"]}


def _is_r1(seed: Seed) -> bool:
    return (seed.link or {}).get("round", "r0") != "r0"


def gen_version(seed: Seed) -> str:
    return TWIST_GEN_VERSION_R1 if _is_r1(seed) else TWIST_GEN_VERSION


def cand_id_of(seed: Seed, model: str, sample: int = 0) -> str:
    return "tw_" + sha1(f"{seed.seed_id}{gen_version(seed)}{model}{sample}")[:12]


def build_prompt(seed: Seed, corpus: SpanCorpus) -> str:
    spans = [corpus.get(i) for i in seed.span_ids]
    labels = [f"S{i + 1}" for i in range(len(spans))]
    grades = sorted({s.grade for s in spans if s.grade is not None})
    subjects = " / ".join(dict.fromkeys(DOMAIN_TO_SUBJECT_VI.get(s.domain, s.domain) for s in spans))
    spec = SPECS[seed.strategy]
    ch = ((seed.link or {}).get("onto") or {}).get("chain")
    if ch:
        spec += ("\nGỢI Ý TỪ ONTOLOGY (chuỗi quan hệ đã trích từ chính các đoạn; mắt xích 1 nằm ở S1, mắt xích 2 ở S2; đối tượng trung gian là “" + ch[0]["to"] +
                 "” — KHÔNG được nêu tên hay thuật ngữ này trong câu hỏi hay phương án):\n" +
                 "\n".join(f"- {c['from']} --{c['rel']}--> {c['to']}" for c in ch))
    p = _COMMON.format(subject=subjects, grades=", ".join(map(str, grades)) or "?", k=len(spans),
                       blocks="\n\n".join(span_block(l, s, 1800) for l, s in zip(labels, spans)), spec=spec,
                       extra=(ADDENDUM_V11 if _is_r1(seed) else "") + (_LIT_NOTE.replace("\n8.", "\n11.") if any(s.domain == "literature" for s in spans) else ""), schema_extra=_SCHEMA_EXTRA[seed.strategy],
                       variant=VARIANT_HINTS.get((seed.link or {}).get("variant_hint"), ""))
    return p


def generate_twist(seed: Seed, corpus: SpanCorpus, llm: LLM, *, sample: int = 0, temperature: float = 0.7, max_tokens: int = 1800) -> dict[str, Any]:
    spans = [corpus.get(i) for i in seed.span_ids]
    labels = [f"S{i + 1}" for i in range(len(spans))]
    lab2id = dict(zip(labels, seed.span_ids))
    cid = cand_id_of(seed, llm.model, sample)
    base = {"cand_id": cid, "seed_id": seed.seed_id, "twist_type": seed.strategy, "domain": seed.domain, "grade": seed.grade,
            "seed_span_ids": seed.span_ids, "strategy": seed.strategy, "sample": sample, "seed_link": seed.link,
            "generator": {"model": llm.model, "prompt_version": gen_version(seed), "temperature": temperature}}
    prompt = build_prompt(seed, corpus)
    try:
        o, r = chat_json(llm, [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}], gen_schema(), max_tokens=max_tokens,
                         temperature=temperature, validate=lambda o: "feasible" in o)
    except Exception as e:  # noqa: BLE001
        return {**base, "feasible": False, "error": f"{type(e).__name__}: {e}"}
    base["generator"].update(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens)
    base["gen_raw"] = o
    feasible = _bool(o.get("feasible")) and bool(str(o.get("question", "")).strip())
    key = str(o.get("correct_answer", "")).strip()
    raw = [str((d.get("text") if isinstance(d, dict) else d) or "").strip() for d in o.get("distractors") or []]
    raw = [x for i, x in enumerate(raw) if x and x.lower() != key.lower() and x.lower() not in {y.lower() for y in raw[:i]}]
    gold = [lab2id[x] for x in dict.fromkeys(_label(g, labels) for g in o.get("gold_spans") or []) if x in lab2id]
    ev = {}
    for k, v in (o.get("evidence") or {}).items():
        l = _label(k, labels)
        if l and isinstance(v, str):
            ev[lab2id[l]] = v
    c = {**base, "feasible": feasible, "twist_note": str(o.get("twist_note", "")), "question": str(o.get("question", "")).strip(), "correct_answer": key,
         "distractors_offered": raw, "gold_span_ids": gold, "evidence": ev, "min_spans": 1 if len(gold) <= 1 else 2,
         "hop_type": TYPE_HOP[seed.strategy], "link": str(o.get("twist_note", "")),
         "calc": o.get("calc") if isinstance(o.get("calc"), dict) else None,
         "reasoning_steps": [{"span_id": g, "step": ev.get(g, "")} for g in gold]}
    # type specific extras
    st = seed.strategy
    if st == "lexical_gap":
        c["avoided_terms"] = [str(x) for x in o.get("avoided_terms") or [] if str(x).strip()]
    if st == "hidden_condition":
        c["condition_true"], c["condition_trap"] = str(o.get("condition_true", "")), str(o.get("condition_trap", ""))
    if st == "misconception":
        c["misconceptions"] = [str(x) for x in o.get("misconceptions") or []]
    if st == "unnamed_bridge":
        c["bridge_concept"] = str(o.get("bridge_concept", "")).strip()
    if st == "near_miss_lesson":
        nm = _label(o.get("near_miss_span"), labels)
        c["near_miss_span_id"] = lab2id.get(nm) if nm else seed.link.get("near_miss_span_id")
        if c["near_miss_span_id"] in gold:
            c["near_miss_span_id"] = seed.link.get("near_miss_span_id")
    tv = str(o.get("twist_variant", "none")).strip()
    if tv not in VARIANTS_BY_TYPE.get(st, []):
        tv = "none"
    if tv == "superlative_stem" and not SUPERLATIVE.search(c["question"]):
        tv = "none"
    if tv in ("unit_chain", "asked_differs_intermediate") and not c["calc"]:
        tv = "none"
    c["twist_variant"] = tv
    c["variant_hint"] = (seed.link or {}).get("variant_hint")
    pick = bs.pick_distractors(key, raw, cid) if len(raw) >= 3 else raw
    c["distractors"] = pick[:3]
    c["distractor_notes"] = []
    if feasible and len(c["distractors"]) == 3 and key:
        c["options"], c["answer"] = assemble_options(cid, key, c["distractors"])
    else:
        c["feasible"] = False
        c["infeasible_reason"] = "model_infeasible" if not feasible else "fewer_than_3_distractors_or_no_key"
    return c


# =====================================================================================================================
# Deterministic checks
# =====================================================================================================================
def length_ok_twist(key: str, dis: list[str], options: dict, answer: str) -> bool:
    """Length-cue rule (fixed before screening): the key is not the unique longest option and its length is within 0.6-1.4 x the mean
    distractor length (or within 6 characters)."""
    if not dis:
        return False
    m = sum(len(d) for d in dis) / len(dis)
    return not bs.unique_longest(options or {}, answer) and (0.6 <= len(key) / max(1.0, m) <= 1.4 or abs(len(key) - m) <= 6)


def twist_structural(c: dict, corpus: SpanCorpus, thresholds: dict) -> dict[str, Any]:
    f = structural_checks(c, corpus)                      # four distinct options, meta references, answer in question, length ratio, ...
    tt = c["twist_type"]
    gold = c.get("gold_span_ids") or []
    seed_ids = set(c.get("seed_span_ids") or [])
    gtxt = {g: corpus.get(g).text for g in gold if corpus.get(g)}
    f["gold_in_seed"] = bool(gold) and set(gold) <= seed_ids
    ev = c.get("evidence") or {}
    f["evidence_verbatim"] = bool(gold) and all(g in ev and bs.quote_in_span(ev[g], gtxt.get(g, "")) for g in gold)
    alltxt = " ".join(gtxt.values()).replace(",", ".")
    calc_item = bool(c.get("calc"))
    key_nums = bs.nontrivial_numbers(c.get("correct_answer", ""))
    f["key_numbers_grounded"] = calc_item or tt == "numeric" or all(n in alltxt for n in key_nums)
    qn = bs.nontrivial_numbers(re.sub(r"(lớp|bài|chương|hình|bảng|trang|tập)\s*\d+", "", c.get("question", ""), flags=re.I))
    f["question_numbers_grounded"] = calc_item or tt == "numeric" or all(n in alltxt for n in qn)
    f["deictic_free"] = bs.deictic_free(c.get("question", ""))
    hay = [corpus.get(g).text + " " + " ".join(corpus.get(g).heading_path) + " " + (corpus.get(g).book_title or "") for g in gold if corpus.get(g)]
    f["names_grounded"] = bs.names_grounded(c.get("question", ""), hay)
    f["length_ok"] = length_ok_twist(c.get("correct_answer", ""), c.get("distractors", []), c.get("options") or {}, c.get("answer"))
    # type-specific deterministic conditions
    extra_ok, extra_fail = True, []
    if tt == "lexical_gap":
        at = [norm(x) for x in c.get("avoided_terms") or [] if len(norm(x)) >= 3]
        f["avoided_terms_declared"] = len(at) >= 2
        f["avoided_terms_absent"] = all(x not in norm(c["question"]) for x in at)
        extra_fail += [k for k in ("avoided_terms_declared", "avoided_terms_absent") if not f[k]]
    if tt == "numeric":
        from otter.evaluation import hard_v2 as v2
        cc = v2.check_calc(c)
        f["calc_present"] = cc["calc_present"]
        f["calc_ok"] = bool(cc["calc_ok"]) if cc["calc_present"] else True
        expr = str((c.get("calc") or {}).get("expression", ""))
        # a genuine multi-step numeric item: an explicit computation over >= 2 numbers with an operator (not a bare look-up of one figure)
        f["key_has_number_or_calc"] = cc["calc_present"] and len(v2.parse_numbers(expr)) >= 2 and bool(re.search(r"[-+*/]", expr.lstrip("-")))
        extra_fail += [k for k in ("calc_ok", "key_has_number_or_calc") if not f[k]]
    if tt == "unnamed_bridge":
        bc = norm(c.get("bridge_concept", ""))
        f["bridge_declared"] = len(bc) >= 3
        f["bridge_not_in_question"] = bool(bc) and bc not in norm(c["question"]) and all(bc not in norm(o) for o in (c.get("options") or {}).values())
        f["two_plus_gold"] = len(gold) >= 2
        mid = (((c.get("seed_link") or {}).get("onto") or {}).get("chain") or [{}])[0].get("to")
        if mid and len(norm(mid)) >= 4:
            f["onto_bridge_not_in_question"] = norm(mid) not in norm(c["question"]) and all(norm(mid) not in norm(o) for o in (c.get("options") or {}).values())
        else:
            f["onto_bridge_not_in_question"] = True
        extra_fail += [k for k in ("bridge_declared", "bridge_not_in_question", "two_plus_gold", "onto_bridge_not_in_question") if not f[k]]
    if tt == "near_miss_lesson":
        f["near_miss_declared"] = bool(c.get("near_miss_span_id")) and c["near_miss_span_id"] in seed_ids and c["near_miss_span_id"] not in gold
        f["single_gold"] = len(gold) == 1
        extra_fail += [k for k in ("near_miss_declared", "single_gold") if not f[k]]
    if tt == "negation_exception":
        f["negation_marker"] = bool(re.search(r"(KHÔNG|NGOẠI TRỪ|không phải|chỉ khi|chỉ|không)", c["question"]))
        extra_fail += [k for k in ("negation_marker",) if not f[k]]
    if tt in ("hidden_condition", "misconception") and False:
        pass
    if tt != "near_miss_lesson" and tt != "unnamed_bridge" and tt != "numeric" and tt != "hidden_condition":
        pass
    base_keys = ("gold_in_seed", "evidence_verbatim", "key_numbers_grounded", "question_numbers_grounded", "deictic_free", "names_grounded", "length_ok")
    if tt == "hidden_condition" and c.get("min_spans") == 1:
        pass
    hard_fail = [k for k in base_keys if not f[k]] + extra_fail
    # structural_checks requires >= 2 gold spans / reasoning in 2 spans only for multi-span items (min_spans)
    f["twist_structural_fail"] = f["structural_fail"] + hard_fail
    f["twist_structural_ok"] = f["structural_ok"] and not hard_fail
    # overlap (rule d)
    ov = stem_overlap(c["question"], list(gtxt.values()))
    thr = thresholds["threshold_lexical_gap_below"] if tt in OVERLAP_TYPES_STRICT else thresholds["threshold_default_below"]
    f["stem_overlap"] = round(ov, 4)
    f["stem_overlap_threshold"] = round(thr, 4)
    f["overlap_ok"] = ov < thr
    return f


# =====================================================================================================================
# LLM checks
# =====================================================================================================================
def check_gold_sample(c: dict, corpus: SpanCorpus, llm: LLM, i: int) -> dict:
    """Rule (a): main model with only the gold spans; sample i has its own option permutation and span order."""
    from otter.evaluation.hard_gen import check_verify
    r = check_verify(dict(c, cand_id=c["cand_id"] + f"#g{i}"), corpus, llm)
    r["sample"] = i
    return r


def check_nm_only(c: dict, corpus: SpanCorpus, llm: LLM, i: int) -> dict:
    """Near-miss lesson only: the other-lesson span alone must NOT yield the key."""
    from otter.evaluation.hard_gen import check_verify
    r = check_verify(dict(c, cand_id=c["cand_id"] + f"#n{i}", gold_span_ids=[c["near_miss_span_id"]]), corpus, llm)
    r["sample"] = i
    return r


def fold_gold(c: dict, get) -> dict:
    """(a): key chosen in >= 2 of 3 samples; (a2): in >= 2 of 3 samples the key is 'correct' and no distractor is 'correct'."""
    rs = [get(f"gold:{i}") for i in range(3)]
    if any(r is None for r in rs):
        return {"gold_done": False}
    key = c["answer"]
    n_key = sum(1 for r in rs if r["answer"] == key)
    uniq = sum(1 for r in rs if r["option_status"].get(key) == "correct" and all(r["option_status"].get(k) != "correct" for k in LETTERS if k != key))
    out = {"gold_done": True, "gold_n_key": n_key, "gold_ok": n_key >= 2, "gold_n_unique": uniq, "gold_unique_ok": uniq >= 2}
    if c["twist_type"] == "negation_exception":
        ot = [get(f"optruth:{i}") for i in range(2)]
        out["optruth_done"] = all(x is not None for x in ot)
        out["optruth_ok"] = all(x["ok"] for x in ot) if out["optruth_done"] else None
        out["gold_unique_ok"] = out["optruth_ok"]               # for 'which is NOT true' items option_status is not meaningful: per-option truth instead
    return out


def fold_nm(c: dict, get) -> dict:
    rs = [get(f"nm:{i}") for i in range(3)]
    if any(r is None for r in rs):
        return {"nm_done": False}
    n_key = sum(1 for r in rs if r["answer"] == c["answer"])
    return {"nm_done": True, "nm_n_key": n_key, "nm_ok": n_key <= 1}


_STRICT_NOTE = {
    "negation_exception": "Đây là câu hỏi PHỦ ĐỊNH/NGOẠI TRỪ: đáp án đúng là phương án KHÔNG thỏa mệnh đề của câu hỏi (sai hoặc ngoài danh sách), ba phương án còn lại ĐÚNG theo đoạn trích. "
                          "“key_supported” = đoạn trích cho phép kết luận chắc chắn phương án được chọn là phương án bị loại và ba phương án kia đều đúng.",
    "lexical_gap": "Câu hỏi cố ý dùng từ ngữ đời thường thay cho thuật ngữ của sách; chỉ đánh “question_clear” false nếu cách diễn đạt mơ hồ hoặc không xác định được khái niệm.",
    "hidden_condition": "Câu hỏi có điều kiện ẩn: đánh “unique_key” false nếu điều kiện nêu trong câu hỏi không đủ để phân biệt chắc chắn với điều kiện bẫy.",
    "numeric": "Câu hỏi cần số liệu từ đoạn trích và phép tính/đổi đơn vị: kiểm tra lại phép tính.",
    "misconception": "Các phương án nhiễu là quan niệm sai phổ biến; đánh “unique_key” false nếu một quan niệm sai được coi là đúng trong thực tế.",
    "unnamed_bridge": "Câu hỏi cố ý không gọi tên khái niệm trung gian; đánh “question_clear” false nếu mô tả trung gian mơ hồ (có thể chỉ nhiều khái niệm) hoặc ghép sai.",
    "near_miss_lesson": "Câu hỏi phải xác định đúng bài/ngữ cảnh của đoạn trích; đánh “question_clear” false nếu ngữ cảnh nêu trong câu hỏi không đủ để phân biệt với bài lân cận.",
}
_STRICT_PROMPT = bs._STRICT_PROMPT.replace("Tiêu chí (true = đạt):", "LƯU Ý RIÊNG: {note}\n\nTiêu chí (true = đạt):")


def check_strict_twist(c: dict, corpus: SpanCorpus, llm_thinking: LLM, perm: int = 0) -> dict:
    gold = list(c["gold_span_ids"])
    labels = [f"E{i + 1}" for i in range(len(gold))]
    opts, ans, back = permute(c["options"], c["answer"], c["cand_id"] + f":strict{perm}")
    p = _STRICT_PROMPT.format(blocks="\n\n".join(span_block(l, corpus.get(s)) for l, s in zip(labels, gold)), question=c["question"],
                              options=fmt_options(opts), answer=f"{ans}. {opts[ans]}", note=_STRICT_NOTE[c["twist_type"]])
    keys = ("key_supported", "unique_key", "question_clear", "facts_accurate", "options_distinct")
    o, _ = chat_json(llm_thinking, [{"role": "user", "content": p}], bs.STRICT_SCHEMA, max_tokens=4000, temperature=0.0, validate=lambda o: all(k in o for k in keys))
    res = {k: _bool(o[k]) for k in keys}
    res.update(issues=str(o.get("issues", ""))[:400], ok=all(res.values()), model=llm_thinking.model, thinking=True, prompt_version=TWIST_STRICT_VERSION, perm=perm)
    return res


# =====================================================================================================================
# almost_enough (abstain format) -- generation + checks
# =====================================================================================================================
AE_SCHEMA = {"type": "object", "properties": {"feasible": {"type": "boolean"}, "missing_datum": {"type": "string"}, "question": {"type": "string"},
                                              "question_with_datum": {"type": "string"}, "correct_answer": {"type": "string"},
                                              "distractors": {"type": "array", "items": {"type": "string"}}}, "required": ["feasible"]}


def ae_cand_id(seed: Seed, model: str, sample: int = 0) -> str:
    ver = TWIST_AE_GEN_VERSION_R1 if (seed.link or {}).get("round", "r0") != "r0" else TWIST_AE_GEN_VERSION
    return "ae_" + sha1(f"{seed.seed_id}{ver}{model}{sample}")[:12]


AE_GIVEAWAY = re.compile(r"(nếu không biết|chưa biết|nếu biết|không đủ thông tin|không thể khẳng định|nếu không có thông tin|chưa rõ|không rõ|thiếu thông tin|không có thông tin)", re.I)
OPT_PREFIX = re.compile(r"^\s*\(?[A-Da-d][.):]\s+")


def generate_ae(seed: Seed, corpus: SpanCorpus, llm: LLM, *, sample: int = 0, temperature: float = 0.7) -> dict:
    s = corpus.get(seed.span_ids[0])
    r1 = (seed.link or {}).get("round", "r0") != "r0"
    ver = TWIST_AE_GEN_VERSION_R1 if r1 else TWIST_AE_GEN_VERSION
    cid = ae_cand_id(seed, llm.model, sample)
    base = {"cand_id": cid, "seed_id": seed.seed_id, "twist_type": "almost_enough", "kind": "almost_enough", "domain": s.domain, "grade": s.grade,
            "gold_span_ids": [s.span_id], "lesson_keys": [s.lesson_key], "book_id": s.book_id, "seed_span_ids": [s.span_id], "sample": sample,
            "generator": {"model": llm.model, "version": ver, "temperature": temperature}}
    prompt = _COMMON_AE_HEAD.format(subject=DOMAIN_TO_SUBJECT_VI.get(s.domain, s.domain), grade=s.grade) + AE_SPEC.format(
        blocks=span_block("S1", s, 1800), extra=AE_ADDENDUM_V11 if r1 else "")
    try:
        o, r = chat_json(llm, [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}], AE_SCHEMA, max_tokens=1200,
                         temperature=temperature, validate=lambda o: "feasible" in o)
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"}
    base["gen_raw"] = o
    if not _bool(o.get("feasible")):
        return {**base, "feasible": False, "failed": "infeasible"}
    q, qd, key = (str(o.get(k) or "").strip() for k in ("question", "question_with_datum", "correct_answer"))
    ds = [str((d.get("text") if isinstance(d, dict) else d) or "").strip() for d in o.get("distractors") or []][:3]
    texts = {norm(key)} | {norm(d) for d in ds}
    if not (q and qd and key and len(ds) == 3 and len(texts) == 4 and "" not in texts):
        return {**base, "feasible": False, "failed": "malformed"}
    if AE_GIVEAWAY.search(q) or any(OPT_PREFIX.match(x) for x in [key] + ds):
        return {**base, "feasible": False, "failed": "giveaway_or_option_prefix"}
    if bs._DEICTIC.search(q) or bs._DEICTIC.search(qd) or re.search(r"(đoạn trích|đoạn văn|văn bản trên|thông tin trên|theo sách|trong sách|\bS1\b)", q + qd, re.I):
        return {**base, "feasible": False, "failed": "meta_reference"}
    if len(norm(key)) >= 6 and (norm(key) in norm(q) or norm(key) in norm(qd)):
        return {**base, "feasible": False, "failed": "answer_in_question"}
    if jaccard(char_ngrams(q), char_ngrams(qd)) > 0.97:
        return {**base, "feasible": False, "failed": "identical_questions"}
    opts, ans = assemble_options(cid, key, ds)
    if not length_ok_twist(key, ds, opts, ans):
        return {**base, "feasible": False, "failed": "length_cue"}
    ov = stem_overlap(q, [s.text])
    return {**base, "feasible": True, "question": q, "question_with_datum": qd, "options": opts, "answer": ans, "correct_answer": key,
            "distractors": ds, "missing_datum": str(o.get("missing_datum", "")), "stem_overlap_with_span": round(ov, 4),
            "min_spans": 1, "hop_type": "single", "reasoning_steps": [], "evidence": {}}


_COMMON_AE_HEAD = "Môn: {subject}. Lớp: {grade}.\n"
