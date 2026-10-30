"""Fixed prompts for ViEduHop-Para. Their sha256 is stored in benchmarks/otter_hard/para/prompts/HASHES.json at freeze time."""

GEN_SYSTEM = "Bạn là chuyên gia ngôn ngữ và giáo viên Việt Nam, giỏi diễn đạt lại câu hỏi trắc nghiệm bằng từ ngữ khác mà không đổi nghĩa."

GEN_USER = """Hãy VIẾT LẠI câu hỏi trắc nghiệm sau (cả phần câu hỏi VÀ cả bốn phương án) bằng từ ngữ KHÁC, sao cho học sinh phải hiểu ý chứ không nhận ra câu chữ của sách giáo khoa.

CÂU HỎI GỐC
Câu hỏi: {question}
A. {A}
B. {B}
C. {C}
D. {D}
Đáp án đúng: {key}

ĐOẠN SÁCH GIÁO KHOA (nguồn của câu hỏi; chỉ để bạn biết cần TRÁNH chép từ ngữ nào):
{gold}

YÊU CẦU
1. Viết lại phần câu hỏi và từng phương án A, B, C, D bằng từ ngữ khác: dùng cách nói đời thường hoặc thuật ngữ tương đương, đổi cấu trúc câu (đảo thành phần, đổi chủ động/bị động, tách hoặc gộp ý, dùng từ đồng nghĩa).
2. Tuyệt đối KHÔNG chép nguyên cụm ba âm tiết liên tiếp nào từ đoạn sách giáo khoa ở trên (trừ tên riêng và số liệu). Thuật ngữ có từ ba âm tiết trở lên (ví dụ "giao thoa sóng", "nhiệt độ nóng chảy") cũng phải đổi: dùng tên gọi khác tương đương hoặc diễn tả bằng lời thường. Mục tiêu: ít nhất hai phần ba số từ trong bản mới KHÔNG xuất hiện trong đoạn sách. Không bắt đầu câu hỏi bằng "Theo sách giáo khoa" hay các cụm chép từ đoạn sách; nếu một phương án chỉ là số hoặc tên riêng thì giữ nguyên phương án đó.
3. GIỮ NGUYÊN: đáp án đúng vẫn là phương án {key}; thứ tự các phương án; mọi con số, đơn vị, tên riêng (người, địa danh, sự kiện, công thức, tên chất); dạng logic của câu hỏi (câu hỏi phủ định như "không", "ngoại trừ", "sai" vẫn là phủ định); vai trò của từng phương án nhiễu (phương án sai vẫn sai, vẫn sai theo cùng một lý do; không biến phương án sai thành đúng và ngược lại).
4. KHÔNG thêm thông tin nào không có trong câu hỏi gốc hoặc đoạn sách; không bỏ thông tin quan trọng; không để lộ đáp án.
5. Không đánh dấu hay giải thích gì thêm.

Chỉ trả lời bằng MỘT đối tượng JSON hợp lệ có dạng:
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}}}"""

JUDGE_SYSTEM = "Bạn là giám khảo nghiêm khắc kiểm tra xem hai phiên bản của một câu hỏi trắc nghiệm có cùng nghĩa hay không. Chỉ trả lời bằng JSON."

JUDGE_USER = """Dưới đây là CÂU HỎI GỐC và BẢN DIỄN ĐẠT LẠI của cùng một câu trắc nghiệm, cùng với đoạn sách giáo khoa là nguồn. Đáp án đúng ở cả hai bản là phương án {key}.

ĐOẠN SÁCH:
{gold}

--- CÂU GỐC ---
{oq}
A. {oA}
B. {oB}
C. {oC}
D. {oD}

--- BẢN DIỄN ĐẠT LẠI ---
{pq}
A. {pA}
B. {pB}
C. {pC}
D. {pD}

Hãy kiểm tra từng điểm sau (true/false), xét CẢ HAI CHIỀU suy ra (gốc suy ra bản mới và bản mới suy ra gốc):
- "stem_equivalent": phần câu hỏi của bản mới hỏi đúng cùng một điều với câu hỏi gốc (cùng đối tượng, cùng phạm vi, cùng dạng khẳng định/phủ định).
- "key_equivalent": phương án {key} của bản mới nêu đúng cùng một nội dung/sự kiện với phương án {key} của câu gốc.
- "distractor_same_claim": với mỗi phương án sai (các chữ cái khác {key}), phương án của bản mới nêu cùng một nhận định với phương án sai tương ứng của câu gốc (không biến thành nhận định khác).
- "distractors_still_wrong": mọi phương án sai của bản mới vẫn SAI so với câu hỏi mới (dựa vào đoạn sách), và phương án {key} vẫn là đáp án đúng duy nhất.
- "adds_information": bản mới có thêm thông tin/sự kiện/điều kiện không có trong câu gốc hoặc làm lộ đáp án (true nếu có).

Trả lời bằng MỘT đối tượng JSON:
{{"stem_equivalent": true/false, "key_equivalent": true/false, "distractor_same_claim": {{"<chữ cái sai 1>": true/false, "<chữ cái sai 2>": true/false, "<chữ cái sai 3>": true/false}}, "distractors_still_wrong": true/false, "adds_information": true/false, "reason": "một câu ngắn"}}"""

ANS_SYSTEM = "Bạn là học sinh giỏi. Chỉ dùng đoạn sách giáo khoa được cung cấp để trả lời câu hỏi trắc nghiệm."

ANS_USER = """ĐOẠN SÁCH GIÁO KHOA:
{gold}

CÂU HỎI: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Chọn đúng MỘT phương án dựa trên đoạn sách. Chỉ trả lời bằng MỘT chữ cái (A, B, C hoặc D)."""

CB_SYSTEM = "Bạn là học sinh giỏi môn học này."

CB_USER = """CÂU HỎI: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Chọn đúng MỘT phương án. Chỉ trả lời bằng MỘT chữ cái (A, B, C hoặc D)."""
