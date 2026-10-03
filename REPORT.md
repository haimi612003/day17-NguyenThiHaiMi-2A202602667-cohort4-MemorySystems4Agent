# Báo cáo Day 17: Memory Systems for AI Agent

Sinh viên: Nguyễn Thị Hải Mi · 2A202602667 · Cohort 4

Tài liệu này là phần phân tích kết quả (Guide bước 8) và giải thích bonus (Guide bước 9). Mọi con số bên dưới lấy từ `python src/benchmark.py` và `python src/ablation.py` ở chế độ offline, lặp lại được 100%.

## 1. Kết quả chính

Cấu hình: `compact_threshold_tokens = 800`, `compact_keep_messages = 4`, `profile_confidence_threshold = 0.7`.

### Standard Benchmark (`data/conversations.json`, 10 hội thoại, 101 lượt)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 2311 | 13769 | 0.00 | 0.40 | 0 | 0 |
| Advanced | 2419 | 26454 | 1.00 | 1.00 | 923 | 0 |

So với Baseline, Advanced có prompt tokens **+92.1%**, agent tokens **+4.7%** và recall **+1.00**.

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 1 hội thoại, 16 lượt rất dài)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 2549 | 21593 | 0.00 | 0.40 | 0 | 0 |
| Advanced | 3260 | 13461 | 1.00 | 1.00 | 628 | 4 |

So với Baseline, Advanced có prompt tokens **−37.7%**, agent tokens **+27.9%**, tổng usage (prompt + agent) **−30.7%** và recall **+1.00**.

Prompt tokens theo từng lượt trong cùng một thread:

| Lượt | Baseline | Advanced | Advanced tiết kiệm |
|-----:|---------:|---------:|-------------------:|
| 2  | 375  | 469 | −25% |
| 4  | 686  | 780 | −14% |
| 6  | 1005 | 650 | +35% |
| 8  | 1310 | 573 | +56% |
| 10 | 1548 | 846 | +45% |
| 12 | 1846 | 767 | +58% |
| 14 | 2121 | 592 | +72% |
| 16 | 2398 | 869 | +64% |

Prompt của Baseline tăng tuyến tính theo số lượt, nên tổng chi phí của cả hội thoại tăng theo bình phương số lượt. Prompt của Advanced dao động quanh ngưỡng compact (khoảng 600–870 token) bất kể hội thoại dài bao nhiêu. Điểm hòa vốn rơi vào khoảng lượt thứ 5.

## 2. Kiến trúc: tách bạch ba lớp memory

| Lớp | Nằm ở đâu | Sống bao lâu | Chứa gì |
|---|---|---|---|
| **Short-term** | `CompactMemoryManager.state[thread]["messages"]` (Advanced), `SessionState.messages` (Baseline) | Một thread | Nguyên văn các message gần nhất |
| **Persistent** | `state/profiles/<user>/User.md` qua `UserProfileStore` | Vĩnh viễn, qua mọi thread và process | Fact ổn định có cấu trúc: tên, nơi ở, nghề, đồ uống, món ăn, thú cưng, sở thích, style |
| **Compact** | `CompactMemoryManager.state[thread]["summary"]` | Một thread | Tóm tắt có giới hạn của phần hội thoại cũ đã bị cắt khỏi short-term |

Prompt của Advanced ở mỗi lượt gồm: system prompt, bản rút gọn của `User.md`, summary và các message gần nhất.

Prompt của Baseline gồm: system prompt và **toàn bộ** thread.

Các file được tách lớp như sau:
- `model_provider.py`: khởi tạo model cho 6 provider.
- `config.py`: cấu hình chung.
- `memory_store.py`: trích xuất fact, `User.md` và compact memory.
- `offline_responder.py`: sinh câu trả lời, dùng chung cho cả hai agent để so sánh công bằng.
- `agent_baseline.py`, `agent_advanced.py`: hai agent.
- `benchmark.py`, `ablation.py`, `test_agents.py`: đo đạc và kiểm thử.

Điểm thiết kế quan trọng: hai agent dùng **cùng một hàm sinh câu trả lời** và **cùng một hàm ước lượng token**. Khác biệt duy nhất là nguồn fact (thread hiện tại hay `User.md`) và cách dựng prompt. Nhờ vậy, chênh lệch trong benchmark đến từ memory chứ không đến từ cách viết câu trả lời.

## 3. Cách đo

- **Agent tokens only**: token của message người dùng, cộng câu trả lời của agent, cộng token mà bộ tóm tắt sinh ra khi compact. Compact không miễn phí nên chi phí này được tính vào.
- **Prompt tokens processed**: tổng ngữ cảnh gửi vào model qua mọi lượt, tính trên tất cả thread kể cả thread hỏi recall. Với Advanced, con số này **cộng cả phần bộ tóm tắt phải đọc lại** (summary cũ cộng message cũ) mỗi lần compact.
- **Cross-session recall**: mỗi `recall_question` được hỏi trong một **thread mới**. Hỏi đủ mọi fact mong đợi thì được 1, đủ một phần được 0.5, không có fact nào được 0. Phép so khớp không phân biệt hoa thường và đã chuẩn hóa Unicode NFC.
- **Response quality** (offline, thang 0–1): 60% độ phủ fact, 20% độ ngắn gọn, 20% cấu trúc và độ trung thực. Câu trả lời "không có thông tin" được điểm cấu trúc vì thừa nhận lỗ hổng tốt hơn bịa ra. Khi chạy `--live`, điểm này do model judge chấm (`JUDGE_PROVIDER` / `JUDGE_MODEL`).
- **Memory growth**: kích thước `User.md` sau khi chạy trừ kích thước trước khi chạy. Benchmark luôn reset `User.md` của user benchmark để kết quả lặp lại được.

## 4. Phân tích

### 4.1. Vì sao Advanced recall tốt hơn

Baseline chỉ có memory theo `thread_id`. Câu hỏi recall luôn nằm ở thread mới nên Baseline không còn gì để nhớ: recall bằng 0 ở cả hai bộ dữ liệu. Đây là hành vi đúng, vì Baseline không được phép giả vờ có long-term memory. Test `test_baseline_remembers_within_same_thread` xác nhận Baseline vẫn nhớ trong cùng thread.

Advanced ghi fact vào `User.md`, và file này sống qua thread lẫn process (`test_cross_session_recall` tạo hẳn một instance agent mới). Recall đạt 1.0 kể cả với các bẫy:
- **Correction**: nơi ở đổi từ Đà Nẵng sang Huế, nghề đổi từ backend sang MLOps; ở bộ stress, nơi ở đổi từ Huế sang Đà Nẵng.
- **Nhiễu**: "product manager chỉ là câu đùa", "Hà Nội chỉ là nơi đi họp", "nếu sau này mình nhắc lại Đà Nẵng…".

### 4.2. Vì sao Advanced tốn hơn ở hội thoại ngắn

Ở bộ Standard, Advanced tốn **gấp khoảng 1.9 lần prompt tokens**. Có ba lý do:
1. Mỗi lượt đều phải mang theo bản rút gọn của `User.md` (khoảng 60–90 token) và một system prompt dài hơn. Ở lượt đầu của một thread, toàn bộ prompt của Baseline chỉ khoảng 40 token.
2. Mỗi hội thoại chỉ khoảng 10 lượt ngắn, chưa bao giờ chạm ngưỡng 800 token, nên **compact không kích hoạt lần nào** (Compactions = 0). Advanced chịu chi phí của persistent memory nhưng chưa nhận lợi ích của compact.
3. Advanced trả lời dài hơn một chút vì xác nhận những gì đã lưu ("Đã lưu vào User.md: …"), nên agent tokens cao hơn 4.7%.

Kết luận: với hội thoại ngắn, thứ ta trả thêm token để mua là **recall qua phiên**, không phải tiết kiệm. Đổi gần gấp đôi prompt lấy recall tăng từ 0 lên 1 là đáng nếu sản phẩm cần cá nhân hóa, và không đáng nếu chỉ là hỏi đáp một lần. Test `test_short_conversation_advanced_can_cost_more` khóa trade-off này lại để nó không bị "sửa" nhầm.

### 4.3. Vì sao compact chủ yếu tối ưu *prompt tokens processed*

Compact không làm người dùng nói ít đi, cũng không làm agent trả lời ngắn đi. Thứ nó thay đổi là **lượng ngữ cảnh cũ phải gửi lại ở mỗi lượt**:
- Baseline gửi lại toàn bộ lịch sử ở mỗi lượt, nên lượt thứ *n* tốn khoảng *n* lần độ dài một message, và tổng cả hội thoại tăng theo bình phương.
- Advanced giữ ngữ cảnh trong khoảng ngưỡng cộng `User.md`, nên lượt thứ *n* tốn chi phí gần như hằng số, và tổng cả hội thoại tăng tuyến tính.

Vì vậy ở bộ Stress, prompt tokens giảm 37.7%, trong khi **agent tokens lại tăng 27.9%**. Phần tăng này gồm token bộ tóm tắt sinh ra cộng các câu xác nhận đã lưu. Tổng usage vẫn giảm 30.7% vì prompt chiếm phần lớn chi phí. Hội thoại càng dài, khoảng cách càng lớn: ở lượt 14, Advanced chỉ còn khoảng 28% prompt của Baseline.

### 4.4. Phát hiện khi tinh chỉnh: compaction thrashing

Bản đầu tiên của `CompactMemoryManager` cho phép summary dài tới 8 dòng. Với ngưỡng nhỏ (300 token), summary tự nó đã gần chạm ngưỡng, nên **mỗi lần append lại kích hoạt compact**: 14 lượt sinh ra 15 lần compact. Vì mỗi lần compact phải đọc lại phần cũ, Advanced còn **tốn hơn** Baseline (8175 so với 8113 prompt tokens).

Cách sửa là giới hạn summary ở mức tối đa 1/3 ngưỡng. Sau khi sửa, cùng kịch bản chỉ còn 5 lần compact và 4665 prompt tokens, tiết kiệm 43% so với Baseline. Test `test_compaction_does_not_thrash` khóa lỗi này lại.

Bài học: compact chỉ có lợi khi phần còn lại sau khi nén **nhỏ hơn hẳn** ngưỡng. Ngưỡng và kích thước summary phải được chỉnh cùng nhau.

### 4.5. Memory file tăng trưởng và rủi ro

- Sau 10 phiên, `User.md` chỉ khoảng 923 byte. File tăng theo **số fact khác nhau**, không theo số lượt chat, vì fact trùng chỉ tăng bộ đếm `mentions`. Danh sách bị giới hạn 6 phần tử, lịch sử đính chính giới hạn 5 dòng. Đây là chặn trên có chủ đích.
- **Rủi ro lưu sai fact**: một fact sai trong `User.md` sẽ sai ở *mọi* phiên sau, tệ hơn quên. Ablation ở mục 5 cho thấy chỉ cần bỏ một guardrail là `User.md` đã ghi "nghề: backend engineer" sai, hoặc ghi đồ uống yêu thích là "gì".
- **Fact cũ (stale)**: nếu người dùng không bao giờ đính chính, agent giữ fact cũ mãi. Decay giảm ưu tiên của item ít được nhắc nhưng không xóa fact đơn trị.
- **Quyền riêng tư**: `User.md` là dữ liệu cá nhân nằm trên đĩa dưới dạng plaintext. Môi trường production cần mã hóa, cơ chế xóa theo yêu cầu, và không commit thư mục `state/` (đã có trong `.gitignore`).
- **Prompt injection vào memory**: ở chế độ live, LLM có tool `save_user_fact`. Người dùng có thể lừa model ghi nội dung độc hại vào `User.md`, và nội dung đó sẽ được tiêm vào mọi prompt sau. Giảm thiểu bằng cách giới hạn tên field (chỉ 8 field), đưa lệnh ghi của tool qua cùng cổng conflict handling, và không bao giờ ghi nguyên văn instruction.

## 5. Bonus: vấn đề, cải thiện và rủi ro

Ablation (`python src/ablation.py`) tắt lần lượt từng guardrail của Advanced. Bộ **Probe** là 12 lượt tổng hợp nhỏ, dùng để kích hoạt các guardrail mà hai bộ dữ liệu gốc không chạm tới: một lần nhắc đồ uống yếu, và 9 sở thích chỉ nhắc một lần.

| Biến thể | Standard recall | Standard: nơi ở / nghề | Stress recall | Probe recall | Probe `User.md` (bytes) | Probe: đồ uống |
|---|---:|---|---:|---:|---:|---|
| **Đủ guardrail** | **1.00** | Huế / MLOps engineer | **1.00** | **1.00** | **320** | trà sữa |
| Không conflict handling | 0.64 | Đà Nẵng / backend engineer | 0.67 | 1.00 | 320 | trà sữa |
| Không lọc nhiễu/phủ định | 0.86 | Huế / backend engineer | 1.00 | 1.00 | 320 | trà sữa |
| Không lọc câu hỏi | 0.68 | Huế / MLOps engineer | 1.00 | 0.50 | 427 | "gì" |
| Không confidence threshold | 1.00 | Huế / MLOps engineer | 1.00 | 0.50 | 445 | nước dừa |
| Không decay / không giới hạn danh sách | 1.00 | Huế / MLOps engineer | 1.00 | 1.00 | 378 | trà sữa |

### 5.1. Conflict handling

- **Vấn đề**: người dùng đính chính ("giờ mình ở Huế chứ không còn ở Đà Nẵng"), nhưng memory vẫn giữ giá trị cũ, hoặc giữ cả hai.
- **Cách làm**: các field đơn trị (tên, nơi ở, nghề, đồ uống, món ăn, thú cưng) theo nguyên tắc *giá trị đủ tin cậy mới nhất thì thắng*. Giá trị cũ được chuyển xuống mục `## Corrections (giá trị cũ, không dùng làm hiện tại)`, giới hạn 5 dòng, nhưng không bao giờ được dùng làm giá trị hiện tại. Từ khóa đính chính ("đính chính", "thực ra", "chuyển sang", "giờ") cộng thêm điểm tin cậy.
- **Cải thiện**: đây là guardrail quan trọng nhất. Tắt nó đi, recall Standard giảm từ 1.00 xuống **0.64** và Stress giảm từ 1.00 xuống **0.67**.
- **Rủi ro**: "mới nhất thì thắng" có thể ghi đè fact đúng bằng một câu nói nhất thời. Rủi ro này được giảm nhờ confidence threshold và bộ lọc nhiễu. Lịch sử đính chính làm `User.md` dài thêm vài chục byte.

### 5.2. Confidence threshold

- **Vấn đề**: không phải câu nào nhắc tới một giá trị cũng là fact ổn định. "Trưa nay mình uống nước dừa" không có nghĩa nước dừa là đồ uống yêu thích.
- **Cách làm**: mỗi fact được trích xuất kèm một điểm tin cậy. Câu nói rõ ("đồ uống yêu thích là …") được 0.95, câu chỉ kể thói quen ("uống …") được 0.55. Chỉ fact đạt ≥ 0.7 mới được ghi. Fact yếu **không được tạo mới hoặc ghi đè**, nhưng **được phép củng cố** fact đã có: "Mình vẫn uống cà phê sữa đá" làm tăng `mentions` của cà phê sữa đá. Các fact bị loại được ghi vào `report.rejected_low_confidence` để tiện debug.
- **Cải thiện**: trên bộ Probe, tắt threshold làm đồ uống bị ghi đè thành "nước dừa", recall giảm từ 1.00 xuống **0.50**, và `User.md` phình thêm 39%. Hai bộ dữ liệu gốc không có trường hợp nào như vậy nên ở đó không thấy khác biệt. Tôi báo cáo trung thực điểm này thay vì nói threshold cải thiện mọi thứ.
- **Rủi ro**: ngưỡng quá cao sẽ bỏ sót fact thật được nói gián tiếp, tức là tăng false negative. Điểm tin cậy hiện do luật đặt tay nên chưa được hiệu chỉnh trên dữ liệu thật.

### 5.3. Entity extraction có cấu trúc

- **Vấn đề**: lưu nguyên văn câu chat vào memory làm file phình to, khó cập nhật và khó trả lời chính xác.
- **Cách làm**: `extract_profile_candidates()` tách câu thành mệnh đề, theo các dấu `:` `;` `,` và các từ "chứ", "nhưng", "dù". Sau đó trích xuất:
  - tên: lấy các từ viết hoa sau "tên là";
  - nơi ở: dùng danh sách địa danh, có mẫu "từ A sang B";
  - nghề: dùng mẫu "… engineer/manager/…";
  - đồ uống, món ăn, thú cưng ("corgi tên Bơ");
  - sở thích kỹ thuật: dùng danh sách thuật ngữ;
  - style: chuẩn hóa thành tag như "ngắn gọn", "3 bullet", "có ví dụ thực tế", "so sánh trade-off".

  Các mệnh đề phủ định, giả định hoặc đùa bị bỏ qua ("không còn", "chỉ là", "đùa", "nếu", "lúc đầu", "họp"). Câu hỏi không bao giờ sinh ra fact. Ngữ cảnh tạm thời ("tuần này", "hôm nay", "tạm thời") không được lưu thành sở thích.
- **Cải thiện**: tắt bộ lọc phủ định, recall Standard giảm còn **0.86** và nghề bị ghi sai thành "backend engineer" (từ câu "đừng nói backend engineer nữa"). Tắt bộ lọc câu hỏi, recall giảm còn **0.68** vì các câu kiểu "đồ uống yêu thích của mình là gì?" bị lưu thành fact. Nhờ có cấu trúc, `User.md` của 10 phiên chỉ khoảng 900 byte, và câu trả lời có thể tuân theo style của người dùng (ví dụ gộp nội dung cho vừa "3 bullet").
- **Rủi ro**: dùng regex và danh sách cố định nên dễ vỡ khi người dùng diễn đạt khác đi. Ví dụ: địa danh không có trong danh sách, nghề không có đuôi "engineer", câu phủ định kiểu khác. Ở production nên dùng LLM extraction có schema (function calling) và giữ bộ luật này làm lớp kiểm tra chéo.

### 5.4. Memory decay

- **Vấn đề**: danh sách sở thích và style chỉ tăng chứ không giảm. Thứ nhắc một lần từ lâu vẫn chiếm chỗ trong prompt mãi.
- **Cách làm**: mỗi item có `mentions` (số lần nhắc) và `seen` (lượt gần nhất được nhắc). Điểm ưu tiên tính bằng `mentions × decay_rate^(lượt hiện tại − seen)`, với `decay_rate = 0.9`. Item được sắp xếp theo điểm này (khi đưa vào prompt và khi trả lời), và danh sách bị cắt còn `max_list_items = 6` bằng cách loại item điểm thấp nhất.
- **Cải thiện**: tắt decay, `User.md` của bộ Probe phình từ 320 lên **378 byte (+18%)**, đồng thời prompt mang theo nhiều sở thích nhắc đúng một lần. Recall không đổi vì sở thích được nhắc nhiều ("Python", 2 lần) vẫn đứng đầu.
- **Rủi ro**: một sở thích thật nhưng hiếm khi được nhắc có thể bị đẩy ra. Decay theo số lượt chứ không theo thời gian thực, nên người dùng chat nhiều sẽ "quên" nhanh hơn. Hệ số 0.9 và 6 item hiện là giá trị chọn tay.

## 6. Giới hạn và hướng phát triển

- **Chế độ offline là deterministic**: câu trả lời được ghép từ fact, không phải văn tự nhiên. Chế độ `--live` dùng LangChain `create_agent` với tool `read_user_profile` / `save_user_fact` (Advanced), và `create_agent` + `InMemorySaver` (Baseline). Phần này đã được kiểm thử bằng model giả (`test_live_mode_wiring_with_fake_model`), nhưng **chưa được benchmark với API key thật**.
- **`estimate_tokens` dùng ước lượng ký tự/4**: với tiếng Việt có dấu, cách này đếm thiếu so với tokenizer thật. Cả hai agent dùng chung hàm nên so sánh vẫn công bằng, nhưng con số tuyệt đối chỉ mang tính tương đối.
- **Summary là extractive**: mỗi message cũ giữ lại câu có nhiều thực thể nhất. Một summary do LLM viết sẽ giữ abstraction tốt hơn (ví dụ "readiness, externality, uncertainty, efficiency" trong bộ stress), đổi lại tốn thêm token output.
- **Quality offline là heuristic**: dùng `--live` để model judge chấm thay.

## 7. Cách chạy

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest

python src/benchmark.py            # 2 bảng benchmark (offline, lặp lại được)
python src/benchmark.py --verbose  # in thêm từng câu trả lời recall
python src/ablation.py             # tắt từng bonus để đo tác động
pytest src/test_agents.py -v       # 22 test

cp .env.example .env               # điền API key rồi:
python src/benchmark.py --live     # chạy với LLM thật + model judge
```
