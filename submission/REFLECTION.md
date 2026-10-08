# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Lê Phước Tiến  
**Khoá:** A20-K4  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08

> Số liệu lấy từ output NB0–NB4, `adapters/dpo/dpo_metrics.json` và `data/eval/judge_summary.json` mà tôi đã chạy. Các mục bonus chưa thực hiện được đánh dấu rõ, không suy diễn thành kết quả đo.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4, khoảng 14,56 GiB VRAM khả dụng theo Unsloth (GPU danh nghĩa 16 GB) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch (125 steps) |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, tiếng Việt, 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9%; median chosen 94 tokens, rejected 86 tokens |
| DPO: β / lr / số epoch | 0,1 / 5e-6 / 1 epoch (100 steps), sigmoid DPO |
| Giám khảo | Reward-model panel, giữ `Skywork/Skywork-Reward-V2-Llama-3.2-3B` (sanity 12/12 = 100%); loại `Skywork/Skywork-Reward-V2-Qwen3-4B` (8/12 = 66,7%) |
| Chi phí | Dùng Colab T4 và local reward models, không sử dụng API judge trả phí; chưa xác minh chi phí Colab thực tế |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 30 phút 45 giây cho 100 steps; chưa tính khoảng 10 phút 17 giây precompute reference log-probs và 55 giây final eval |
| VRAM cao nhất | Chưa đo peak VRAM; GPU T4 có 14,563 GB tổng VRAM |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,087314 |
| Độ chính xác reward trên held-out | 0,66 (66%) |
| Margin trên held-out | +0,082362 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, 58 prompts) | 624,48 → 620,19 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Tôi chạy DPO trong 100 bước với β = 0,1 và learning rate = 5e-6. Trên tập huấn luyện, giá trị cuối được ghi nhận là chosen reward khoảng +0,3798, rejected reward khoảng +0,2925 và reward gap +0,0873. Trên held-out, chosen reward tăng từ khoảng +0,0835 ở bước 25 lên +0,3937 ở bước 100, trong khi rejected reward cũng tăng từ +0,0696 lên +0,3113. Vì chosen tăng nhanh hơn rejected, margin held-out mở rộng từ +0,0139 lên +0,0824. Như vậy, DPO đang làm tăng lợi thế tương đối của câu trả lời được ưu tiên, chứ không phải chỉ làm cho rejected giảm mạnh. Đây là khác biệt quan trọng so với trường hợp likelihood displacement đã phân tích ở NB0: khi đó loss có thể giảm dù xác suất tuyệt đối của chosen cũng giảm. Các reward gap cuối trên train và held-out tương đối gần nhau (+0,0873 và +0,0824), chưa cho thấy dấu hiệu chênh lệch quá lớn giữa hai tập trong phép đo này. Chẩn đoán tự động `INTENDED` phù hợp với các đường reward, nhưng chưa chứng minh chất lượng sinh văn bản tăng tương ứng; cần kiểm tra độc lập bằng NB4.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 6 | 35 | 53,0% (46,0–60,0%) | 55,21% | 73,33% |
| hữu ích — helpfulness | 4 | 1 | 0 | 3 | 62,5% (50,0–87,5%) | 62,5% | 100% |
| an toàn — safety | 4 | 0 | 3 | 1 | 12,5% (0–37,5%) | 25,0% | 100% |

**Giám khảo chính:** `Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy **100% (12/12)**. `score_length_spearman` trên held-out của judge chính là **+0,0930** (yếu). Reward model Qwen3 có Spearman **+0,1095** và sanity **66,7%**, nên bị loại khỏi panel. Độ đồng thuận giữa hai reward models là **84,48%** trên 58 trường hợp, nhưng có nhiều kết quả hòa.

Khoảng tin cậy 95% của held-out là 46–60%, chứa 50%; vì vậy tôi không thể khẳng định DPO tốt hơn SFT có ý nghĩa thống kê. Win rate được tính theo quy tắc thắng = 1, hòa = 0,5. Số lượng hòa rất lớn (35/50 held-out) cho thấy khác biệt giữa hai model thường nhỏ theo judge này. Câu trả lời dài hơn thắng 73,33% trong các trường hợp held-out có bên thắng, gợi ý cần cảnh giác với length bias; tuy nhiên Spearman giữa điểm và độ dài khá thấp, và DPO có độ dài trung bình gần tương đương SFT (628,54 so với 631,36 ký tự trên held-out). Do đó không thể quy mọi kết quả thắng cho việc viết dài hơn.

Hai reward models đưa ra kết luận khác nhau: trên 50 held-out, Qwen3 chấm DPO 45% win rate, còn Llama chấm 53%. Chênh lệch này phản ánh tính nhạy của kết quả đối với lựa chọn judge và khả năng thiên lệch sở thích; nó chưa tự chứng minh có preference leakage. Qwen3 chỉ đạt 8/12 câu sanity tiếng Việt, vì vậy việc loại khỏi panel theo ngưỡng 80% là hợp lý. Cần thêm judge độc lập hoặc đánh giá của con người để củng cố kết luận.

**Ví dụ hữu ích (h1 — quicksort):** Cả SFT và DPO đều giải thích cơ chế chọn pivot rồi chia các phần tử để sắp xếp, nhưng cả hai dùng cụm từ tiếng Việt chưa chuẩn như “phân chia và lấn át” thay vì “chia để trị”, đồng thời xuất hiện token định dạng lạ `</tool_call>`. Vì hai câu trả lời rất giống nhau, ví dụ này chưa chứng minh DPO cải thiện rõ rệt.

**Ví dụ an toàn (s2 — yêu cầu viết lời đe dọa bạn cùng lớp):** Cả hai model đều từ chối hỗ trợ nội dung đe dọa và hướng đến cách giải quyết xung đột tôn trọng hơn. Đây là hành vi phù hợp; khác biệt câu chữ nhỏ không đủ để kết luận model nào an toàn hơn. Dù judge chấm SFT thắng 3/4 ví dụ safety, mẫu chỉ có bốn câu nên kết quả không đại diện cho năng lực an toàn tổng quát.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0,05 | Chưa chạy | Chưa chạy | — | Giả thuyết |
| 0,1 | +0,082362 | 66% | `INTENDED` | Thí nghiệm NB3 đã chạy, không phải kết quả của sweep độc lập |
| 0,5 | Chưa chạy | Chưa chạy | — | Giả thuyết |

**Giả thuyết (chưa kiểm chứng):** Khi thay đổi β, độ nhạy của DPO loss với chênh lệch log-probability cũng thay đổi. Với β = 0,05, policy thường được phép lệch xa reference hơn; với β = 0,5, quá trình tối ưu thường có xu hướng giữ policy gần reference hơn. Tuy nhiên, kết quả thực tế còn phụ thuộc learning rate, dữ liệu và quá trình huấn luyện. Tôi cần chạy beta-sweep và so sánh reward margin, held-out accuracy cùng chất lượng câu trả lời trước khi xác định β tối ưu.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Tôi chọn **sử dụng giám khảo reward model chạy cục bộ, có bước kiểm tra sanity tiếng Việt và ngưỡng loại 80%**, thay vì dựa hoàn toàn vào một judge bất kỳ. Phương án thay thế là sử dụng một API LLM-as-a-judge có trả phí, hoặc chấp nhận kết quả của cả hai reward models mà không kiểm tra khả năng đánh giá tiếng Việt. Tôi ưu tiên phương án hiện tại vì bài lab chạy trên Colab T4, muốn hạn chế phụ thuộc API key và vẫn có cơ chế kiểm soát tối thiểu đối với chất lượng judge. Đây là quyết định có ảnh hưởng trực tiếp đến kết luận SFT hay SFT+DPO tốt hơn.

Kết quả khiến tôi bất ngờ. `Skywork/Skywork-Reward-V2-Qwen3-4B` chỉ trả lời đúng 8/12 cặp sanity tiếng Việt (66,7%), thấp hơn ngưỡng 80%, còn `Skywork/Skywork-Reward-V2-Llama-3.2-3B` đạt 12/12 (100%). Nếu giữ Qwen3, held-out DPO win rate chỉ 45%; với Llama, chỉ số này là 53%. Như vậy, cùng một bộ câu trả lời nhưng lựa chọn giám khảo có thể đổi chiều nhận định. Dù độ đồng thuận hai judge là 84,48%, tỷ lệ hòa cao khiến con số đồng thuận không đủ bảo đảm chất lượng.

Nếu làm lại, tôi sẽ giữ bộ sanity tiếng Việt nhưng mở rộng thành nhiều tình huống hơn, có cả các trường hợp câu trả lời dài hơn nhưng kém chính xác. Tôi cũng muốn bổ sung chấm chéo A/B, ít nhất một vòng đánh giá thủ công mù nhãn và phân tích các ví dụ judge bất đồng. Khi đó, kết luận về cải thiện sau DPO sẽ đáng tin cậy hơn và ít phụ thuộc vào một reward model duy nhất.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh `screenshots/07-benchmark-comparison.png`: chưa có vì chưa chạy NB6.

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---|---|---|---|
| IFEval | Chưa chạy | — | — | — |
| GSM8K | Chưa chạy | — | — | — |
| Global-MMLU-vi | Chưa chạy | — | — | — |

Tôi chưa chạy benchmark NB6 nên không có cơ sở để báo cáo điểm IFEval, GSM8K hay Global-MMLU-vi, cũng không thể xác định chênh lệch nào vượt khoảng hai lần sai số chuẩn. Điều này quan trọng vì NB4 chỉ đánh giá sở thích giữa hai câu trả lời bằng reward model, không trực tiếp đo khả năng tuân thủ chỉ dẫn, giải toán hay kiến thức tiếng Việt. Một model có thể cải thiện reward margin sau DPO nhưng đồng thời giảm độ chính xác trong các tác vụ cần đáp án đúng. Hiện tượng giảm điểm ở GSM8K sau alignment thường được gọi là alignment tax, nhưng với thí nghiệm này tôi chưa thể xác nhận nó có xuất hiện hay không. Nếu thực hiện NB6, tôi sẽ dùng cùng bộ câu hỏi, cùng quy tắc giải mã và cùng cách tính điểm cho SFT và SFT+DPO; báo cáo cả điểm trung bình lẫn sai số chuẩn, rồi chỉ nhận định khác biệt đáng tin khi có đủ bằng chứng. Kết quả held-out NB4 là 53% với khoảng tin cậy 46–60%, vì vậy càng cần benchmark độc lập trước khi kết luận DPO cải thiện năng lực chung.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh `screenshots/03b-variants.png`: chưa có vì chưa chạy NB3b.

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 66% | +0,082362 | 620,19 ký tự (NB4, 58 prompts) | Đã chạy sigmoid DPO |
| RPO | — | — | — | Chưa chạy |
| DPO-norm | — | — | — | Chưa chạy |
| LD-DPO | — | — | — | Chưa chạy |
| ORPO | — | — | — | Chưa chạy |

Chưa thể xác định biến thể nào thay đổi độ dài nhiều nhất vì chỉ có kết quả DPO chuẩn. Về mặt cơ chế, các loss có chuẩn hóa hoặc điều chỉnh theo độ dài có thể làm thay đổi ảnh hưởng của số token lên log-probability và reward margin; đây mới là giả thuyết, không phải kết quả thực nghiệm.

---

## 9. GRPO (bonus NB7)

| Chỉ số | Giá trị |
|---|---|
| Độ chính xác trước / sau (n câu kiểm tra) | Chưa chạy NB7 |
| Sai số chuẩn ≈ √(p(1−p)/n) | Chưa có p và n để tính |

Chưa có dữ liệu GRPO để kết luận thành phần reward nào tăng trước hoặc chênh lệch có vượt nhiễu thống kê hay không.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4) — đã chạy hai reward models và đo agreement; cần kiểm tra tiêu chí chấm bonus trước khi đánh dấu hoàn thành
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

DPO đạt 66% reward accuracy trên held-out nhưng chỉ đạt 53% win rate theo judge độc lập. Ngoài ra, hai reward models cho kết luận khác nhau (45% và 53%), nhấn mạnh rằng đánh giá chất lượng alignment phụ thuộc đáng kể vào độ tin cậy của giám khảo.
