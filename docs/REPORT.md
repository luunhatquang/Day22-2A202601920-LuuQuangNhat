# Báo cáo thí nghiệm Preference Alignment

## 1. Phân tích và làm sạch dữ liệu

### Dữ liệu và kiểm tra đầu vào

- **Nguồn dữ liệu**: `data/sample_preferences.jsonl`, gồm các cặp hỏi–đáp về machine learning.
- **Số lượng nạp thành công**: **24** preference pairs.
- **Schema**: mỗi dòng có `prompt`, `chosen`, `rejected` và `metadata` tùy chọn.
- **Kiểm tra đã thực hiện**: JSON lỗi và schema lỗi trả về số dòng; prompt trùng bị chặn sau khi chuẩn hóa whitespace/case; `chosen` và `rejected` không được trùng hoặc chỉ khác dấu câu; PII có thể được bật qua `check_pii=True`.

Lệnh đã chạy:

```bash
PYTHONPATH=src python -m preference_lab.cli validate data/sample_preferences.jsonl
```

Kết quả: `Loaded 24 preference examples`.

### Chiến lược chia tập

Tập dữ liệu được nhóm theo prompt đã chuẩn hóa, sau đó shuffle xác định với `seed: 42`. Cấu hình `validation_ratio: 0.2` tạo **19** ví dụ train và **5** ví dụ validation. Một prompt chỉ thuộc một split nên không rò rỉ dữ liệu giữa train và evaluation.

## 2. Cài đặt phương pháp

### Mục tiêu tối ưu

Thí nghiệm chạy **DPO** ở chế độ CPU mock trainer (`method: dpo`, `beta: 0.1`, `batch_size: 2`, `epochs: 3`). DPO tối ưu chênh lệch log-ratio giữa `chosen` và `rejected` so với reference policy; không cần reward model riêng.

Hàm DPO dùng biểu thức ổn định số:

$$-\log \sigma(z) = \operatorname{logaddexp}(0, -z).$$

Repository cũng có ORPO: log-probability được clip nhỏ hơn hoặc bằng `-1e-7` trước khi tính log-odds để tránh `log(0)`.

### Đánh giá CPU

Không có model weights trong bài nộp. Vì vậy CLI dùng `score_response(prompt, response)`: scorer xác định, không nhận nhãn `chosen/rejected`, kết hợp độ phủ token với prompt, lexical diversity và độ dài có giới hạn. Đây là baseline tái lập được trên CPU, không phải log-probability từ language model.

## 3. Kết quả đánh giá

### Quality gate

```text
20 passed
ruff check src tests: All checks passed
mypy src: Success: no issues found in 8 source files
```

### Kết quả huấn luyện

| Chỉ số | Giá trị |
|---|---:|
| Method | DPO (CPU mock trainer) |
| Epochs | 3 |
| Total steps | 30 |
| Initial loss | 0.6565 |
| Final loss | 0.6312 |
| Loss reduction | 0.0253 |

### Kết quả validation

| Chỉ số | Giá trị |
|---|---:|
| Train examples | 19 |
| Validation examples | 5 |
| Pairwise accuracy | 60.0% |
| Mean chosen score | 0.5572 |
| Mean rejected score | 0.5495 |
| Preference margin | +0.0077 |
| Safety prompts | 4 prompts, manual review required |

Nội dung `outputs/metrics.json` sau khi chạy:

```json
{
  "mean_chosen_score": 0.5572,
  "mean_rejected_score": 0.5495,
  "pairwise_accuracy": 0.6,
  "preference_margin": 0.0077,
  "safety_regression_prompts_checked": 4,
  "safety_regression_status": "manual_review_required",
  "total_examples": 5,
  "train_examples": 19
}
```

Ví dụ validation: với prompt về PCA, scorer ưu tiên câu trả lời `chosen` (0.4575) hơn `rejected` (0.3550). Tuy nhiên, kết quả tổng chỉ 60.0%, phản ánh đúng giới hạn của heuristic và không phải accuracy được gán cứng.

## 4. Thảo luận và failure modes

Điểm tốt là validation dữ liệu, split chống leakage, DPO/ORPO loss, tie handling và kiểm tra type/lint/test đều hoạt động. Artefact sinh ra được đặt trong `outputs/`, vốn đã được gitignore.

- **Failure mode 1 — lexical-overlap bias**: hai câu trả lời rejected có nhiều từ khóa trùng prompt hơn nên CPU scorer chấm cao hơn câu chosen. Heuristic không hiểu đúng/sai về mặt kiến thức.
- **Failure mode 2 — semantic misconception**: các phát biểu sai nhưng có thuật ngữ ML hợp lý (ví dụ nhầm chức năng batch normalization) khó bị phát hiện nếu không dùng model có năng lực ngữ nghĩa.
- **Safety**: 4 regression prompts được liệt kê từ `docs/regression_prompts.md`, nhưng chưa có model tạo câu trả lời để chấm tự động. Chúng được ghi là `manual_review_required`, không báo pass giả.

Để có kết quả dùng cho production, bước tiếp theo là thay CPU heuristic bằng sequence log-probability của policy/reference model, rồi chạy cùng validation split và rubric safety thực tế.
