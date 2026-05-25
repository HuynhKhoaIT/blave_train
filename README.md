# Fine-tune LoRA BLIP-2 trên VizWiz (cho pipeline BLaVe-CoT)

Bộ code này huấn luyện adapter LoRA cho BLIP-2 ở bước **sinh đáp án** — tái tạo
lại bản `blip2_vqa_finetuned_epoch_40` mà bài báo dùng, vì bản gốc chỉ phát hành
trên Baidu (khó tải ngoài Trung Quốc).

> **Lưu ý phạm vi:** Đây CHỈ là bước sinh đáp án (1 trong các tầng của BLaVe-CoT).
> Các tầng grounding (PolyFormer), suy luận/explanation (LLM), so nghĩa (MiniLM)
> nằm ngoài bộ code này.

---

## Cấu trúc file

| File | Vai trò |
|------|---------|
| `config.py` | **Chỉnh duy nhất ở đây.** Mọi thông số (demo/full, batch, epoch, đường dẫn). |
| `prepare_data.py` | Chuyển Annotations VizWiz → `train.json` (majority vote, lọc unanswerable). |
| `model.py` | Dựng BLIP-2 + gắn LoRA (đóng băng vision & LM, chỉ train Q/K của Q-Former). |
| `train.py` | Vòng huấn luyện chính. Đọc toàn bộ từ `config.py`. |
| `infer.py` | Chạy thử sau train, so sánh đáp án trước/sau fine-tune. |
| `inspect_model.py` | **Công cụ gỡ lỗi** khi báo "target modules not found". |
| `demo_colab.ipynb` | Notebook chạy demo nhanh trên Colab. |
| `requirements.txt` | Phiên bản thư viện đã ghim. |

---

## Quy trình 2 giai đoạn

### Giai đoạn A — Demo nhỏ trên Colab (kiểm chứng pipeline)

Mục tiêu: xác nhận code chạy thông từ đầu đến cuối, KHÔNG nhằm mô hình tốt.

1. Mở `demo_colab.ipynb` trên Google Colab, bật GPU (T4).
2. Chạy lần lượt các cell: cài thư viện → tải code → tải dữ liệu (200 mẫu) →
   train 2 epoch → thử inference.
3. Trong `config.py` đảm bảo `PROFILE = "demo"`.
4. Nếu chạy hết và xuất ra `./blip2_lora_demo/final`, pipeline ĐÚNG. Bàn giao.

### Giai đoạn B — Train đầy đủ trên GPU chính

Người chạy GPU mạnh chỉ cần:

1. Cài môi trường:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải dữ liệu VizWiz (KHÔNG qua Baidu — link chính thức Colorado):
   ```bash
   mkdir -p data/vizwiz && cd data/vizwiz
   wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip      # ~10.5GB
   wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip        # ~3.7GB
   wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip
   unzip "*.zip"
   find . -name "._*" -delete    # dọn file rác nếu có
   cd ../..
   ```
   > Chỉ cần `train.zip` + `Annotations.zip` để train. `val.zip` để đánh giá.
   > Bỏ qua `test.zip` (không có đáp án công khai).
3. Tạo `train_converted.json` (và val tương tự):
   ```bash
   python prepare_data.py \
       --raw data/vizwiz/Annotations/train.json \
       --out data/train_converted.json

   python prepare_data.py \
       --raw data/vizwiz/Annotations/val.json \
       --out data/val_converted.json
   ```
   > Đường dẫn output phải khớp với `PATHS["train_json"]` / `PATHS["val_json"]` trong `config.py`.
   > Đừng ghi đè lên `train.json` raw — train.py không đọc được định dạng đó.
4. Trong `config.py`, đặt `PROFILE = "full"` và kiểm tra đường dẫn trong `PATHS`
   khớp với nơi đã giải nén (đặc biệt `image_dir`).
5. Train:
   ```bash
   python train.py
   ```
   Checkpoint lưu mỗi 10 epoch; bản cuối ở `./blip2_lora_full/final`.

---

## Phần cứng & cấu hình tương ứng

| VRAM | Đặt trong config | Ghi chú |
|------|------------------|---------|
| ≥ 24GB | `use_qlora: False`, batch 8 | Thoải mái nhất (profile "full" mặc định) |
| 16GB | `use_qlora: False`, batch 2-4 + grad_accum | Vẫn ổn nếu giảm batch |
| 8-12GB | `use_qlora: True`, batch 1-2 | Cần bitsandbytes; chậm hơn |

Inference (chạy thử) nhẹ hơn train nhiều: ~6-8GB là đủ.

---

## Gỡ lỗi thường gặp

**"target modules not found" khi gắn LoRA**
Tên lớp Query/Key của Q-Former khác giữa các phiên bản `transformers`. Chạy:
```bash
python inspect_model.py
```
Nó in ra tên lớp thật. Cập nhật `LORA["target_modules"]` trong `config.py` cho khớp.
Đây là lý do `requirements.txt` ghim `transformers==4.40.0`.

**Hết VRAM (CUDA out of memory)**
Giảm `batch_size`, tăng `grad_accum` để giữ batch hiệu dụng. Hoặc bật `use_qlora`.

**`bitsandbytes` lỗi trên GPU chính**
Thư viện này kén CUDA. Nếu GPU mạnh, đặt `use_qlora: False` để né hẳn — không cần nó.

**Ảnh không tồn tại**
Kiểm tra `image_dir` trong config trỏ đúng thư mục ảnh đã giải nén. Tên thư mục
con sau khi unzip có thể khác (vd `train/` vs `images/`).

---

## Kiểm tra mô hình sau train

Đừng tin ngay — chạy `infer.py` với `--compare` để xem trước/sau fine-tune:
```bash
python infer.py --adapter ./blip2_lora_full/final \
    --image data/vizwiz/val/<tên_ảnh>.jpg \
    --question "What is this?" --compare
```
Nếu đáp án sau fine-tune sát hơn, ngắn gọn, đúng vật thể hơn → thành công.

---

## Adapter đầu ra

File `adapter_model.safetensors` chỉ vài chục MB (không phải vài GB), vì LoRA chỉ
lưu các ma trận hạng thấp gắn thêm — đúng như bản `epoch_40` gốc (20MB). Để dùng
lại: nạp BLIP-2 gốc rồi gắn adapter qua `PeftModel.from_pretrained` (xem `infer.py`).
