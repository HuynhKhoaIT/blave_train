"""
config.py — Cấu hình tập trung cho việc train LoRA BLIP-2.

ĐÂY LÀ FILE DUY NHẤT BẠN CẦN CHỈNH giữa bản demo (Colab) và bản thật (GPU chính).
Code train (train.py) đọc toàn bộ thông số từ đây, nên không cần sửa code.

Cách dùng:
  - Demo nhanh trên Colab T4 16GB : đặt PROFILE = "demo"
  - Train đầy đủ trên GPU mạnh     : đặt PROFILE = "full"
"""

# Chọn 1 trong 2: "demo" hoặc "full"
PROFILE = "demo"

# ---- Đường dẫn dữ liệu (chỉnh cho đúng máy của bạn) ----
PATHS = {
    "train_json": "data/train_converted.json",   # file đã qua prepare_data.py
    "val_json":   "data/val_converted.json",      # (tuỳ chọn) chạy convert cho val khi cần
    "image_dir":  "data/vizwiz/train",             # ảnh train sau khi unzip train.zip
    "val_image_dir": "data/vizwiz/val",            # ảnh val sau khi unzip val.zip
}

# ---- Mô hình gốc trên Hugging Face ----
BASE_MODEL = "Salesforce/blip2-opt-2.7b"

# ---- Hai hồ sơ cấu hình ----
PROFILES = {
    # Bản demo: nhỏ + nhanh, chỉ để kiểm chứng pipeline chạy thông.
    # T4 16GB đủ chạy fp16 không cần qlora → tránh phụ thuộc bitsandbytes
    # (hay xung đột phiên bản CUDA trên Colab).
    "demo": {
        "use_qlora": True,
        "prepare_kbit": False,     # T4 16GB: TẮT để tránh OOM (cast fp16→fp32 ngốn 2x VRAM)
        "batch_size": 2,
        "grad_accum": 2,           # batch hiệu dụng = 2*2 = 4
        "num_epochs": 5,           # chỉ 5 epoch để xem có lỗi không
        "max_train_samples": 200,  # CHỈ lấy 200 mẫu
        "max_val_samples": 50,     # eval nhanh khi demo
        "eval_every": 1,           # eval mỗi epoch
        "lr": 1e-4,
        "save_every": 1,
        "output_dir": "./blip2_lora_demo",
    },
    # Bản thật: train đủ như bài báo (epoch 40), toàn bộ dữ liệu
    "full": {
        "use_qlora": True,        # GPU mạnh (>=24GB) không cần lượng tử hoá
        "prepare_kbit": True,      # GPU ≥24GB: BẬT để khớp BLaVe-CoT (gradient ckpt + LayerNorm fp32)
        "batch_size": 8,
        "grad_accum": 2,           # batch hiệu dụng = 16
        "num_epochs": 40,          # đúng checkpoint tốt nhất trong bài báo
        "max_train_samples": None, # None = dùng toàn bộ
        "max_val_samples": None,   # None = dùng toàn bộ val set
        "eval_every": 1,           # eval mỗi epoch để theo dõi overfit
        "lr": 1e-4,
        "save_every": 10,          # lưu checkpoint mỗi 10 epoch
        "output_dir": "./blip2_lora_full",
    },
}

# ---- Cấu hình LoRA (khớp finetune_blip2.py của BLaVe-CoT: q_proj/k_proj của OPT decoder) ----
LORA = {
    # r=16, alpha=32, dropout=0.05 — y hệt finetune_blip2.py
    "r": 16,
    "lora_alpha": 32,           # giữ tỉ lệ alpha = 2*r
    "lora_dropout": 0.05,
    # q_proj, k_proj là Query/Key projection của OPT decoder (language model trong BLIP-2).
    # Nếu báo "target modules not found", chạy inspect_model.py để tìm tên đúng.
    "target_modules": ["q_proj", "k_proj"],
}

# ---- Độ dài token ----
MAX_PROMPT_LEN = 32
MAX_ANSWER_LEN = 16


def get_active_config():
    """Trả về dict cấu hình đang được chọn, gộp PATHS + LORA + profile."""
    if PROFILE not in PROFILES:
        raise ValueError(f"PROFILE='{PROFILE}' không hợp lệ. Chọn: {list(PROFILES)}")
    cfg = dict(PROFILES[PROFILE])
    cfg["profile_name"] = PROFILE
    cfg["base_model"] = BASE_MODEL
    cfg["paths"] = PATHS
    cfg["lora"] = LORA
    cfg["max_prompt_len"] = MAX_PROMPT_LEN
    cfg["max_answer_len"] = MAX_ANSWER_LEN
    return cfg


if __name__ == "__main__":
    import json
    print(f"PROFILE đang chọn: {PROFILE}\n")
    print(json.dumps(get_active_config(), indent=2, ensure_ascii=False))
