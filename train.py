"""
train.py — Vòng huấn luyện chính cho LoRA BLIP-2 trên VizWiz.

KHÔNG cần sửa file này. Mọi thông số nằm trong config.py.

Cách dùng:
  1. Sửa PROFILE trong config.py ("demo" hoặc "full")
  2. Chạy:  python train.py

Kết quả: adapter LoRA (~20MB) được lưu vào output_dir trong config.
"""
# Load config -> Load model + LoRA -> Load dataset -> DataLoader chia batch -> Forward -> Tính loss -> Backward -> Update LoRA weights -> Save adapter
import os
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import get_active_config
from model import build_model_for_training

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class VizWizDataset(Dataset):
    def __init__(self, json_path, image_dir, processor, cfg, max_samples=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.image_dir = image_dir
        self.processor = processor
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    #Mỗi lần DataLoader cần 1 sample: hàm này sẽ chạy
    def __getitem__(self, idx):
        it = self.data[idx]
        img_path = os.path.join(self.image_dir, it["image"])
        image = Image.open(img_path).convert("RGB")

        # Weighted sampling từ 10 câu trả lời (khớp finetune_blip2.py của BLaVe-CoT).
        # Nếu data cũ chỉ có field "answer" thì fallback dùng nó.
        if it.get("answers"):
            counts = Counter(it["answers"])
            answer = random.choices(
                it["answers"],
                weights=[counts[a] for a in it["answers"]],
            )[0]
        else:
            answer = it["answer"]

        # BLaVe-CoT style: input = ảnh + câu hỏi (không có prompt template "Question:...Answer:")
        #                  labels = TOKENS CỦA ANSWER (không lẫn prompt → tránh prompt leaking).
        # Tách image_processor và tokenizer thủ công để đảm bảo padding nhất quán
        # (processor() của BLIP-2 không honor padding="max_length" cho text → lỗi stack).
        max_len = self.cfg["max_prompt_len"]    # dùng chung max_len cho cả question và answer (khớp BLaVe-CoT = 32)
        tokenizer = self.processor.tokenizer

        # Ảnh
        pixel_values = self.processor.image_processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Câu hỏi (input)
        q_enc = tokenizer(
            it["question"],
            return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_len,
        )
        input_ids = q_enc.input_ids.squeeze(0)
        attention_mask = q_enc.attention_mask.squeeze(0)

        # Đáp án (labels) — không thêm special token để khớp BLaVe-CoT,
        # pad_token bị mask thành -100 để không tính loss trên padding.
        a_enc = tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_len,
            add_special_tokens=False,
        )
        labels = a_enc.input_ids.squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """Chạy 1 vòng val: tắt dropout, không tính grad, trả về loss trung bình."""
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
def main():
    cfg = get_active_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"PROFILE          : {cfg['profile_name']}")
    print(f"Device           : {device}")
    print(f"QLoRA (4-bit)    : {cfg['use_qlora']}")
    print(f"prepare_kbit     : {cfg.get('prepare_kbit', False)}")
    print(f"Batch size       : {cfg['batch_size']}  x grad_accum {cfg['grad_accum']}"
          f"  = effective {cfg['batch_size'] * cfg['grad_accum']}")
    print(f"Epochs           : {cfg['num_epochs']}")
    print(f"Max train samples: {cfg['max_train_samples']}")
    print("=" * 60)

    if device == "cpu":
        print("⚠ CẢNH BÁO: Không tìm thấy GPU. Train trên CPU sẽ cực kỳ chậm.")

    # --- Mô hình + LoRA ---
    model, processor = build_model_for_training(cfg)
    # Model 4-bit (qlora) đã được đặt device qua device_map="auto" lúc nạp;
    # gọi .to() trên model 4-bit sẽ raise ValueError.
    if not cfg["use_qlora"]:
        model.to(device)
    model.train()

    # --- Dữ liệu ---
    train_path = cfg["paths"]["train_json"]
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Không thấy {train_path}. Chạy prepare_data.py trước để tạo file này."
        )

    dataset = VizWizDataset(
        train_path, cfg["paths"]["image_dir"], processor, cfg,
        max_samples=cfg["max_train_samples"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    print(f"Số mẫu train: {len(dataset)}  |  Số batch/epoch: {len(loader)}")

    # --- Dữ liệu validation (tuỳ chọn) ---
    val_loader = None
    val_path = cfg["paths"].get("val_json")
    val_img_dir = cfg["paths"].get("val_image_dir")
    if val_path and os.path.exists(val_path):
        val_dataset = VizWizDataset(
            val_path, val_img_dir, processor, cfg,
            max_samples=cfg.get("max_val_samples"),
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        print(f"Số mẫu val  : {len(val_dataset)}  |  Số batch eval : {len(val_loader)}")
    else:
        print(f"(Bỏ qua validation — không thấy {val_path}. Chạy prepare_data.py cho val nếu cần.)")

    # --- Optimizer (chỉ tham số LoRA có requires_grad=True) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
    )

    accum = cfg["grad_accum"]
    eval_every = cfg.get("eval_every", 1)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # --- Vòng huấn luyện ---
    for epoch in range(cfg["num_epochs"]):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            running += loss.item() * accum

            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = running / max(len(loader), 1)

        # --- Validation ---
        val_loss = None
        if val_loader is not None and (epoch + 1) % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)

        if val_loss is not None:
            print(f"Epoch {epoch + 1}/{cfg['num_epochs']}  "
                  f"train_loss = {train_loss:.4f}  |  val_loss = {val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{cfg['num_epochs']}  train_loss = {train_loss:.4f}")

        # Lưu checkpoint định kỳ
        if (epoch + 1) % cfg["save_every"] == 0:
            ckpt = os.path.join(cfg["output_dir"], f"epoch_{epoch + 1}")
            model.save_pretrained(ckpt)
            print(f"  ↳ đã lưu checkpoint: {ckpt}")

    # --- Lưu bản cuối ---
    final = os.path.join(cfg["output_dir"], "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ Hoàn tất. Adapter cuối lưu tại: {final}")
    print("  (file adapter_model.safetensors chỉ vài chục MB — đúng như bản gốc)")


if __name__ == "__main__":
    main()
